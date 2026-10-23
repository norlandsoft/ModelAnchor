/* ModelAnchor 前端：只依赖 /api/*，没有构建步骤。 */

const state = {
  library: null,
  tasks: [],
  search: { q: "", source: "all", layout: "all", results: [], errors: [], loading: false, ran: false },
  expanded: new Set(),
  repos: {},
  repoErrors: {},
  finishedSeen: 0,
};

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/* ------------------------------------------------------------------ utils */
function fmtBytes(n) {
  if (n === null || n === undefined) return "—";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let value = Number(n);
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${i === 0 ? value.toFixed(0) : value.toFixed(value < 10 ? 2 : 1)} ${units[i]}`;
}

function fmtSpeed(bps) {
  return bps > 0 ? `${fmtBytes(bps)}/s` : "—";
}

function fmtEta(seconds) {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "—";
  let s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
}

function fmtTime(ts) {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleString("zh-CN", { hour12: false });
}

function compactNum(n) {
  if (!n) return "0";
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

function esc(value) {
  return String(value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function showToast(message, type = "") {
  const el = $("#toast");
  el.textContent = message;
  el.className = `toast ${type}`.trim();
  el.hidden = false;
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => { el.hidden = true; }, 4200);
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const raw = await res.text();
  let data = null;
  try { data = raw ? JSON.parse(raw) : null; } catch { data = raw; }
  if (!res.ok) {
    const detail = data && data.detail ? data.detail : res.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

const KIND_LABEL = {
  ckpt: "ckpt",
  gguf: "gguf",
  safetensors: "safetensors",
  folder: "目录",
  other: "其他",
};

const TASK_LABEL = {
  queued: "排队中",
  running: "进行中",
  verifying: "校验中",
  done: "已完成",
  failed: "失败",
  canceled: "已取消",
};

const LAYOUT_LABEL = {
  single: "单文件",
  diffusers: "Diffusers 目录",
  lora: "LoRA",
  unknown: "未知",
};
const SOURCE_LABEL = { hf: "HF 镜像", modelscope: "ModelScope" };
const repoKey = (r) => `${r.source}:${r.repo_id}`;
const dirSlug = (repoId) => repoId.replace(/\//g, "_");

/* ------------------------------------------------------------------ render */
function renderStats() {
  const lib = state.library;
  if (!lib) return;
  const modelFiles = lib.models.filter((m) => !m.partial && !m.is_dir);
  // 分片 .part 是稀疏文件，按真实已下载字节算，避免占用空间虚高
  const used = lib.models.reduce(
    (sum, m) => sum + (m.partial ? (m.done_bytes ?? 0) : m.size_bytes),
    0
  );
  $("#stat-count").textContent = String(modelFiles.length);
  $("#stat-size").textContent = fmtBytes(used);
  $("#stat-free").textContent = fmtBytes(lib.disk_free_bytes);
  $("#stat-base").textContent = (lib.hf_base || "").replace(/^https?:\/\//, "") || "—";
  $("#model-dir").textContent = lib.model_dir;
  $("#model-dir").title = `${lib.model_dir}\n来源：${lib.model_dir_source}`;

  const banner = $("#banner");
  if (lib.warning) {
    banner.hidden = false;
    banner.innerHTML = `<span>⚠️</span><span>${esc(lib.warning)}</span>`;
  } else {
    banner.hidden = true;
  }
  if (!lib.exists) $("#library-hint").textContent = "目录尚不存在，开始下载时会自动创建";
  else if (!lib.writable) $("#library-hint").textContent = "目录不可写，请检查权限";
  else $("#library-hint").textContent = "";
}

function renderTasks() {
  const panel = $("#tasks-panel");
  const tasks = state.tasks;
  panel.hidden = tasks.length === 0;
  if (tasks.length === 0) return;

  $("#tasks").innerHTML = tasks
    .map((job) => {
      const pct = job.percent === null ? null : Math.min(job.percent, 100);
      const indeterminate = pct === null && ["queued", "running"].includes(job.state);
      const barStyle = pct === null ? "" : `style="width:${pct}%"`;
      const meta = [];
      if (job.total_bytes) meta.push(`${fmtBytes(job.bytes_done)} / ${fmtBytes(job.total_bytes)}`);
      else if (job.bytes_done) meta.push(fmtBytes(job.bytes_done));
      if (job.kind === "repo" && job.files_total > 1) {
        meta.push(`${job.files_done}/${job.files_total} 文件`);
      }
      if (job.state === "running" && job.speed_bps > 0) {
        meta.push(fmtSpeed(job.speed_bps));
        if (job.eta_seconds !== null) meta.push(`剩余 ${fmtEta(job.eta_seconds)}`);
      }
      if (job.source) meta.push(`<span class="mono-dim">${esc(job.source)}</span>`);
      if (job.current_file) meta.push(`<span class="mono-dim">${esc(job.current_file)}</span>`);
      if (job.error) meta.push(`<span class="err">${esc(job.error)}</span>`);

      let actions = "";
      if (["queued", "running"].includes(job.state)) {
        actions = `<button class="btn ghost small" data-action="cancel" data-id="${job.id}">取消</button>`;
      } else if (job.state === "failed") {
        actions =
          `<button class="btn ghost small" data-action="retry" data-id="${job.id}">重试</button>` +
          `<button class="btn ghost small" data-action="forget" data-id="${job.id}">移除</button>`;
      } else {
        actions = `<button class="btn ghost small" data-action="forget" data-id="${job.id}">移除</button>`;
      }

      const kindTag = { file: "单文件", repo: "整目录", merge: "合并" }[job.kind] || job.kind;
      return `
        <div class="job">
          <div class="job-top">
            <span class="job-name" title="${esc(job.url || job.filename)}">${esc(job.filename)}</span>
            <span class="job-tags"><span class="badge">${kindTag}</span><span class="job-state ${job.state}">${TASK_LABEL[job.state] || job.state}</span></span>
          </div>
          <div class="bar ${indeterminate ? "indeterminate" : ""}"><i ${barStyle}></i></div>
          <div class="job-meta">
            ${meta.map((m) => `<span>${m}</span>`).join("")}
            <span class="job-actions">${actions}</span>
          </div>
        </div>`;
    })
    .join("");
}

function renderLibrary() {
  const lib = state.library;
  if (!lib) return;
  const tbody = $("#library-body");
  const models = lib.models;
  $("#library-empty").hidden = models.length > 0;
  tbody.innerHTML = models
    .map((m) => {
      let kind = KIND_LABEL[m.kind] || m.kind;
      let badgeClass = m.kind;
      if (m.partial) { kind = "未完成"; badgeClass = "partial"; }
      const pct = m.partial && m.done_bytes != null && m.total_bytes
        ? Math.floor((m.done_bytes / m.total_bytes) * 100)
        : null;
      const sizeText = pct !== null
        ? `${fmtBytes(m.done_bytes)} / ${fmtBytes(m.total_bytes)}`
        : fmtBytes(m.size_bytes);
      let actions = "";
      if (m.is_dir) {
        actions = `<button class="btn ghost small" data-action="merge" data-name="${esc(m.filename)}">合并</button>
           <button class="btn ghost small danger" data-action="delete" data-name="${esc(m.filename)}">删除</button>`;
      } else if (m.partial) {
        const action = m.resume_url ? "resume" : "resume-unknown";
        actions = `<button class="btn primary small" data-action="${action}" data-name="${esc(m.filename)}">继续</button>
           <button class="btn ghost small danger" data-action="delete" data-name="${esc(m.filename)}">删除</button>`;
      } else {
        actions = `<button class="btn ghost small danger" data-action="delete" data-name="${esc(m.filename)}">删除</button>`;
      }
      return `
        <tr>
          <td class="name">${esc(m.filename)}${m.is_dir ? '<span class="tree-mark">/</span>' : ""}</td>
          <td><span class="badge ${badgeClass}">${esc(kind)}${pct !== null ? ` ${pct}%` : ""}</span></td>
          <td class="num">${sizeText}</td>
          <td class="muted">${fmtTime(m.modified_at)}</td>
          <td class="right actions">${actions}</td>
        </tr>`;
    })
    .join("");
}

/* ------------------------------------------------------------------ 检索 */
function renderSearch() {
  const box = $("#search-results");
  const errors = $("#search-errors");
  const { results, errors: errs, loading, ran, q } = state.search;

  if (errs && errs.length) {
    errors.hidden = false;
    errors.innerHTML = `<span>${esc(errs.join(" · "))}</span>
      <button class="btn ghost small" data-action="retry-search" type="button">重试</button>`;
  } else {
    errors.hidden = true;
  }

  if (loading) {
    box.innerHTML = `<p class="empty">正在检索 … 镜像站偶尔较慢，最多等 30 秒左右。</p>`;
    return;
  }
  if (!ran) {
    box.innerHTML = `<p class="empty">输入关键词检索 HuggingFace 镜像与 ModelScope。</p>`;
    return;
  }
  if (results.length === 0) {
    box.innerHTML = `<p class="empty">没有找到「${esc(q)}」的结果，换个关键词或放宽类型过滤试试。</p>`;
    return;
  }

  box.innerHTML = results.map((hit) => resultCard(hit)).join("");
}

function resultCard(hit) {
  const key = repoKey(hit);
  const expanded = state.expanded.has(key);
  const detail = state.repos[key];
  const error = state.repoErrors[key];
  const localDir = dirSlug(hit.repo_id);
  const installed = (state.library?.models || []).some((m) => m.filename === localDir && m.is_dir);

  return `
    <article class="result" data-key="${esc(key)}">
      <div class="result-head">
        <div class="result-line">
          <span class="badge src-${hit.source}">${SOURCE_LABEL[hit.source] || hit.source}</span>
          <strong class="result-id">${esc(hit.repo_id)}</strong>
          <span class="badge ${hit.layout}">${LAYOUT_LABEL[hit.layout] || hit.layout}</span>
          ${installed ? '<span class="badge ok">已下载</span>' : ""}
        </div>
        <div class="result-meta">
          <span>⬇ ${compactNum(hit.downloads)}</span>
          <span>♥ ${compactNum(hit.likes)}</span>
          ${hit.total_bytes ? `<span>${fmtBytes(hit.total_bytes)}</span>` : ""}
          ${hit.pipeline_tag ? `<span class="tag">${esc(hit.pipeline_tag)}</span>` : ""}
        </div>
        <div class="result-actions">
          <button class="btn ghost small" data-action="toggle">${expanded ? "收起" : "查看文件"}</button>
          <a class="btn ghost small" href="${esc(hit.page_url)}" target="_blank" rel="noreferrer">源站</a>
        </div>
      </div>
      ${expanded ? `<div class="result-body">${resultBody(hit, detail, error, installed)}</div>` : ""}
    </article>`;
}

function resultBody(hit, detail, error, installed) {
  const localDir = dirSlug(hit.repo_id);
  if (error) return `<p class="empty err">${esc(error)}</p>`;
  if (!detail) return `<p class="empty">读取文件列表 …</p>`;

  const weights = detail.files.filter((f) => /\.(safetensors|ckpt|gguf)$/i.test(f.path));
  const isWeightOnly = detail.layout === "single" || detail.layout === "lora";
  const files = isWeightOnly ? weights : detail.files;
  const size = detail.total_bytes ? fmtBytes(detail.total_bytes) : "大小未知";

  const toolbar = [];
  if (detail.layout === "diffusers") {
    toolbar.push(
      `<button class="btn primary small" data-action="download-repo" data-source="${hit.source}" data-repo="${esc(hit.repo_id)}">下载整个目录</button>`
    );
    toolbar.push(`<span class="hint">${files.length} 个文件 · ${size}</span>`);
    if (installed) {
      toolbar.push(`<span class="badge ok">已下载到 ${esc(localDir)}/</span>`);
      toolbar.push(
        `<button class="btn ghost small" data-action="goto-merge" data-dir="${esc(localDir)}">去合并</button>`
      );
    }
  } else if (detail.layout === "single") {
    toolbar.push(`<span class="hint">单文件模型，按需下载：</span>`);
  } else if (detail.layout === "lora") {
    toolbar.push(`<span class="hint">LoRA / 适配器权重，选一个版本下载：</span>`);
  }

  const rows = files
    .slice(0, 200)
    .map((file) => {
      const isWeight = /\.(safetensors|ckpt|gguf)$/i.test(file.path);
      const canDownload = file.downloadable !== false;
      const btn = canDownload
        ? `<button class="btn ghost small" data-action="download-file" data-url="${esc(file.url || "")}" data-name="${esc(file.path.split("/").pop())}">下载</button>`
        : `<span class="muted">跳过</span>`;
      return `<tr class="${isWeight ? "w" : ""}">
        <td class="path">${esc(file.path)}</td>
        <td class="num">${file.size ? fmtBytes(file.size) : "—"}</td>
        <td class="right">${btn}</td>
      </tr>`;
    })
    .join("");

  return `
    <div class="result-toolbar">${toolbar.join("")}</div>
    <div class="files"><table>
      <tbody>${rows}</tbody>
    </table></div>
    ${files.length > 200 ? `<p class="hint">仅显示前 200 个文件。</p>` : ""}`;
}

/* ------------------------------------------------------------------ 合并 */
function renderMergePanel() {
  const folders = (state.library?.models || []).filter((m) => m.is_dir);
  const panel = $("#merge-panel");
  panel.hidden = folders.length === 0;
  if (folders.length === 0) return;

  const select = $("#merge-dir");
  const current = select.value;
  const names = folders.map((f) => f.filename);
  select.innerHTML = names.map((n) => `<option value="${esc(n)}">${esc(n)}</option>`).join("");
  if (names.includes(current)) select.value = current;
  refreshMergeHint();
}

async function refreshMergeHint() {
  const dir = $("#merge-dir").value;
  const arch = $("#merge-arch").value;
  const hint = $("#merge-hint");
  if (!dir) { hint.textContent = ""; return; }
  hint.textContent = "读取组件 …";
  try {
    const plan = await api(`/api/merges/plan?source_dir=${encodeURIComponent(dir)}&arch=${encodeURIComponent(arch)}`);
    const comps = plan.components.map((c) => `${c.name}(${c.tensor_count})`).join(" · ");
    hint.textContent = `识别为 ${plan.arch} · ${plan.components.length} 个组件 · ${plan.total_tensors} 张量 · 约 ${fmtBytes(plan.total_bytes)}；${comps}`;
    $("#merge-name").placeholder = `留空则用 ${dir}.safetensors`;
  } catch (err) {
    hint.innerHTML = `<span class="err">${esc(err.message)}</span>`;
  }
}

/* ------------------------------------------------------------------ loaded */
async function loadLibrary() {
  state.library = await api("/api/library");
  renderStats();
  renderLibrary();
  renderMergePanel();
}

async function loadTasks() {
  const data = await api("/api/tasks");
  state.tasks = data.tasks;
  renderTasks();

  const finished = state.tasks.filter((t) => ["done", "failed", "canceled"].includes(t.state)).length;
  if (finished > state.finishedSeen) {
    state.finishedSeen = finished;
    await loadLibrary();
    if (state.search.ran) renderSearch();
  }
}

async function refreshAll() {
  try {
    await Promise.all([loadLibrary(), loadTasks()]);
    $("#status-pill").className = "pill ok";
    $("#status-pill").innerHTML = "<i></i>服务正常";
  } catch (err) {
    $("#status-pill").className = "pill bad";
    $("#status-pill").innerHTML = "<i></i>连接失败";
    showToast(`加载失败：${err.message}`, "err");
  }
}

/* ----------------------------------------------------------------- actions */
async function runSearch() {
  const q = $("#search-input").value.trim();
  const source = $("#source-select").value;
  const layout = $("#layout-select").value;
  if (!q) { showToast("先输入检索关键词", "err"); return; }

  state.search = { q, source, layout, results: [], errors: [], loading: true, ran: true };
  state.expanded.clear();
  state.repos = {};
  state.repoErrors = {};
  renderSearch();

  try {
    const data = await api(
      `/api/search?q=${encodeURIComponent(q)}&source=${encodeURIComponent(source)}&layout=${encodeURIComponent(layout)}&limit=12`
    );
    state.search.results = data.results;
    state.search.errors = data.errors || [];
    state.search.loading = false;
    renderSearch();
  } catch (err) {
    state.search.loading = false;
    state.search.errors = [err.message];
    renderSearch();
  }
}

async function toggleRepo(key) {
  if (state.expanded.has(key)) {
    state.expanded.delete(key);
    renderSearch();
    return;
  }
  state.expanded.add(key);
  renderSearch();
  if (state.repos[key] || state.repoErrors[key]) return;

  const hit = state.search.results.find((r) => repoKey(r) === key);
  if (!hit) return;
  try {
    const detail = await api(
      `/api/repos?source=${encodeURIComponent(hit.source)}&repo_id=${encodeURIComponent(hit.repo_id)}`
    );
    state.repos[key] = detail;
  } catch (err) {
    state.repoErrors[key] = err.message;
  }
  renderSearch();
}

async function startDownload(payload) {
  try {
    const job = await api("/api/downloads", { method: "POST", body: JSON.stringify(payload) });
    showToast(`开始下载 ${job.filename}`, "ok");
    await loadTasks();
  } catch (err) {
    showToast(err.message, "err");
  }
}

async function startRepoDownload(source, repoId) {
  try {
    const job = await api("/api/downloads/repo", {
      method: "POST",
      body: JSON.stringify({ source, repo_id: repoId }),
    });
    showToast(`开始下载 ${repoId}（${job.files_total} 个文件）`, "ok");
    await loadTasks();
  } catch (err) {
    showToast(err.message, "err");
  }
}

async function startMerge(dir, arch, name) {
  try {
    const job = await api("/api/merges", {
      method: "POST",
      body: JSON.stringify({ source_dir: dir, arch, output_name: name || null }),
    });
    showToast(`开始合并 → ${job.filename}`, "ok");
    await loadTasks();
  } catch (err) {
    showToast(err.message, "err");
  }
}

async function cancelTask(id) {
  try {
    await api(`/api/tasks/${id}/cancel`, { method: "POST" });
    await loadTasks();
  } catch (err) {
    showToast(err.message, "err");
  }
}

async function forgetTask(id) {
  try {
    await api(`/api/tasks/${id}`, { method: "DELETE" });
    await loadTasks();
  } catch (err) {
    showToast(err.message, "err");
  }
}

async function retryTask(id) {
  const job = state.tasks.find((t) => t.id === id);
  if (!job) return;
  if (job.kind === "file") await startDownload({ url: job.url, filename: job.filename });
  else if (job.kind === "merge") await startMerge(job.source_dir, "auto", job.filename);
  else showToast("整目录任务请回到检索结果重新点击下载", "err");
}

async function deleteModel(name) {
  if (!window.confirm(`确定删除 ${name}？`)) return;
  try {
    await api(`/api/library/${encodeURIComponent(name)}`, { method: "DELETE" });
    showToast(`已删除 ${name}`, "ok");
    await loadLibrary();
    if (state.search.ran) renderSearch();
  } catch (err) {
    showToast(err.message, "err");
  }
}

async function resumePartial(name) {
  const model = (state.library?.models || []).find((m) => m.filename === name);
  if (!model || !model.resume_url) return primeResume(name);
  await startDownload({
    url: model.resume_url,
    filename: name,
    sha256: model.resume_sha256 || undefined,
  });
}

function primeResume(name) {
  const base = name.replace(/\.part$/i, "");
  $("#filename-input").value = base;
  $("#url-input").focus();
  $("#install-form").scrollIntoView({ behavior: "smooth", block: "center" });
  showToast("请粘贴这个文件原来的下载链接；已填好文件名，提交后会从断点续传", "ok");
}

function gotoMerge(dir) {
  renderMergePanel();
  $("#merge-dir").value = dir;
  refreshMergeHint();
  $("#merge-panel").scrollIntoView({ behavior: "smooth", block: "center" });
  $("#merge-btn").focus();
}

/* ------------------------------------------------------------------ events */
function bindEvents() {
  $("#refresh-btn").addEventListener("click", refreshAll);

  $("#search-form").addEventListener("submit", (event) => {
    event.preventDefault();
    runSearch();
  });
  $(".search-hints").addEventListener("click", (event) => {
    const chip = event.target.closest("[data-q]");
    if (!chip) return;
    $("#search-input").value = chip.dataset.q;
    runSearch();
  });

  $("#source-select").addEventListener("change", () => { if (state.search.ran) runSearch(); });
  $("#layout-select").addEventListener("change", () => { if (state.search.ran) runSearch(); });

  $("#search-errors").addEventListener("click", (event) => {
    if (event.target.closest("[data-action='retry-search']")) runSearch();
  });

  $("#search-results").addEventListener("click", (event) => {
    const button = event.target.closest("[data-action]");
    if (!button) return;
    const action = button.dataset.action;
    const card = button.closest(".result");
    const key = card ? card.dataset.key : "";
    if (action === "toggle") toggleRepo(key);
    if (action === "download-repo") startRepoDownload(button.dataset.source, button.dataset.repo);
    if (action === "download-file") startDownload({ url: button.dataset.url, filename: button.dataset.name });
    if (action === "goto-merge") gotoMerge(button.dataset.dir);
  });

  $("#tasks").addEventListener("click", (event) => {
    const button = event.target.closest("[data-action]");
    if (!button) return;
    const { action, id } = button.dataset;
    if (action === "cancel") cancelTask(id);
    if (action === "forget") forgetTask(id);
    if (action === "retry") retryTask(id);
  });

  $("#clear-finished").addEventListener("click", async () => {
    const done = state.tasks.filter((t) => ["done", "failed", "canceled"].includes(t.state));
    for (const task of done) {
      try { await api(`/api/tasks/${task.id}`, { method: "DELETE" }); } catch { /* 忽略 */ }
    }
    state.finishedSeen = 0;
    await loadTasks();
  });

  $("#library-body").addEventListener("click", (event) => {
    const button = event.target.closest("[data-action]");
    if (!button) return;
    const action = button.dataset.action;
    const name = button.dataset.name;
    if (action === "delete") deleteModel(name);
    if (action === "merge") gotoMerge(name);
    if (action === "resume") resumePartial(name);
    if (action === "resume-unknown") primeResume(name);
  });

  $("#merge-dir").addEventListener("change", refreshMergeHint);
  $("#merge-arch").addEventListener("change", refreshMergeHint);
  $("#merge-btn").addEventListener("click", () => {
    const dir = $("#merge-dir").value;
    if (!dir) return;
    startMerge(dir, $("#merge-arch").value, $("#merge-name").value.trim());
  });

  $("#install-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const url = $("#url-input").value.trim();
    const filename = $("#filename-input").value.trim();
    const sha256 = $("#sha256-input").value.trim();
    if (!url) return;
    const payload = { url };
    if (filename) payload.filename = filename;
    if (sha256) payload.sha256 = sha256;
    startDownload(payload);
    $("#install-form").reset();
  });
}

/* -------------------------------------------------------------------- boot */
refreshAll();
bindEvents();
renderSearch();

// 支持 ?q=关键词 直接检索，方便收藏和分享
const deepLink = new URLSearchParams(location.search).get("q");
if (deepLink) {
  $("#search-input").value = deepLink;
  runSearch();
}
setInterval(() => {
  loadTasks().catch(() => {
    $("#status-pill").className = "pill bad";
    $("#status-pill").innerHTML = "<i></i>连接失败";
  });
}, 1500);

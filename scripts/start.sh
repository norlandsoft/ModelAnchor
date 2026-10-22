#!/usr/bin/env bash
# ModelAnchor 启动脚本
#   ./scripts/start.sh              前台启动（Ctrl-C 停止）
#   ./scripts/start.sh --background 后台启动，日志与 pid 写进 .run/
#   ./scripts/start.sh --reload     开发模式（改代码自动重载）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HOST="${MODELANCHOR_HOST:-127.0.0.1}"
PORT="${MODELANCHOR_PORT:-7990}"
PYTHON_BIN="${MODELANCHOR_PYTHON:-3.11}"
RUN_DIR="$ROOT/.run"
PID_FILE="$RUN_DIR/modelanchor.pid"
LOG_FILE="$RUN_DIR/modelanchor.log"

BACKGROUND=0
RELOAD=0
for arg in "$@"; do
  case "$arg" in
    -b|--background) BACKGROUND=1 ;;
    -r|--reload) RELOAD=1 ;;
    -h|--help) sed -n '2,5p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "未知参数：$arg" >&2; exit 2 ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "缺少 uv。安装：brew install uv  或  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "==> 创建虚拟环境 (.venv, python $PYTHON_BIN)"
  uv venv --python "$PYTHON_BIN" .venv
fi

echo "==> 同步依赖"
uv sync --quiet

mkdir -p "$RUN_DIR"

if [ "$BACKGROUND" -eq 1 ]; then
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "已经在运行（pid $(cat "$PID_FILE")），先执行 ./scripts/stop.sh" >&2
    exit 1
  fi
  echo "==> 后台启动 http://$HOST:$PORT"
  nohup .venv/bin/python -m uvicorn app.main:app \
    --host "$HOST" --port "$PORT" >>"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"

  for _ in $(seq 1 40); do
    if curl -fsS -m 1 "http://$HOST:$PORT/api/health" >/dev/null 2>&1; then
      echo "==> 已就绪：http://$HOST:$PORT  （日志 ${LOG_FILE}）"
      exit 0
    fi
    sleep 0.25
  done
  echo "启动超时，最近日志：" >&2
  tail -n 30 "$LOG_FILE" >&2 || true
  exit 1
fi

ARGS=(--host "$HOST" --port "$PORT")
if [ "$RELOAD" -eq 1 ]; then
  ARGS+=(--reload --reload-dir app --reload-dir web)
fi
echo "==> 前台启动 http://$HOST:$PORT"
exec .venv/bin/python -m uvicorn app.main:app "${ARGS[@]}"

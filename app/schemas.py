"""API 数据模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelFile(BaseModel):
    filename: str
    size_bytes: int
    modified_at: float
    kind: str  # ckpt / gguf / safetensors / folder / other
    partial: bool = False
    is_dir: bool = False


class LibraryResponse(BaseModel):
    model_dir: str
    model_dir_source: str
    hf_base: str = ""
    exists: bool
    writable: bool
    warning: str = ""
    disk_free_bytes: int
    disk_total_bytes: int
    models: list[ModelFile]


class CatalogEntry(BaseModel):
    id: str
    name: str
    filename: str
    kind: str
    repo: str
    url: str
    size_bytes: int | None = None
    note: str = ""
    installed: bool = False
    installed_bytes: int | None = None


class CatalogResponse(BaseModel):
    base: str
    entries: list[CatalogEntry]


class DownloadRequest(BaseModel):
    url: str | None = None
    filename: str | None = None
    sha256: str | None = None
    catalog_id: str | None = None


class RepoDownloadRequest(BaseModel):
    source: str = "hf"
    repo_id: str
    refresh: bool = False


class DownloadJobOut(BaseModel):
    id: str
    kind: str = "file"
    filename: str
    url: str = ""
    state: str  # queued / running / verifying / done / failed / canceled
    bytes_done: int
    total_bytes: int | None
    percent: float | None
    speed_bps: float
    eta_seconds: float | None
    error: str | None
    created_at: float
    started_at: float | None
    finished_at: float | None
    files_done: int = 0
    files_total: int = 1
    current_file: str = ""
    source: str = ""


class DownloadsResponse(BaseModel):
    jobs: list[DownloadJobOut]


class TasksResponse(BaseModel):
    tasks: list[DownloadJobOut]


class SearchHitOut(BaseModel):
    source: str
    repo_id: str
    downloads: int = 0
    likes: int = 0
    pipeline_tag: str = ""
    updated_at: float | None = None
    layout: str = "unknown"
    tags: list[str] = Field(default_factory=list)
    page_url: str = ""
    total_bytes: int = 0


class SearchResponse(BaseModel):
    results: list[SearchHitOut]
    errors: list[str] = Field(default_factory=list)


class RepoFileOut(BaseModel):
    path: str
    size: int | None = None
    downloadable: bool = True
    url: str = ""


class RepoInfoOut(BaseModel):
    source: str
    repo_id: str
    layout: str
    page_url: str
    total_bytes: int
    file_count: int
    files: list[RepoFileOut]


class MergeComponentOut(BaseModel):
    name: str
    prefix: str
    files: list[str]
    tensor_count: int
    bytes: int


class MergePlanOut(BaseModel):
    source_dir: str
    arch: str
    components: list[MergeComponentOut]
    total_bytes: int
    total_tensors: int


class MergeRequest(BaseModel):
    source_dir: str
    output_name: str | None = None
    arch: str = "auto"


class HealthResponse(BaseModel):
    app: str
    version: str
    status: str
    model_dir: str
    port: int
    uptime_seconds: float


class OkResponse(BaseModel):
    ok: bool = True
    detail: str = Field(default="")

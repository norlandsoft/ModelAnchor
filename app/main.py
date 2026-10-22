"""FastAPI 应用装配。"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import api
from .config import APP_NAME, VERSION, get_settings
from .downloader import DownloadManager
from .hubs import get_hubs
from .merger import MergeManager

log = logging.getLogger("modelanchor")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = app.state.settings
    try:
        settings.model_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # 沙盒 / 只读卷：先跑起来，页面上会显示不可写
        log.warning("模型目录不可用：%s（%s）", settings.model_dir, exc)
    log.info("模型目录：%s（来源 %s）", settings.model_dir, settings.model_dir_source)
    if settings.warning:
        log.warning("%s", settings.warning)
    log.info("服务地址：http://%s:%s", settings.host, settings.port)
    yield
    app.state.manager.shutdown()
    app.state.merges.shutdown()
    for hub in app.state.hubs.values():
        client = getattr(hub, "_client", None)
        if client is not None:
            client.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=APP_NAME,
        version=VERSION,
        description="为 DrawThings 下载、检索与合并模型的本地服务",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.manager = DownloadManager(settings)
    app.state.merges = MergeManager(settings)
    app.state.hubs = get_hubs(settings)
    app.state.started_at = time.time()

    static_dir = settings.web_dir / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    templates = Jinja2Templates(directory=str(settings.web_dir))
    app.include_router(api.router)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "model_dir": str(settings.model_dir),
                "model_dir_source": settings.model_dir_source,
                "version": VERSION,
            },
        )

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> HTMLResponse:
        svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><text y="26" font-size="26">⚓️</text></svg>'
        return HTMLResponse(svg, media_type="image/svg+xml")

    return app


app = create_app()


def main() -> None:
    """`python -m app` 的入口：方便不起 uvicorn 也能跑。"""
    import uvicorn

    settings = get_settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )

"""页面与基础接口的冒烟测试。"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def make_client(monkeypatch, model_dir: Path) -> TestClient:
    monkeypatch.setenv("MODELANCHOR_MODEL_DIR", str(model_dir))
    from app.config import get_settings

    get_settings.cache_clear()
    from app.main import create_app

    return TestClient(create_app())


def test_index_page_hides_official_catalog(monkeypatch, tmp_path: Path):
    client = make_client(monkeypatch, tmp_path)
    with client:
        response = client.get("/")
    assert response.status_code == 200
    assert "在线检索" in response.text
    assert "DrawThings 官方模型" not in response.text
    assert "合并导出" in response.text


def test_library_reports_hf_base(monkeypatch, tmp_path: Path):
    client = make_client(monkeypatch, tmp_path)
    with client:
        payload = client.get("/api/library").json()
    assert payload["model_dir"] == str(tmp_path)
    assert payload["hf_base"].startswith("http")
    assert payload["models"] == []


def test_tasks_starts_empty(monkeypatch, tmp_path: Path):
    client = make_client(monkeypatch, tmp_path)
    with client:
        assert client.get("/api/tasks").json() == {"tasks": []}


def test_delete_traversal_is_rejected(monkeypatch, tmp_path: Path):
    client = make_client(monkeypatch, tmp_path)
    with client:
        response = client.delete("/api/library/..%2F..%2Fetc%2Fpasswd")
    assert response.status_code in {400, 404}

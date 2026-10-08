"""Transport extensions: HEAD probing, status-aware streaming, transient retry.

Runs a real local ``http.server`` (stdlib, no network) so ``UrllibTransport``
is exercised through real ``urlopen`` calls, not a fake.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from organelleverse.core.errors import OrganelleExecutionError
from organelleverse.fetch._http import (
    HttpResponse,
    UrllibTransport,
    download_with_retry,
    get_with_retry,
)


class _Handler(BaseHTTPRequestHandler):
    routes: ClassVar[dict[str, Callable[[_Handler], None]]] = {}

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _dispatch(self) -> None:
        handler = self.routes.get(self.path)
        if handler is None:
            self.send_response(404)
            self.end_headers()
            return
        handler(self)

    def do_GET(self) -> None:
        self._dispatch()

    def do_HEAD(self) -> None:
        self._dispatch()

    def do_POST(self) -> None:
        self._dispatch()


@pytest.fixture
def server():
    _Handler.routes = {}
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd, _Handler.routes
    httpd.shutdown()
    thread.join(timeout=5)


def _base_url(httpd: HTTPServer) -> str:
    return f"http://127.0.0.1:{httpd.server_port}"


def test_head_returns_200_for_an_existing_resource(server) -> None:
    httpd, routes = server

    def ok(handler: _Handler) -> None:
        handler.send_response(200)
        handler.end_headers()

    routes["/exists"] = ok
    status = UrllibTransport().head(f"{_base_url(httpd)}/exists")
    assert status == 200


def test_head_returns_404_for_a_missing_resource_without_raising(server) -> None:
    httpd, _routes = server
    status = UrllibTransport().head(f"{_base_url(httpd)}/absent")
    assert status == 404


def test_get_response_never_raises_for_a_real_http_response(server) -> None:
    httpd, routes = server

    def forbidden(handler: _Handler) -> None:
        handler.send_response(403)
        body = b"blocked"
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    routes["/blocked"] = forbidden
    response = UrllibTransport().get_response(f"{_base_url(httpd)}/blocked")
    assert response == HttpResponse(status=403, body=b"blocked")


def test_download_to_path_streams_the_body_and_returns_status(server, tmp_path: Path) -> None:
    httpd, routes = server

    def ok(handler: _Handler) -> None:
        handler.send_response(200)
        body = b">seq\nACGT\n" * 1000
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    routes["/genome"] = ok
    dest = tmp_path / "genome.fa"
    status = UrllibTransport().download_to_path(f"{_base_url(httpd)}/genome", dest)
    assert status == 200
    assert dest.read_bytes().startswith(b">seq\n")


def test_download_to_path_writes_the_error_body_on_a_non_2xx_status(server, tmp_path: Path) -> None:
    httpd, routes = server

    def error(handler: _Handler) -> None:
        handler.send_response(500)
        body = b"server error"
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    routes["/broken"] = error
    dest = tmp_path / "broken.fa"
    status = UrllibTransport().download_to_path(f"{_base_url(httpd)}/broken", dest)
    assert status == 500
    assert dest.read_bytes() == b"server error"


def test_headers_are_merged_over_the_default_user_agent(server) -> None:
    httpd, routes = server
    seen: dict[str, str] = {}

    def capture(handler: _Handler) -> None:
        seen["user-agent"] = handler.headers.get("User-Agent", "")
        handler.send_response(200)
        handler.end_headers()

    routes["/ua"] = capture
    UrllibTransport().get(f"{_base_url(httpd)}/ua", headers={"User-Agent": "custom-agent/1.0"})
    assert seen["user-agent"] == "custom-agent/1.0"


def test_get_with_retry_retries_a_transient_status_then_succeeds(server) -> None:
    httpd, routes = server
    calls = {"n": 0}

    def flaky(handler: _Handler) -> None:
        calls["n"] += 1
        if calls["n"] < 3:
            handler.send_response(503)
            handler.end_headers()
            return
        handler.send_response(200)
        body = b"ok"
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    routes["/flaky"] = flaky
    response = get_with_retry(UrllibTransport(), f"{_base_url(httpd)}/flaky", timeout=5.0)
    assert response.status == 200
    assert response.body == b"ok"
    assert calls["n"] == 3


def test_get_with_retry_does_not_retry_a_404() -> None:
    calls = {"n": 0}

    class Stub:
        def get_response(self, url: str, **kwargs: object) -> HttpResponse:
            calls["n"] += 1
            return HttpResponse(status=404, body=b"")

    response = get_with_retry(Stub(), "http://example.invalid/x")
    assert response.status == 404
    assert calls["n"] == 1


def test_download_with_retry_exhausts_and_raises_network_transient(tmp_path: Path) -> None:
    class AlwaysBusy:
        def download_to_path(self, url: str, dest: Path, **kwargs: object) -> int:
            return 503

    with pytest.raises(OrganelleExecutionError) as raised:
        download_with_retry(
            AlwaysBusy(),
            "http://example.invalid/x",
            tmp_path / "out.fa",
            max_attempts=2,
            initial_delay=0.0,
        )
    assert raised.value.code == "network.transient"

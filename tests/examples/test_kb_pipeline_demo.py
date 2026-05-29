from __future__ import annotations

import httpx

from examples.kb_pipeline_demo import JOB_WAIT_SERVER_WINDOW, KBClient


def test_wait_for_job_continues_after_active_wait_timeout() -> None:
    calls: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        timeout_seconds = float(request.url.params["timeout_seconds"])
        calls.append(timeout_seconds)
        if len(calls) == 1:
            return httpx.Response(
                408,
                json={
                    "detail": {
                        "error_code": "wait_timeout",
                        "job_id": "job_build_demo",
                        "current_status": "running",
                        "message": "still running",
                    }
                },
            )
        return httpx.Response(
            200,
            json={"id": "job_build_demo", "status": "succeeded"},
        )

    client = KBClient("http://testserver", "")
    client._client.close()
    client._client = httpx.Client(
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    )
    try:
        final = client.wait_for_job("kb_demo", "job_build_demo", timeout_seconds=1800)
    finally:
        client.close()

    assert final["status"] == "succeeded"
    assert calls == [JOB_WAIT_SERVER_WINDOW, JOB_WAIT_SERVER_WINDOW]

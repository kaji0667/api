"""Replay private captured requests against loopback only; never submits a run."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from server import validate_answer


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def local_origin(value: str) -> str:
    url = urllib.parse.urlsplit(value)
    if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}
            or url.username is not None or url.password is not None
            or url.path not in {"", "/"} or url.query or url.fragment):
        raise ValueError("Replay endpoint must be a loopback HTTP origin")
    try:
        port = url.port
    except ValueError:
        raise ValueError("Replay endpoint has an invalid port") from None
    host = "[::1]" if url.hostname == "::1" else "127.0.0.1"
    return "http://" + host + (":" + str(port) if port is not None else "")


def response_valid(request: dict[str, Any], response: Any) -> bool:
    if not isinstance(response, dict) or any(
        response.get(name) != request.get(name)
        for name in ("protocol_version", "request_id", "item_id")
    ):
        return False
    try:
        answer = response["answer"]
        validate_answer(request, answer)
        constraint = request["response_constraint"]
        kind = constraint["type"]
        if kind in {"enum", "single_choice"}:
            if constraint.get("case_insensitive"):
                return any(answer.casefold() == value.casefold() for value in constraint["values"])
            return answer in constraint["values"]
        if kind == "integer":
            return bool(re.fullmatch(r"(?:0|[1-9]\d*)", answer)) and int(answer) >= constraint.get("minimum", 0)
        if kind == "short_text":
            return len(answer) >= constraint.get("min_length", 0)
        if kind == "bbox":
            return (0 <= answer[0] < answer[2] <= request["image_width"]
                    and 0 <= answer[1] < answer[3] <= request["image_height"])
        return False
    except Exception:
        return False


def replay(directory: Path, endpoint: str, key: str, timeout: float = 120) -> list[dict[str, Any]]:
    origin = local_origin(endpoint)
    paths = sorted(path for path in directory.glob("*.json")
                   if re.fullmatch(r"[0-9a-f]{64}\.json", path.name))
    if not paths:
        raise ValueError("No captured request files; nothing was sent")
    results = []
    for path in paths:
        if path.stat().st_size > 128 * 1024:
            raise ValueError("Captured request exceeds protocol size limit")
        body = path.read_bytes()
        request = json.loads(body)
        started = time.monotonic()
        status, valid, error = None, False, None
        call = urllib.request.Request(origin + "/v1/predict", data=body, headers={
            "Authorization": "Bearer " + key, "Content-Type": "application/json",
        })
        # Direct loopback: do not forward private replay data through an HTTP proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        try:
            with opener.open(call, timeout=timeout) as response:
                status = response.status
                valid = status == 200 and response_valid(request, json.loads(response.read(1024 * 1024 + 1)))
        except urllib.error.HTTPError as exception:
            status = exception.code
            error = "http_error"
            exception.close()
        except Exception as exception:
            error = type(exception).__name__
        results.append({"item_id": request.get("item_id"), "http_status": status,
                        "response_valid": valid, "error": error,
                        "latency_ms": round((time.monotonic() - started) * 1000)})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:9001")
    args = parser.parse_args()
    key = os.environ.get("MODEL_API_KEY", "")
    if len(key) < 32:
        parser.error("Set MODEL_API_KEY to the running local API key")
    try:
        rows = replay(args.requests, args.endpoint, key)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"local_replay": True, "accuracy_evaluated": False, "results": rows},
                     ensure_ascii=False, indent=2))
    raise SystemExit(0 if all(row["response_valid"] for row in rows) else 1)


if __name__ == "__main__":
    main()

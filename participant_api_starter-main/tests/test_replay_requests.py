"""Synthetic local HTTP fixtures: no evaluator calls or competition questions."""

import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hfsa_adapter import _capture_replay_request
from model_adapter import DemoModel
from replay_requests import local_origin, replay
import test_server as server_fixtures


class ReplayTests(unittest.TestCase):
    setUp = server_fixtures.APITests.setUp
    tearDown = server_fixtures.APITests.tearDown
    request_data = server_fixtures.APITests.request_data

    def capture(self, *requests):
        directory = self.package / "private-replay"
        with patch.dict("os.environ", {"HFSA_API_REPLAY_DIR": str(directory)}):
            for request in requests:
                _capture_replay_request(request)
        return directory

    def test_public_endpoints_and_credentials_in_urls_are_rejected(self):
        for endpoint in ("https://example.org", "http://192.168.1.2:9001",
                         "http://key@127.0.0.1:9001", "http://127.0.0.1:9001/api",
                         "http://127.0.0.1:9001?key=secret"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                local_origin(endpoint)
        self.assertEqual(local_origin("http://localhost:9001/"), "http://127.0.0.1:9001")

    def test_replays_success_and_failure_locally_without_exposing_request_or_answer(self):
        class Model:
            def predict(self, request, image_paths):
                if request["item_id"] == "TEST_FAIL":
                    raise RuntimeError("private exception")
                return DemoModel().predict(request, image_paths)

        self.server.model = Model()
        successful = self.request_data()
        failed = self.request_data()
        failed.update(item_id="TEST_FAIL", request_id="test:failed:1")
        rows = replay(self.capture(successful, failed), self.origin, "x" * 32, timeout=3)
        by_id = {row["item_id"]: row for row in rows}
        self.assertTrue(by_id["TEST_1"]["response_valid"])
        self.assertEqual(by_id["TEST_FAIL"]["http_status"], 500)
        self.assertFalse(by_id["TEST_FAIL"]["response_valid"])
        self.assertNotIn("Synthetic question", json.dumps(rows))
        self.assertNotIn("private exception", json.dumps(rows))
        self.assertNotIn("answer", json.dumps(rows))

    def test_http_200_with_an_invalid_enum_answer_is_flagged(self):
        class Model:
            def predict(self, request, image_paths):
                return "outside allowed values"

        self.server.model = Model()
        rows = replay(self.capture(self.request_data()), self.origin, "x" * 32, timeout=3)
        self.assertEqual(rows[0]["http_status"], 200)
        self.assertFalse(rows[0]["response_valid"])

    def test_redirects_are_not_followed(self):
        destination = self.origin

        class Redirect(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(302)
                self.send_header("Location", destination + "/v1/predict")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            origin = "http://127.0.0.1:" + str(server.server_port)
            rows = replay(self.capture(self.request_data()), origin, "x" * 32, timeout=3)
            self.assertEqual(rows[0]["http_status"], 302)
            self.assertFalse(rows[0]["response_valid"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()

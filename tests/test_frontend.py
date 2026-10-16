"""Offline HTTP checks for the optional demo interface."""

import json
import threading
import unittest
from http.client import HTTPConnection
from unittest.mock import patch

import frontend


class FrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = frontend.ThreadingHTTPServer(("127.0.0.1", 0), frontend.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, method, path, body=None, headers=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port)
        connection.request(method, path, body, headers or {})
        response = connection.getresponse()
        result = response.status, response.read().decode()
        connection.close()
        return result

    def test_page_has_code_and_isolated_preview(self):
        status, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn('sandbox="allow-scripts"', body)
        self.assertIn('id="code"', body)

    def test_generation_calls_existing_graph(self):
        html = "<html><body>Counter</body></html>"
        with patch.dict("os.environ", {"GROQ_API_KEY": "offline"}), patch.object(frontend.app, "invoke", return_value={"html": html}) as invoke:
            status, body = self.request("POST", "/api/generate", json.dumps({"prompt": " Build a counter "}), {"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["html"], html)
        invoke.assert_called_once_with({"request": "Build a counter"})

    def test_invalid_requests_do_not_call_graph(self):
        with patch.object(frontend.app, "invoke") as invoke:
            for body in ['{', '[]', '{"prompt": ""}']:
                status, _ = self.request("POST", "/api/generate", body, {"Content-Type": "application/json"})
                self.assertEqual(status, 400)
            status, _ = self.request("POST", "/api/generate", '{}', {"Content-Type": "application/json", "Origin": "https://example.com"})
            self.assertEqual(status, 403)
        invoke.assert_not_called()

    def test_busy_server_rejects_concurrent_generation(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "offline"}), frontend.GENERATION_LOCK:
            status, _ = self.request("POST", "/api/generate", '{"prompt":"counter"}', {"Content-Type": "application/json"})
        self.assertEqual(status, 409)

    def test_failure_does_not_expose_provider_details(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "offline"}), patch.object(frontend.app, "invoke", side_effect=RuntimeError("private provider details")):
            status, body = self.request("POST", "/api/generate", '{"prompt":"counter"}', {"Content-Type": "application/json"})
        self.assertEqual(status, 502)
        self.assertNotIn("private provider details", body)

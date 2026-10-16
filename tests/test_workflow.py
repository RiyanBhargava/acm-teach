"""Offline checks for state flow, output handling, and model failures."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main
import nodes
from prompts import SYSTEM_PROMPT


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "output" / "index.html"
        self.path_patch = patch.object(nodes, "OUTPUT_FILE", self.output)
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)
        self.model_patch = patch.object(nodes, "ChatGroq")
        self.model = self.model_patch.start().return_value
        self.addCleanup(self.model_patch.stop)

    def test_graph_preserves_request_and_saves_unwrapped_html(self):
        html = "<html><body>Counter: 0</body></html>"
        self.model.invoke.return_value = SimpleNamespace(content=f"```html\n{html}\n```")
        result = main.app.invoke({"request": "Build a counter"})
        self.assertEqual(result["request"], "Build a counter")
        self.assertEqual(result["html"], html)
        self.assertEqual(result["file_path"], str(self.output))
        self.assertEqual(self.output.read_text(encoding="utf-8"), html)
        self.model.invoke.assert_called_once_with([
            ("system", SYSTEM_PROMPT), ("human", "Build a counter")
        ])

    def test_new_generation_replaces_old_output(self):
        self.output.parent.mkdir()
        self.output.write_text("old app", encoding="utf-8")
        html = "<!DOCTYPE html><HTML><body>Quiz ✓</body></HTML>"
        self.model.invoke.return_value = SimpleNamespace(content=html)
        main.app.invoke({"request": "Build a quiz"})
        self.assertEqual(self.output.read_text(encoding="utf-8"), html)

    def test_bad_output_preserves_previous_file(self):
        self.output.parent.mkdir()
        self.output.write_text("previous app", encoding="utf-8")
        for content in ("", "Here is your app", "<html><body>unfinished"):
            with self.subTest(content=content):
                self.model.invoke.return_value = SimpleNamespace(content=content)
                with self.assertRaises(ValueError):
                    main.app.invoke({"request": "Build a counter"})
                self.assertEqual(self.output.read_text(encoding="utf-8"), "previous app")

    def test_provider_failure_does_not_create_output(self):
        self.model.invoke.side_effect = RuntimeError("Provider unavailable")
        with self.assertRaisesRegex(RuntimeError, "Provider unavailable"):
            main.app.invoke({"request": "Build a counter"})
        self.assertFalse(self.output.exists())

    def test_missing_key_stops_before_prompt_or_model_call(self):
        with patch.dict("os.environ", {}, clear=True), patch.object(main, "load_dotenv"), patch("builtins.input") as read:
            with self.assertRaisesRegex(ValueError, "GROQ_API_KEY"):
                main.main()
        read.assert_not_called()
        self.model.invoke.assert_not_called()

    def test_blank_request_stops_before_model_call(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "offline-test"}), patch.object(main, "load_dotenv"), patch("builtins.input", return_value="  "):
            with self.assertRaisesRegex(ValueError, "app idea"):
                main.main()
        self.model.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()

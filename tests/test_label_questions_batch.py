"""Offline checks for direct prompt calls and per-provider output files."""

import ast
import contextlib
import io
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/label_questions_batch.ipynb"


def result(provider):
    return {"id": 999, "relevant_docs": [provider], "relevant_articles": []}


class BatchNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]

    def scope(self, directory):
        root = Path(directory)
        original = root / "questions.json"
        original.write_text(json.dumps([
            {"id": 9, "question": "nine", "topic": "t"},
            {"id": 1, "question": "one"}, {"id": 20, "question": "untested"}
        ]), encoding="utf-8")
        return dict(json=json, Path=Path, OUTPUT_DIR=root, ORIGINAL_PATH=original,
                    PROMPT='My rules: {"example": 1}',
                    questions=[{"id": 9, "question": "nine", "relevant_docs": ["old"]},
                               {"id": 1, "question": "one"}])

    def clients(self):
        clients = {p: Mock() for p in ("openai", "gemini")}
        clients["openai"].responses.create.return_value = SimpleNamespace(
            status="completed", output_text=json.dumps(result("openai"))
        )
        clients["gemini"].post.return_value.ok = True
        clients["gemini"].post.return_value.json.return_value = {"choices": [{
            "finish_reason": "stop", "message": {"content": json.dumps(result("gemini"))}
        }]}
        modules = {
            "openai": SimpleNamespace(OpenAI=Mock(return_value=clients["openai"])),
            "requests": SimpleNamespace(Session=Mock(return_value=clients["gemini"])),
        }
        return clients, modules

    def helpers(self, scope):
        for source in self.cells[2:]:
            tree = ast.parse(source)
            tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
            exec(compile(tree, str(NOTEBOOK), "exec"), scope)

    def test_notebook_compiles_without_executing_user_configuration(self):
        for source in self.cells:
            compile(source, str(NOTEBOOK), "exec")
        self.assertNotIn("batches.create", self.cells[2])
        self.assertNotIn("/batches", self.cells[2])
        self.assertIn("responses.create", self.cells[2])
        self.assertIn("/chat/completions", self.cells[2])

    def test_optional_providers_direct_requests_and_separate_outputs(self):
        for enabled in ((), ("openai",), ("gemini",), ("openai", "gemini")):
            with self.subTest(enabled=enabled), TemporaryDirectory() as directory:
                scope = self.scope(directory)
                clients, modules = self.clients()
                scope["configs"] = {
                    p: {"api_key": "test" if p in enabled else None, "model": "chosen-model",
                        "params": {"temperature": 0, "unused": None}}
                    for p in clients
                }
                with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
                    exec(self.cells[2], scope)
                    exec(self.cells[3], scope)
                self.assertEqual(set(scope["ready_labels"]), set(enabled))
                expected = scope["PROMPT"] + "\n\n" + json.dumps({"id": 9, "question": "nine"})
                for provider, client in clients.items():
                    call = client.responses.create if provider == "openai" else client.post
                    self.assertEqual(call.call_count, 2 if provider in enabled else 0)
                    output = Path(directory) / f"questions_completed_{provider}.json"
                    self.assertEqual(output.exists(), provider in enabled)
                    if provider not in enabled:
                        continue
                    args, kwargs = call.call_args_list[0]
                    body = kwargs if provider == "openai" else kwargs["json"]
                    self.assertNotIn("unused", body)
                    self.assertEqual(body["temperature"], 0)
                    self.assertEqual(body["model"], "chosen-model")
                    if provider == "openai":
                        self.assertEqual(body["input"], expected)
                    else:
                        self.assertEqual(args[0], "https://openrouter.ai/api/v1/chat/completions")
                        self.assertEqual(body["messages"], [{"role": "user", "content": expected}])
                        client.headers.update.assert_called_once_with({"Authorization": "Bearer test"})
                    rows = json.loads(output.read_text())
                    self.assertEqual([r["id"] for r in rows], [1, 9, 20])
                    self.assertEqual(rows[1]["relevant_docs"], [provider])
                    self.assertEqual(rows[1]["topic"], "t")
                    self.assertNotIn("relevant_docs", rows[2])
                    client.close.assert_called_once()

    def test_failure_keeps_partial_results_and_other_provider_finishes(self):
        with TemporaryDirectory() as directory:
            scope = self.scope(directory)
            clients, modules = self.clients()
            clients["openai"].responses.create.side_effect = [
                SimpleNamespace(status="completed", output_text=json.dumps(result("openai"))),
                RuntimeError("service unavailable"),
            ]
            scope["configs"] = {p: {"api_key": "test", "model": "model", "params": None} for p in clients}
            with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()) as captured:
                exec(self.cells[2], scope)
                exec(self.cells[3], scope)
            self.assertIn("service unavailable", captured.getvalue())
            self.assertEqual(set(scope["ready_labels"]), {"gemini"})
            partial = json.loads((Path(directory) / "labels_openai.json").read_text())
            self.assertEqual([r["id"] for r in partial], [9])
            self.assertFalse((Path(directory) / "questions_completed_openai.json").exists())
            clients["openai"].close.assert_called_once()

    def test_invalid_responses_do_not_save_labels(self):
        for provider, mode in (("openai", "incomplete"), ("gemini", "http"),
                               ("gemini", "length"), ("gemini", "json")):
            with self.subTest(provider=provider, mode=mode), TemporaryDirectory() as directory:
                scope = self.scope(directory)
                self.helpers(scope)
                clients, modules = self.clients()
                if provider == "openai":
                    clients[provider].responses.create.return_value.status = "incomplete"
                else:
                    response = clients[provider].post.return_value
                    if mode == "http":
                        response.ok, response.status_code, response.text = False, 401, "Unauthorized"
                    elif mode == "length":
                        response.json.return_value["choices"][0]["finish_reason"] = "length"
                    else:
                        response.json.return_value["choices"][0]["message"]["content"] = "not JSON"
                with patch.dict(sys.modules, modules), self.assertRaises((RuntimeError, ValueError)):
                    scope["generate_labels"](provider, {"api_key": "test", "model": "model", "params": None},
                                             scope["questions"], scope["PROMPT"])
                self.assertEqual(json.loads((Path(directory) / f"labels_{provider}.json").read_text()), [])
                clients[provider].close.assert_called_once()

    def test_json_and_fenced_json_are_accepted_without_repairing_invalid_json(self):
        scope = dict(json=json)
        self.helpers(scope)
        expected = result("gemini")
        text = json.dumps(expected)
        for content in (text, "\ufeff" + text, "```json\n" + text + "\n```", "```\n" + text + "\n```"):
            self.assertEqual(scope["parse_model_json"](content), expected)
        for content in (None, "", "  ", "[]", "{}", 'Here is JSON: ' + text,
                        '```json\n{"relevant_docs": [],}\n```'):
            with self.subTest(content=content), self.assertRaises(ValueError):
                scope["parse_model_json"](content)

    def test_http_and_model_json_errors_save_raw_response(self):
        for mode in ("http_json", "empty", "invalid"):
            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                scope = self.scope(directory)
                self.helpers(scope)
                clients, modules = self.clients()
                response = clients["gemini"].post.return_value
                raw = "<html>Unavailable</html>" if mode == "http_json" else ("" if mode == "empty" else "not JSON")
                if mode == "http_json":
                    response.status_code = 200
                    response.text = raw
                    response.json.side_effect = ValueError("bad HTTP JSON")
                else:
                    response.json.return_value["choices"][0]["message"]["content"] = raw
                with patch.dict(sys.modules, modules), self.assertRaisesRegex(ValueError, "ID 9:"):
                    scope["generate_labels"]("gemini", {"api_key": "test", "model": "model", "params": None},
                                             scope["questions"], scope["PROMPT"])
                self.assertEqual((Path(directory) / "response_error_gemini.txt").read_text(encoding="utf-8"), raw)
                self.assertEqual(json.loads((Path(directory) / "labels_gemini.json").read_text()), [])

    def test_subset_merge_rejects_duplicate_and_unknown_ids(self):
        with TemporaryDirectory() as directory:
            scope = self.scope(directory)
            self.helpers(scope)
            labels = Path(directory) / "labels.json"
            output = Path(directory) / "out.json"
            before = scope["ORIGINAL_PATH"].read_bytes()
            update = dict(result("gemini"), id=1)
            for bad in ([update, update], [dict(update, id=99)]):
                labels.write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaises(ValueError):
                    scope["merge_with_questions"](scope["ORIGINAL_PATH"], labels, output)
            self.assertEqual(scope["ORIGINAL_PATH"].read_bytes(), before)


if __name__ == "__main__":
    unittest.main()


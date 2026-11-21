from __future__ import annotations
import importlib.util
from pathlib import Path
import unittest
spec = importlib.util.spec_from_file_location("chat_adapter", Path(__file__).resolve().parents[1] / "adapters/chat_completions.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
_output_contract = adapter._output_contract
_parse_visible_judge_response = adapter._parse_visible_judge_response
_visible_output = adapter._visible_output

class AdapterTests(unittest.TestCase):
    def test_chat_completion_visible_output(self):
        content, refusal, finish = _visible_output({
            "choices": [{
                "message": {"content": '{"results":[]}', "refusal": None},
                "finish_reason": "stop",
            }]
        })
        self.assertEqual(content, '{"results":[]}')
        self.assertIsNone(refusal)
        self.assertEqual(finish, "stop")

    def test_output_contract_names_every_item_and_batch_envelope(self):
        contract = _output_contract(
            {"type": "object", "required": ["schema_version", "results"]},
            {
                "request_id": "request-1",
                "judge_id": "judge_a",
                "items": [{"item_id": "item-1"}, {"item_id": "item-2"}],
            },
        )
        self.assertIn("exactly one JSON batch object", contract)
        self.assertIn('"request_id":"request-1"', contract)
        self.assertIn('"results"', contract)
        self.assertLess(contract.index("item-1"), contract.index("item-2"))

    def test_invalid_visible_json_is_preserved_for_schema_validation(self):
        visible_text = '{"schema_version":"ward-judge-response-1.0","results":['
        self.assertEqual(_parse_visible_judge_response(visible_text), visible_text)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ttrl2.agent.transformers_loop import (  # noqa: E402
    parse_llama3_json, parse_qwen3_xml,
)


def test_qwen3_xml_single_call():
    text = ('<tool_call>\n<function=find_user_id_by_name_zip>\n'
            '<parameter=first_name>\n"Yusuf"\n</parameter>\n'
            '<parameter=zip>\n19122\n</parameter>\n</tool_call>')
    calls = parse_qwen3_xml(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "find_user_id_by_name_zip"
    assert calls[0]["arguments"]["first_name"] == "Yusuf"
    assert calls[0]["arguments"]["zip"] == 19122


def test_qwen3_xml_multi_call():
    text = ('<tool_call>\n<function=f1>\n<parameter=a>\n1\n</parameter>\n</tool_call>'
            '<tool_call>\n<function=f2>\n<parameter=b>\n2\n</parameter>\n</tool_call>')
    calls = parse_qwen3_xml(text)
    assert [c["name"] for c in calls] == ["f1", "f2"]


def test_llama3_json_parameters_key():
    # Llama-3.1 emits {"name":..., "parameters": {...}} (verified 2026-08-30)
    text = ('{"name": "find_user_id_by_email", '
            '"parameters": {"email": "yusuf@example.com"}}<|eom_id|>')
    calls = parse_llama3_json(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "find_user_id_by_email"
    assert calls[0]["arguments"]["email"] == "yusuf@example.com"


def test_llama3_json_nested_objects():
    # multiple calls separated by markers; nested braces must survive
    text = ('{"name": "a", "parameters": {"x": {"y": 1}}}<|eom_id|>'
            '{"name": "b", "parameters": {"z": 2}}<|eom_id|>')
    calls = parse_llama3_json(text)
    assert [c["name"] for c in calls] == ["a", "b"]
    assert calls[0]["arguments"]["x"] == {"y": 1}


def test_llama3_json_tool_calls_wrapper():
    text = '{"tool_calls": [{"function": {"name": "f", "arguments": {"k": "v"}}}]}'
    calls = parse_llama3_json(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "f"


def test_no_calls():
    assert parse_qwen3_xml("just prose") == []
    assert parse_llama3_json("just prose") == []

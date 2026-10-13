"""MCP dependency hints must remain valid through provider schema conversion."""

import copy
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from tools.mcp_tool_schema import _convert_mcp_schema
from tools.schema_sanitizer import sanitize_tool_schemas


def _provider_parameters(schema):
    tool = SimpleNamespace(name="knowledge_update", description="Update knowledge", inputSchema=schema)
    converted = _convert_mcp_schema("spexus", tool)
    return sanitize_tool_schemas([{"type": "function", "function": converted}])[0]["function"]["parameters"]


@pytest.mark.parametrize("dependency", [
    {"status_comment": [{"type": "object", "properties": {}}]},
    {"status_comment": ["status", None]},
    {"status_comment": "status"},
    {"status_comment": ["status", "status"]},
    ["status"],
    None,
])
def test_malformed_mcp_dependencies_cannot_poison_provider_schema(dependency):
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "status_comment": {"type": "string"},
            "payload": {
                "type": "object",
                "properties": {"status": {"type": "string"}},
                "dependentRequired": copy.deepcopy(dependency),
            },
        },
        "dependentRequired": copy.deepcopy(dependency),
    }
    original = copy.deepcopy(schema)
    parameters = _provider_parameters(schema)

    Draft202012Validator.check_schema(parameters)
    assert "dependentRequired" not in parameters
    assert "dependentRequired" not in parameters["properties"]["payload"]
    assert schema == original


def test_valid_mcp_dependencies_keep_their_meaning_beside_invalid_entries():
    dependency = {"status_comment": ["status"], "required": ["properties"]}
    schema = {
        "type": "object",
        "properties": {name: {"type": "string"} for name in (
            "status", "status_comment", "required", "properties", "broken",
        )},
        "dependentRequired": {**dependency, "broken": [{}]},
    }
    original = copy.deepcopy(schema)
    parameters = _provider_parameters(schema)

    Draft202012Validator.check_schema(parameters)
    assert parameters["dependentRequired"] == dependency
    validator = Draft202012Validator(parameters)
    assert not validator.is_valid({"status_comment": "superseded"})
    assert validator.is_valid({"status_comment": "superseded", "status": "inactive"})
    assert not validator.is_valid({"required": "trigger"})
    assert validator.is_valid({"required": "trigger", "properties": "dependency"})
    assert schema == original

"""Map recipe constraints to actual vLLM constrained sampling.

Strict DSML parameters use JSON values (string=false), including JSON-quoted
strings. This lets xgrammar enforce the value schema without interpreting raw
DSML strings as JSON. Property order is fixed; optional properties may be omitted.
"""
from __future__ import annotations

import copy
import json

from .protocol import ProtocolError


def strict_tool_tag(tools, required):
    from xgrammar import StructuralTag
    from xgrammar.structural_tag import (
        ConstStringFormat, JSONSchemaFormat, OptionalFormat, SequenceFormat,
        TagFormat, TagsWithSeparatorFormat, TriggeredTagsFormat,
    )

    invokes = []
    for tool in tools:
        function = tool["function"]
        name = function["name"]
        if any(c in name for c in '\"<>\n\r'):
            raise ProtocolError("Tool name cannot be represented in DSML")
        schema = function.get("parameters") or {"type": "object", "properties": {}}
        if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
            raise ProtocolError("Strict tools require a closed object parameter schema")
        if any(k in schema for k in ("oneOf", "anyOf", "allOf", "patternProperties", "$ref")):
            raise ProtocolError("Root schema composition is not supported for strict DSML tools")
        required_names = set(schema.get("required", []))
        parameters = []
        for key, prop in schema.get("properties", {}).items():
            if any(c in key for c in '\"<>\n\r'):
                raise ProtocolError("Parameter name cannot be represented in DSML")
            value_schema = copy.deepcopy(prop)
            if "$defs" in schema:
                value_schema.setdefault("$defs", schema["$defs"])
            parameter = TagFormat(
                begin=f'<｜DSML｜ parameter name="{key}" string="false">',
                content=JSONSchemaFormat(json_schema=value_schema),
                end="</｜DSML｜ parameter>\n",
            )
            parameters.append(parameter if key in required_names else OptionalFormat(content=parameter))
        invokes.append(TagFormat(
            begin=f'<｜DSML｜ invoke name="{name}">\n',
            content=SequenceFormat(elements=parameters) if parameters else ConstStringFormat(value=""),
            end="</｜DSML｜ invoke>\n",
        ))
    calls = TagFormat(
        begin="<｜DSML｜ calls>\n",
        content=TagsWithSeparatorFormat(tags=invokes, separator="", at_least_one=True),
        end="</｜DSML｜ calls>",
    )
    if required:
        return StructuralTag(format=SequenceFormat(elements=[
            # The recipe prompt already contains the opening calls tag.
            # Constrain only the continuation, not a second nested calls block.
            calls.content, ConstStringFormat(value="</｜DSML｜ calls>"),
        ]))
    return StructuralTag(format=TriggeredTagsFormat(
        triggers=["<｜DSML｜ calls>"], tags=[calls], excludes=["<think>", "</think>"],
    ))


def structured_outputs(prepared):
    from vllm.sampling_params import StructuredOutputsParams

    constraint = prepared.inference_options.output_constraint
    if constraint is None:
        return None
    if constraint.kind == "json":
        return StructuredOutputsParams(json=json.loads(constraint.schema)) if constraint.schema else StructuredOutputsParams(json_object=True)
    if constraint.kind == "regex":
        return StructuredOutputsParams(regex=constraint.schema)
    if constraint.kind in {"tool_call", "tool_call_required"}:
        tools = json.loads(constraint.schema)
        required = constraint.kind == "tool_call_required"
        if not tools:
            return None
        if all(t["function"].get("strict") is True for t in tools):
            return StructuredOutputsParams(structural_tag=strict_tool_tag(tools, required).model_dump_json())
        # Automatic, non-strict calls retain the model's normal distribution.
        if not required:
            return None
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
        from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag
        tag = get_model_structural_tag(
            "deepseek_v41", [ChatCompletionToolsParam.model_validate(t) for t in tools],
            "required", False,
        )
        if tag is None:
            raise ProtocolError("Required tool constraints are unavailable")
        # R2's registry builds a whole assistant answer. Recipe has already
        # emitted its required-call prefix in the prompt, so remove that exact
        # prefix from the pinned registry grammar before continuing generation.
        from xgrammar import StructuralTag
        from xgrammar.structural_tag import SequenceFormat, ConstStringFormat
        elements = getattr(tag.format, "elements", None)
        if not elements or not isinstance(elements[0], ConstStringFormat) or elements[0].value != "\n\n<｜DSML｜ calls>\n":
            raise ProtocolError("Required tool grammar prefix differs from the pinned recipe contract")
        tag = StructuralTag(format=SequenceFormat(elements=elements[1:]))
        return StructuredOutputsParams(structural_tag=tag.model_dump_json())
    raise ProtocolError("Unsupported backend output constraint")

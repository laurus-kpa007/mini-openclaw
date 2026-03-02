"""Example plugin demonstrating the @tool decorator API."""

from mini_openclaw.tools.base import ToolContext, ToolResult
from mini_openclaw.tools.plugin_loader import tool


@tool(
    name="word_count",
    description="Count the number of words, characters, and lines in a given text.",
    parameters=[
        {
            "name": "text",
            "type": "string",
            "description": "The text to analyze",
            "required": True,
        },
    ],
    category="text",
)
async def word_count(arguments: dict, context: ToolContext) -> ToolResult:
    text = arguments.get("text", "")
    if not text:
        return ToolResult(success=False, output="", error="No text provided")

    words = len(text.split())
    chars = len(text)
    lines = text.count("\n") + 1

    return ToolResult(
        success=True,
        output=f"Words: {words}\nCharacters: {chars}\nLines: {lines}",
    )


@tool(
    name="text_transform",
    description="Transform text: uppercase, lowercase, title case, or reverse.",
    parameters=[
        {
            "name": "text",
            "type": "string",
            "description": "The text to transform",
            "required": True,
        },
        {
            "name": "operation",
            "type": "string",
            "description": "Transformation to apply",
            "required": True,
            "enum": ["uppercase", "lowercase", "title", "reverse"],
        },
    ],
    category="text",
)
async def text_transform(arguments: dict, context: ToolContext) -> ToolResult:
    text = arguments.get("text", "")
    operation = arguments.get("operation", "uppercase")

    ops = {
        "uppercase": str.upper,
        "lowercase": str.lower,
        "title": str.title,
        "reverse": lambda s: s[::-1],
    }

    transform = ops.get(operation)
    if not transform:
        return ToolResult(success=False, output="", error=f"Unknown operation: {operation}")

    return ToolResult(success=True, output=transform(text))

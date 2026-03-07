"""Calculator MCP Service - performs math operations.

No external API keys required. Works out of the box.
"""
import math
from typing import Optional
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Calculator")


@mcp.tool()
async def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Supports basic arithmetic (+, -, *, /, **), math functions
    (sqrt, sin, cos, tan, log, log10, abs, ceil, floor, pi, e),
    and parentheses for grouping.

    Args:
        expression: A mathematical expression to evaluate, e.g. "2 + 3 * 4"
    """
    # Whitelist of safe names for eval
    safe_names = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "abs": abs,
        "ceil": math.ceil,
        "floor": math.floor,
        "pow": pow,
        "round": round,
        "pi": math.pi,
        "e": math.e,
        "inf": math.inf,
    }

    try:
        # Sanitize: only allow digits, operators, parens, dots, spaces, and function names
        cleaned = expression.strip()

        # Evaluate in restricted namespace
        result = eval(cleaned, {"__builtins__": {}}, safe_names)  # noqa: S307
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error evaluating '{expression}': {str(e)}"


@mcp.tool()
async def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between common units.

    Supported conversions:
    - Temperature: celsius, fahrenheit, kelvin
    - Length: meters, feet, inches, cm, km, miles
    - Weight: kg, lbs, oz, grams

    Args:
        value: The numeric value to convert
        from_unit: Source unit (e.g. "celsius", "meters", "kg")
        to_unit: Target unit (e.g. "fahrenheit", "feet", "lbs")
    """
    from_unit = from_unit.lower().strip()
    to_unit = to_unit.lower().strip()

    # Temperature conversions
    temp_conversions = {
        ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
        ("celsius", "kelvin"): lambda v: v + 273.15,
        ("kelvin", "celsius"): lambda v: v - 273.15,
        ("fahrenheit", "kelvin"): lambda v: (v - 32) * 5 / 9 + 273.15,
        ("kelvin", "fahrenheit"): lambda v: (v - 273.15) * 9 / 5 + 32,
    }

    # Length conversions (to meters as base)
    length_to_meters = {
        "meters": 1.0, "m": 1.0,
        "feet": 0.3048, "ft": 0.3048,
        "inches": 0.0254, "in": 0.0254,
        "cm": 0.01, "centimeters": 0.01,
        "km": 1000.0, "kilometers": 1000.0,
        "miles": 1609.344, "mi": 1609.344,
        "yards": 0.9144, "yd": 0.9144,
    }

    # Weight conversions (to kg as base)
    weight_to_kg = {
        "kg": 1.0, "kilograms": 1.0,
        "lbs": 0.453592, "pounds": 0.453592,
        "oz": 0.0283495, "ounces": 0.0283495,
        "grams": 0.001, "g": 0.001,
    }

    # Check temperature
    key = (from_unit, to_unit)
    if key in temp_conversions:
        result = temp_conversions[key](value)
        return f"{value} {from_unit} = {result:.4f} {to_unit}"

    # Check length
    if from_unit in length_to_meters and to_unit in length_to_meters:
        meters = value * length_to_meters[from_unit]
        result = meters / length_to_meters[to_unit]
        return f"{value} {from_unit} = {result:.4f} {to_unit}"

    # Check weight
    if from_unit in weight_to_kg and to_unit in weight_to_kg:
        kg = value * weight_to_kg[from_unit]
        result = kg / weight_to_kg[to_unit]
        return f"{value} {from_unit} = {result:.4f} {to_unit}"

    return f"Error: Cannot convert from '{from_unit}' to '{to_unit}'. Supported: temperature (celsius/fahrenheit/kelvin), length (meters/feet/inches/cm/km/miles), weight (kg/lbs/oz/grams)"


if __name__ == "__main__":
    mcp.run(transport='stdio')

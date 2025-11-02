import yfinance as yf
from utils.data import compare_companies_json
import json
import re


def compare_companies(ticker1: str, ticker2: str) -> dict:
    """
    Compare two companies' financial performance using Yahoo Finance data.
    """
    try:
        if not ticker1 or not ticker2:
            return {"error": "Both ticker symbols must be provided."}

        c1 = yf.Ticker(ticker1).info
        c2 = yf.Ticker(ticker2).info

        if not c1 or "shortName" not in c1:
            return {"error": f"Could not retrieve data for '{ticker1}'. Please check the ticker symbol."}
        if not c2 or "shortName" not in c2:
            return {"error": f"Could not retrieve data for '{ticker2}'. Please check the ticker symbol."}

        metrics = [
            "shortName", "sector", "marketCap", "currentPrice",
            "revenueGrowth", "grossMargins", "profitMargins",
            "trailingPE", "dividendYield"
        ]

        company1_data = {m: c1.get(m, "N/A") for m in metrics}
        company2_data = {m: c2.get(m, "N/A") for m in metrics}

        insight = (
            f"{company1_data['shortName']} vs {company2_data['shortName']}\n"
            f"Sector: {company1_data['sector']} | {company2_data['sector']}\n"
            f"Market Cap: {company1_data['marketCap']} | {company2_data['marketCap']}\n"
            f"Profit Margin: {company1_data['profitMargins']} | {company2_data['profitMargins']}\n"
            f"P/E Ratio: {company1_data['trailingPE']} | {company2_data['trailingPE']}\n"
            f"Dividend Yield: {company1_data['dividendYield']} | {company2_data['dividendYield']}\n"
        )

        return {
            "insight": insight,
            "company1": company1_data,
            "company2": company2_data
        }

    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


tools = [
    {"type": "function", "function": compare_companies_json}
]


def handle_tool_calls(tool_calls):
    """Handle tool calls from OpenAI"""
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        tool = globals().get(tool_name)
        result = tool(**arguments) if tool else {}

        results.append({
            "role": "tool",
            "content": json.dumps(result),
            "tool_call_id": tool_call.id
        })

    return results


def is_comparison_query(text: str) -> bool:
    """Detects whether a user's input is a comparison-type query."""
    comparison_patterns = [
        r"\bbetter\b",
        r"\bworse\b",
        r"\bcompare\b",
        r"\bcomparing\b",
        r"\bcomparison\b",
        r"\bversus\b",
        r"\bvs\b",
        r"\bvs\.\b",
        r"\bagainst\b",
        r"\bbetween\b",
        r"\bdifference\s+(between|of)\b",
        r"\bhow\s+does\b.*\bcompare\b.*\bto\b",
        r"\bhow\s+do\b.*\bcompare\b",
        r"\bwhich\s+(is|has|performs|does)\b.*\b(better|worse|higher|lower|more|less)\b",
        r"\b(is|are)\s+(better|worse|higher|lower|more|less)\b",
        r"\bthan\b",
        r"\brelative\s+to\b",
    ]

    combined_pattern = "|".join(comparison_patterns)
    return bool(re.search(combined_pattern, text, flags=re.IGNORECASE))

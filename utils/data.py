system_prompt = """You are a helpful financial assistant that specializes in comparing companies.

When a user asks to compare two companies, you should:
1. Use the compare_companies function to fetch financial data
2. Analyze the results and provide a clear, insightful comparison
3. Highlight key differences and similarities
4. Be objective and data-driven in your analysis

Always present information in a clear, structured format that's easy to understand."""

compare_companies_json = {
    "name": "compare_companies",
    "description": "Compare two companies' financial performance using their stock ticker symbols. Returns key metrics including market cap, profit margins, P/E ratio, dividend yield, and more.",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker1": {
                "type": "string",
                "description": "The stock ticker symbol of the first company (e.g., 'AAPL' for Apple)"
            },
            "ticker2": {
                "type": "string",
                "description": "The stock ticker symbol of the second company (e.g., 'MSFT' for Microsoft)"
            }
        },
        "required": ["ticker1", "ticker2"]
    }
}

output_format = """
\nFormat your final response as follows:

Summary:
Provide a short 2–3 sentence overview comparing the entities (e.g., companies, sectors, etc.). 
Focus on overall performance, trends, or key insights.

Key Points:
* Highlight 3–5 bullet points comparing important metrics or differences.
* Each point should start with either the entity name or the metric being compared.
* Be specific — use data-driven phrasing if applicable.

Choice:
Conclude with which entity performs better overall and briefly justify why.
"""

system_prompt += output_format
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from open_deep_research.graph import builder
import uuid
import asyncio
import os
from dotenv import load_dotenv
from typing import Dict, Any, List

# Disable ChromaDB telemetry before importing
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

load_dotenv()
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

from langchain_community.tools.tavily_search import TavilySearchResults

LLM_PROVIDERS = {
    "openai": {
        "planner_model": "gpt-4o",
        "writer_model": "gpt-4o"
    },
    # Uncomment if you have valid keys and access
    "anthropic": {
        "planner_model": "claude-3-sonnet-20240229",
        "writer_model": "claude-3-sonnet-20240229"
    },
    # "deepseek": {
    #     "planner_model": "deepseek-r1",
    #     "writer_model": "deepseek-r1"
    # }
}

def should_use_deep_research(query: str) -> bool:
    deep_keywords = [
        "compare", "analysis", "research", "trend", "history",
        "timeline", "difference", "advantages", "disadvantages",
        "impact", "future", "comprehensive", "detailed", "deep research"
    ]
    return (any(kw in query.lower() for kw in deep_keywords) or len(query.split()) >= 3)

def user_friendly_error(e: Exception) -> str:
    msg = str(e)
    if "credit balance is too low" in msg or "insufficient_quota" in msg:
        return "❌ Your credit balance is too low to access the API. Please check your billing or API credits."
    if "rate_limit_exceeded" in msg or "429" in msg:
        return "❌ The AI provider hit its rate or token limit. Please try again later or use a shorter/simpler query."
    if "Request too large" in msg:
        return "❌ Your request is too large for the current model/token limit. Please try a shorter or simpler query."
    return f"❌ An error occurred: {msg}"

async def run_research_workflow(topic: str, llm_provider: str) -> dict:
    config = LLM_PROVIDERS[llm_provider]
    thread = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "search_api": "tavily",
            "planner_provider": llm_provider,
            "planner_model": config["planner_model"],
            "writer_provider": llm_provider,
            "writer_model": config["writer_model"],
            "max_search_depth": 2,
            "number_of_queries": 3,
            "report_structure": [
                "Executive Summary",
                "Background & Origins",
                "Historical Timeline",
                "Key Developments",
                "Current Status",
                "Analysis & Insights",
                "Conclusion"
            ]
        }
    }
    final_state = None
    instructions = "Provide a detailed, multi-section report with at least 5 sources. Include a summary and a timeline if relevant."
    try:
        async for event in graph.astream({"topic": topic, "instruction": instructions}, thread, stream_mode="values"):
            final_state = event
    except Exception as e:
        # Optionally log the full traceback for debugging:
        # import traceback; print(traceback.format_exc())
        return {
            "provider": llm_provider,
            "state": None,
            "report": user_friendly_error(e)
        }
    return {
        "provider": llm_provider,
        "state": final_state,
        "report": extract_final_report(final_state)
    }

def extract_final_report(state: dict) -> str:
    if not state or "messages" not in state:
        return "No research results found."
    messages = state["messages"]
    last_msg = None
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if content and len(content) > 500:
            if any(marker in content.lower() for marker in ["## introduction", "# executive summary", "## background", "## timeline"]):
                return content
        last_msg = messages[-1] if messages else None
        return getattr(last_msg, "content", "No report generated.")
    if not last_msg:
        return "No report generated."
    sources = state.get("sources", [])
    if sources:
        sources_md = "\n\n## Sources\n"
        for i, src in enumerate(sources, 1):
            title = src.get("title", "Source")
            url = src.get("url", "")
            snippet = src.get("snippet", "")
            sources_md += f"{i}. [{title}]({url})"
            if snippet:
                sources_md += f" — {snippet[:150]}..."
            sources_md += "\n"
        return last_msg + sources_md
    return last_msg

def summarize_tavily_results(result):
    import json
    import re
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception:
            return result
    if not isinstance(result, list):
        return str(result)
    timeline = []
    for i, item in enumerate(result, 1):
        title = item.get('title', '')
        url = item.get('url', '')
        content = item.get('content', '')
        events = re.findall(r'(\b(19|20)\d{2}\b[^\n\r\-:]*[\-:–—][^\n\r]+)', content)
        if events:
            for event in events:
                timeline.append(f"{event[0].strip()}")
        else:
            sentences = re.split(r'(?<=[.!?]) +', content.strip())
            short_content = ' '.join(sentences[:3])
            timeline.append(f"{title}: {short_content}\nSource: {url}")
    if timeline:
        return "\n\n".join(timeline)
    return "No relevant timeline or content found in search results."

async def generate_research_report(topic: str, provider: str = "openai") -> str:
    if provider not in LLM_PROVIDERS:
        provider = "openai"
    config = LLM_PROVIDERS[provider]
    if "gpt-3.5" in os.getenv("OPENAI_MODEL", "").lower() or "gpt-3.5" in provider:
        warning = (
            "⚠️ [WARNING] The selected model (gpt-3.5-turbo) does not support advanced tool/function calling. "
            "Deep research may not work as expected. For best results, use GPT-4 or Claude 3."
        )
    else:
        warning = ""
    if not should_use_deep_research(topic):
        tavily = TavilySearchResults(max_results=3)
        try:
            result = await tavily.ainvoke(topic)
            summary = summarize_tavily_results(result)
            return f"{warning}\n\nQuery is simple. Here is a summary of search results:\n\n{summary}"
        except Exception as e:
            return f"{warning}\n\nQuery doesn't require deep research and quick search failed: {user_friendly_error(e)}"
    try:
        tasks = [run_research_workflow(topic, provider) for provider in LLM_PROVIDERS.keys()]
        results = await asyncio.gather(*tasks)
        best_result = max(results, key=lambda r: len(r["report"]))
        if not best_result["report"] or "No research results found" in best_result["report"]:
            tavily = TavilySearchResults(max_results=3)
            try:
                result = await tavily.ainvoke(topic)
                summary = summarize_tavily_results(result)
                return f"{warning}\n\nDeep research failed. Here is a summary of search results:\n\n{summary}"
            except Exception as e:
                return f"{warning}\n\nDeep research failed and quick search failed: {user_friendly_error(e)}"
        return f"{warning}\n\n### Deep Research Report (via {best_result['provider']})\n\n{best_result['report']}"
    except Exception as e:
        tavily = TavilySearchResults(max_results=3)
        try:
            result = await tavily.ainvoke(topic)
            summary = summarize_tavily_results(result)
            return f"{warning}\n\nDeep research failed. Here is a summary of search results:\n\n{summary}"
        except Exception as e2:
            return f"{warning}\n\nSorry, all research tools failed. Please try again later or use a simpler query.\n\nDetails: {user_friendly_error(e)}\nQuick search also failed: {user_friendly_error(e2)}"

@tool("deep_research")
async def deep_research_tool(query: str) -> str:
    """
    Generates a comprehensive research report with sources using multiple LLMs.
    """
    return await generate_research_report(query)


tools = [
    deep_research_tool,
    TavilySearchResults(max_results=2),
    # ...add other tools here
]

import json
import os
import requests
from typing import Dict, Literal
from typing_extensions import Literal
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph

from assistant.configuration import Configuration, SearchAPI
from assistant.utils import deduplicate_and_format_sources, tavily_search, format_sources, perplexity_search
from assistant.state import SummaryState, SummaryStateInput, SummaryStateOutput
from assistant.prompts import (
    language_detection_instructions,
    query_writer_instructions,
    summarizer_instructions,
    reflection_instructions
)

# Schema definitions
LANGUAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "language": {"type": "string"},
        "language_code": {"type": "string"}
    },
    "required": ["language", "language_code"]
}

QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "aspect": {"type": "string"},
        "rationale": {"type": "string"}
    },
    "required": ["query", "aspect", "rationale"]
}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"}
    },
    "required": ["summary"]
}

REFLECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "knowledge_gap": {"type": "string"},
        "follow_up_query": {"type": "string"}
    },
    "required": ["knowledge_gap", "follow_up_query"]
}


def make_llm_request(prompt: str, schema: Dict, system_prompt: str = None) -> Dict:
    """Make a structured request to the LLM."""
    messages = [
        {"role": "system", "content": system_prompt or "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    request_data = {
        "messages": messages,
        "model": os.getenv("OPENAI_MODEL"),
        "max_tokens": 2000,
        "temperature": 0,
        "guided_json": json.dumps(schema),
        "guided_decoding_backend": "xgrammar"
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }

    try:
        response = requests.post(
            f"{os.getenv('OPENAI_API_BASE')}/chat/completions",
            json=request_data,
            headers=headers
        )
        response.raise_for_status()
        result = response.json()
        return json.loads(result['choices'][0]['message']['content'])
    except Exception as e:
        print(f"Error in LLM request: {str(e)}")
        raise


def detect_language(state: SummaryState, config: RunnableConfig):
    """Detect the language of the input text."""
    print(f"Detecting language for: {state.research_topic}")

    result = make_llm_request(
        prompt=language_detection_instructions.format(input_text=state.research_topic),
        schema=LANGUAGE_SCHEMA,
        system_prompt="You are a language detection expert."
    )

    print(f"Detected language: {result['language']} ({result['language_code']})")
    return {
        "language": result["language"],
        "language_code": result["language_code"]
    }


def generate_query(state: SummaryState, config: RunnableConfig):
    """Generate a query for web search using structured output."""
    query_writer_instructions_formatted = query_writer_instructions.format(
        research_topic=state.research_topic,
        language=state.language
    )

    result = make_llm_request(
        prompt=f"Generate a query for web search in {state.language}:",
        schema=QUERY_SCHEMA,
        system_prompt=query_writer_instructions_formatted
    )

    return {"search_query": result["query"]}


def web_research(state: SummaryState, config: RunnableConfig):
    """Gather information from the web."""
    configurable = Configuration.from_runnable_config(config)
    search_api = (configurable.search_api.value
                  if not isinstance(configurable.search_api, str)
                  else configurable.search_api)

    print(f"Using search API: {search_api}")
    print(f"Search query ({state.language}): {state.search_query}")

    if search_api == "tavily":
        search_results = tavily_search(state.search_query, include_raw_content=True, max_results=1)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000,
                                                    include_raw_content=True)
    elif search_api == "perplexity":
        search_results = perplexity_search(state.search_query, state.research_loop_count)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000,
                                                    include_raw_content=False)
    else:
        raise ValueError(f"Unsupported search API: {configurable.search_api}")

    return {
        "sources_gathered": [format_sources(search_results)],
        "research_loop_count": state.research_loop_count + 1,
        "web_research_results": [search_str]
    }


def summarize_sources(state: SummaryState, config: RunnableConfig):
    """Summarize the gathered sources using structured output."""
    most_recent_web_research = state.web_research_results[-1]

    if state.running_summary:
        prompt = (
            f"<User Input>\n{state.research_topic}\n</User Input>\n\n"
            f"<Existing Summary>\n{state.running_summary}\n</Existing Summary>\n\n"
            f"<New Search Results>\n{most_recent_web_research}\n</New Search Results>\n\n"
            f"Provide summary in {state.language}."
        )
    else:
        prompt = (
            f"<User Input>\n{state.research_topic}\n</User Input>\n\n"
            f"<Search Results>\n{most_recent_web_research}\n</Search Results>\n\n"
            f"Provide summary in {state.language}."
        )

    result = make_llm_request(
        prompt=prompt,
        schema=SUMMARY_SCHEMA,
        system_prompt=summarizer_instructions
    )

    return {"running_summary": result["summary"]}


def reflect_on_summary(state: SummaryState, config: RunnableConfig):
    """Reflect on the summary and generate a follow-up query using structured output."""
    prompt = (
        f"Identify a knowledge gap and generate a follow-up web search query "
        f"based on our existing knowledge: {state.running_summary}\n"
        f"Provide response in {state.language}."
    )

    result = make_llm_request(
        prompt=prompt,
        schema=REFLECTION_SCHEMA,
        system_prompt=reflection_instructions.format(
            research_topic=state.research_topic,
            language=state.language
        )
    )

    return {"search_query": result["follow_up_query"]}


def finalize_summary(state: SummaryState):
    """Finalize the summary."""
    all_sources = "\n".join(source for source in state.sources_gathered)
    state.running_summary = f"## Research Topic\n{state.research_topic}\n\n## Summary\n\n{state.running_summary}\n\n### Sources:\n{all_sources}"
    return {"running_summary": state.running_summary}


def route_research(state: SummaryState, config: RunnableConfig) -> Literal["finalize_summary", "web_research"]:
    """Route the research based on the follow-up query."""
    configurable = Configuration.from_runnable_config(config)
    if state.research_loop_count <= configurable.max_web_research_loops:
        return "web_research"
    else:
        return "finalize_summary"


# Build the graph
builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput, config_schema=Configuration)

# Add nodes
builder.add_node("detect_language", detect_language)  # New language detection node
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("summarize_sources", summarize_sources)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("finalize_summary", finalize_summary)

# Add edges
builder.add_edge(START, "detect_language")  # Start with language detection
builder.add_edge("detect_language", "generate_query")  # Then proceed to query generation
builder.add_edge("generate_query", "web_research")
builder.add_edge("web_research", "summarize_sources")
builder.add_edge("summarize_sources", "reflect_on_summary")
builder.add_conditional_edges("reflect_on_summary", route_research)
builder.add_edge("finalize_summary", END)

# Compile the graph
graph = builder.compile()
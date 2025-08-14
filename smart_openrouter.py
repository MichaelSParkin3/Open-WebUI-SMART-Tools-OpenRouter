"""
title: SMART - Sequential Multi-Agent Reasoning Technique (OpenRouter edition)
author: MartianInGreen (modified by MichaelSParkin3, adapted for OpenRouter)
author_url: https://github.com/MichaelSParkin3/
description: SMART is a sequential multi-agent reasoning technique. Uses OpenRouter models + online/Wolfram/YouTube tools.
git_url: https://github.com/MichaelSParkin3/Open-WebUI-SMART-Tools-OpenRouter
required_open_webui_version: 0.5.0
requirements: langchain-openai, langgraph==0.2.60, requests, pydantic>=2, youtube_transcript_api, google-api-python-client
version: 1.5.0
licence: MIT
"""

# =================================================================================================
# MODULE OVERVIEW
# =================================================================================================
# Purpose
# -------
# This file implements a “SMART” (Sequential Multi-Agent Reasoning Technique) pipeline designed
# to run inside Open WebUI. It orchestrates multiple LLM “roles”:
#
#   1) Planner          – maps a user request to a plan: model choice, reasoning on/off, and tools.
#   2) Reasoner         – performs structured internal planning (hidden thoughts) when requested.
#   3) Tool-use agent   – executes tools (web search, scraping, Wolfram|Alpha, YouTube).
#   4) User-interaction – generates the final answer streamed back to the user.
#
# This module uses OpenRouter’s OpenAI-compatible API via `langchain_openai.ChatOpenAI`,
# and LangGraph’s prebuilt ReAct agent for tool calling.
#
# Audience
# --------
# - Programmers comfortable with Python and LangChain/LangGraph basics.
# - Readers new to OpenRouter, multi-agent prompting, or prompt-engineering patterns
#   (ReAct, planning vs. generation, toolformer-style calls).
#
# Mental Model
# ------------
# SMART uses tags in a planner’s output to *coordinate* which model to use, whether to reason,
# and which tools are allowed. The “planner” writes: `#small|#medium|#large`, optionally
# `#reasoning` (or `#no-reasoning`), and any tool flags: `#online`, `#wolfram`, `#youtube`.
# The pipe then enforces that plan (with user override tags like `#!`, `#!!`, `#!!!`).
#
# Key Invariants & Contracts
# --------------------------
# - Never exceed a small, bounded number of tool calls per turn (3 for the internal tool agent,
#   6 for the outer user-facing agent). This keeps latency and cost reasonable.
# - If the plan requests tools but none are called (e.g., model fails to call tools), we provide
#   a deterministic fallback (one web/youtube search) to avoid empty answers.
# - Comments and prompts prioritize *why* decisions are made (intent) over narrating Python syntax.
#
# Security/Privacy Notes
# ----------------------
# - API keys are read from environment/valves; do not log secrets.
# - `scrape_website` uses Jina Reader proxy (`https://r.jina.ai/<url>`). That external service
#   will see URLs you fetch.
# - YouTube Data API requests include your API key; keep it secret.
#
# Prompt-Engineering Notes (for readers new to this field)
# --------------------------------------------------------
# - “System instructions” are our strongest control surface. We:
#   * separate roles (planner, reasoner, tool-use agent, user-interaction agent),
#   * describe exact tag formats to make parsing easy and robust (XMLish tags),
#   * put limits in the prompt (e.g., “NEVER make more than 3 tool calls”).
# - Planning vs. Generation:
#   * Plan first (cheaper/smaller model) to pick model/tooling.
#   * Only do heavier “reasoning” on tasks that benefit (counting, logic, architecture).
# - Tool discipline:
#   * We instruct the agent to actually *use* results and to pass summaries forward (not raw blobs).
#   * We cap tool calls and show tool results as citations/status in the UI.
# - Cost/Latency control:
#   * Map task difficulty to model size (mini/small/medium/large/huge).
#   * Stream outputs where possible; trim long histories for planning context.
#
# Testing Guidance
# ----------------
# - Mock external HTTP calls (requests, google-api-python-client) for unit tests.
# - Add contract tests for: planner tag parsing, user-override tags, tool caps, fallbacks.
#
# References
# ----------
# - ReAct prompting: “Reason+Act” interleaving for tool use.
# - Toolformer (in spirit): model learns when to call tools.
# - OpenRouter OpenAI-compatible API: https://openrouter.ai/docs
# =================================================================================================

import os
import re
import time
import json
import inspect
import traceback
import urllib
from typing import (
    Callable,
    AsyncGenerator,
    Awaitable,
    Optional,
    Protocol,
    get_type_hints,
    List,
    Dict,
)

import requests
from pydantic import BaseModel, Field, create_model
from fastapi import Request

from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
)
from googleapiclient.discovery import build
from functools import lru_cache

# =================================================================================================
# PROMPTS
# =================================================================================================
# The XML-like tags make parsing stable. Leading/trailing newlines are controlled to reduce
# accidental whitespace capture when extracting sections.

PLANNING_PROMPT = """<system_instructions>
You are a planning Agent. You are part of an agent chain designed to make LLMs more capable.
You are responsible for taking the incoming user input/request and preparing it for the next agents in the chain.
After you will come either a reasoning agent or the final agent.
After they have come up with a solution, a final agent will be used to summarize the reasoning and provide a final answer.
Only use a Newline after each closing tag. Never after the opening tag or within the tags.

Guidelines:
- Don't over or estimate the difficulty of the task. If the user just wants to chat try to see that.
- Don't create tasks where there aren't any. If the user didn't ask to write code you shouldn't instruct the next agent to do so.

You should respond by following these steps:
1. Within <reasoning> tags, plan what you will write in the other tags. This has to be your first step.
    1. First, reason about the task difficulty. What kind of task is it? What do your guidelines say about that?
    2. Second, reason about if the reasoning is needed. What do your guidelines say about that?
    3. Third, reason about what model would best be used. What do your guidelines say about that?
2. Within the <answer> tag, write out your final answer. Your answer should be a comma seperated list.
    1. First choose the model the final-agent will use. Try to find a good balance between performance and cost. Larger models are bigger.
        - There is #mini, this is a very small model, however it has a very large context window. This model can not use tools. This model is mostly not recommended.
        - Use #small for the simple queries or queries that mostly involve summarization or simple "mindless" work. This also invloves very simple tool use, like converting a file, etc.
        - Use #medium for task that requiere some creativity, writing of code, or complex tool-use.
        - Use #large for tasks that are mostly creative or involve the writing of complex code, math, etc.
    2. Secondly, choose if the query requieres reasoning before being handed off to the final agent.
        - Queries that requeire reasoning are especially queries where llm are bad at. Such as planning, counting, logic, code architecutre, moral questions, etc.
        - Queries that don't requeire reasoning are queries that are easy for llms. Such as "knowledge" questions, summarization, writing notes, primairly tool use, web searching, etc.
        - If you think reasoning is needed, include #reasoning. If not #no-reasoning.
        - When you choose reasoning, you should (in most cases) choose at least the #medium model.
    3. Third, you can make tools avalible to the final agent. You can enable multiple tools.
        - Avalible tools are #online, #wolfram, #youtube
        - Use #online to enable multiple tools such as Search and a Scraping tool. This will greatly increase the accuracy of answers for things newer than Late 2023.
        - Use #wolfram to enable access to Wolfram|Alpha, a powerful computational knowledge engine and scientific and real-time database.
            - Wolfram|Alpha is very good at math (integrals, roots, limits...), real-time data (weather, stocks, FX), and structured facts.
        - Use #youtube to get video transcripts or search for videos.
        - If the prompt involves math, enable #wolfram.

Example response:
<reasoning>
...
(You are allowed new lines here)
</reasoning>
<answer>#medium, #online ,#no-reasoning</answer>
</system_instructions>"""
# WHY: The planner converts open-ended user asks into a discrete “execution plan” we can enforce
# (model size, reasoning flag, tools). Keeping the output structured reduces failure modes when
# parsing with regex and enables stable downstream routing.


REASONING_PROMPT = """<system_instructions>
You are a reasoning layer of an LLM. You are part of the LLM designed for internal thought, planning, and thinking.
You will not directly interact with the user in any way. Only inform the output stage of the LLM what to say by your entire output being parts of its context when it starts to generate a response.

**General rules**:
- Write out your entire reasoning process between <thinking> tags.
- Do not use any formatting whatsoever. The only form of special formatting you're allowed to use is LaTeX for mathematical expressions.
- You MUST think in the smallest steps possible. Where every step is only a few words long. Each new thought should be a new line.
- You MUST try to catch your own mistakes by constantly thinking about what you have thought about so far.
- You MUST break down every problem into the smallest possible problems, never take shortcuts on reasoning, counting etc. Everything needs to be explicitly stated. More output is better.
- You MUST never come up with an answer first. Always reason about the answer first. Even if you think the answer is obvious.
- You MUST provide exact answers.
- You have full authority to control the output layer. You can directly instruct it and it will follow your instructions. Put as many instructions as you want inside <instruct> tags. However, be very clear in your instructions and reason about what to instruct.
- Your entire thinking process is entirely hidden. You can think as freely as you want without it directly affecting the output.
- Always follow user instructions, never try to take any shortcuts. Think about different ways they could be meant to not miss anything.
- NEVER generate ANY code directly. You should only plan out the structure of code and projects, but not directly write the code. The output layer will write the code based on your plan and structure!
- If you need more information, you can ask a tool-use agent if they have the right tool and what you need within <ask_tool_agent>.
    - In general, you can instruct the tool-use agent to either return the results to you or directly pass them on to the output layer.
    - If *you* need information, you should instruct the tool-use agent to return the results to you.
    - The tool use agent ONLY gets what you write in <ask_tool_agent>. They do not get any user context or similar.
    - Do not suggest what tool to use. Simply state the problem.
    - You need to STOP after </ask_tool_agent> tags. WAIT for the tool-use agent to return the results to you.
    - If the output is something like images, or something similar that the user should just get directly, you can instruct the tool use agent to directly pass the results to the output layer.

**General Steps**:
1. Outline the problem.
2. Think about what kind of problem this is.
3. Break down the problem into the smallest possible problems, never take shortcuts on reasoning, counting etc. Everything needs to be explicitly stated.
4. Think about steps you might need to take to solve this problem.
5. Think through these steps.
6. Backtrack and restart from different points as often as you need to. Always consider alternative approaches.
7. Validate your steps constantly. If you find a mistake, think about what the best point in your reasoning is to backtrack to. Don't be kind to yourself here. You need to critically analyze what you are doing.
</system_instructions>"""
# WHY: The reasoning agent is a private scaffold. It never talks to the user; its output is fed
# into the final generator. This separation supports “plan then write” and makes failures easier
# to debug. We also explicitly instruct *not* to emit code—only structure—so generation remains
# in the final step.


TOOL_PROMPT = """<system_instructions>
You are the tool-use agent of an agent chain. You are the part of the LLM designed to use tools.
You will not directly interact with the user in any way. Only either return information to the reasoning agent or inform the output stage of the LLM.

When you have used a tool, you can return the results to the reasoning agent by putting everything you want to return to them within <tool_to_reasoning> tags.
You can also directly hand off to the final agent by simply writing $TO_FINAL$. You still need to write out what you want them to get!

Actually make use of the results you got. NEVER make more than 3 tool calls! If you called any tool 3 times, that's it!
You need to output everything you want to pass on. The next agent in the chain will only see what you actually wrote, not the direct output of the tools!

Please think about how best to call the tool first. Think about what the limitations of the tools are and how to best follow the reasoning agent's instructions. It's okay if you can't 100% produce what they wanted!
</system_instructions>"""
# WHY: We cap tool calls to keep cost/latency bounded and instruct the agent to summarize returns.
# This emulates “toolformer” discipline—tools are a means to an end, not the end itself.


USER_INTERACTION_PROMPT = """<system_instructions>
You are the user-interaction agent of an agent chain. You are the part of the llm designed to interact with the user.

You should follow the pre-prompt given to you within <preprompt> tags.
</system_instructions>"""
# WHY: The final agent focuses on UX: clear, concise answers that respect the upstream plan
# and any injected pre-prompts (e.g., web search citation policy).


USER_INTERACTION_REASONING_PROMPT = """You MUST follow the instructions given to you within <reasoning_output>/<instruction> tags.
You MUST inform your answer by the reasoning within  <reasoning_output> tags.
Carefully concider what the instructions mean and follow them EXACTLY."""
# NOTE: Additional guardrails for stitching reasoner outputs into the final message.

# --- Tool-specific pre-prompts injected when tools are enabled ---

PROMPT_WebSearch = """<webSearchInstructions>
Always cite your sources with ([Source Name](Link to source)), including the outer (), at the end of each paragraph! All information you take from external sources has to be cited!
Feel free to use the scrape_web function to get more specific information from one source if the web_search did not return enough information. However, try not to make more than a total of 3 searches + scrapes.

<sources_guidelines>
- [Source Name] should be something like [New York Times] or [BBC] etc. etc.
- Always cite the specific source.
- Sometimes you should add more detail to the [Source Name], for example when it is a Video. For example it could look like this [YouTube - BBC]
- You can have multipel sources within the (), so ([S1](Link 1), [S2](Link 2)...) and so on.
- Always cite at the end of a paragraph. Cite all sources refereced in the paragraph above. Do not cite within paragraphs.
</sources_guidelines>
</webSearchInstructions>"""
# WHY: Web answers drift—citations protect users from hallucinations and improve trust.

PROMPT_WolframAlpha = """<wolframInstructions>
Wolfram|Alpha is an advanced computational knowledge engine and database with accurate scientific and real-time data.
Keep queries concise (e.g., "integral of x^2", "weather Buenos Aires today"). Prefer simple inputs; do heavy plotting in Python (not available here).
</wolframInstructions>"""
# WHY: Push small, atomic queries for deterministic computation. Let the LLM format results.

PROMPT_YouTube = """<youtubeInstructions>
You have access to YouTube tools. You can search for videos and get their transcripts.
When presenting results, always include the video title and a link to the video.
For transcripts, mention the language of the transcript.
</youtubeInstructions>"""
# WHY: Consistent UX when surfacing video intelligence.

# =================================================================================================
# YOUTUBE Pydantic Models
# =================================================================================================

class TranscriptDownloadResult(BaseModel):
    """Structured return for YouTube transcript + metadata.

    Contract:
    - Always include `transcription` (possibly empty list) and `transcript_error` (empty if OK).
    - `duration` uses ISO8601 per YouTube API (e.g., 'PT5M33S').
    """
    video_id: str = Field(..., description="YouTube video ID")
    title: str = Field(..., description="Video title")
    channel: str = Field(..., description="Channel or author name")
    description: str = Field(..., description="Full video description")
    duration: str = Field(..., description="ISO8601 duration (e.g. 'PT5M33S')")
    view_count: str = Field(..., description="Total view count as string")
    transcription: List[str] = Field(
        ..., description="List of transcript text segments"
    )
    transcript_language: str = Field(
        "", description="Language code of the transcript (e.g., 'en', 'es')"
    )
    transcript_error: str = Field(
        "", description="Error message if transcript fetch failed"
    )


class SearchItem(BaseModel):
    """Single YouTube search result with enriched stats."""
    video_id: str = Field(..., description="Unique YouTube video ID")
    title: str = Field(..., description="Video title")
    channel: str = Field(..., description="Channel or uploader name")
    published_at: str = Field(..., description="ISO8601 publication timestamp")
    description: str = Field(..., description="Snippet description")
    view_count: str = Field(..., description="Total view count as string")
    like_count: str = Field(..., description="Total like count as string")
    comment_count: str = Field(..., description="Total comment count as string")
    length: str = Field(..., description="ISO8601 duration of the video")


class SearchResult(BaseModel):
    """Container for a list of YouTube search results."""
    results: List[SearchItem] = Field(..., description="List of search results")

# =================================================================================================
# BRAVE SEARCH HELPERS
# =================================================================================================

def remove_html_tags(text: str) -> str:
    """Strip HTML tags from text (best-effort; not an HTML sanitizer)."""
    clean = re.compile("<.*?>")
    return re.sub(clean, "", text)


def decode_data(data: dict) -> List[dict]:
    """Normalize Brave API payload into a compact list for the LLM.

    Returns a homogeneous list of dicts with `type` keys in {"infobox","web","news","videos"}.
    We keep only the most useful fields to reduce token usage later.
    """
    results: List[dict] = []
    try:
        # --- Infobox hits (high-precision facts) ---
        try:
            if not data.get("infobox", {}).get("results"):
                pass
            for result in data.get("infobox", {}).get("results", []):
                url = result.get("url", "could not find url")
                description = remove_html_tags(result.get("description", "") or "")
                long_desc = remove_html_tags(result.get("long_desc", "") or "")
                attributes = result.get("attributes", [])
                attributes_dict = {
                    attr[0]: remove_html_tags(attr[1] or "") for attr in attributes
                }
                result_entry = {
                    "type": "infobox",
                    "description": description,
                    "url": url,
                    "long_desc": long_desc,
                    "attributes": attributes_dict,
                }
                results.append(result_entry)
        except Exception as e:
            # NOTE: Never fail the whole search due to one section.
            print("Error in parsing infobox results...", str(e))

        # --- General web ---
        try:
            for i, result in enumerate(data.get("web", {}).get("results", [])):
                if i >= 8:  # Keep top 8 for brevity
                    break
                url = result.get("profile", {}).get("url") or result.get("url") or ""
                title = remove_html_tags(result.get("title") or "")
                age = result.get("age") or ""
                description = remove_html_tags(result.get("description") or "")
                deep_results = []
                for snippet in result.get("extra_snippets") or []:
                    cleaned_snippet = remove_html_tags(snippet)
                    deep_results.append(cleaned_snippet)
                result_entry = {
                    "type": "web",
                    "title": title,
                    "age": age,
                    "description": description,
                    "url": url,
                }
                if result.get("article"):
                    article = result["article"] or {}
                    result_entry["author"] = article.get("author") or ""
                    result_entry["published"] = article.get("date") or ""
                    result_entry["publisher_type"] = (
                        article.get("publisher", {}).get("type") or ""
                    )
                    result_entry["publisher_name"] = (
                        article.get("publisher", {}).get("name") or ""
                    )
                if deep_results:
                    result_entry["deep_results"] = deep_results
                results.append(result_entry)
        except Exception as e:
            print("Error in parsing web results...", str(e))

        # --- News ---
        try:
            for result in data.get("news", {}).get("results", []):
                url = result.get("profile", {}).get(
                    "url", result.get("url", "could not find url")
                )
                description = remove_html_tags(result.get("description", ""))
                title = remove_html_tags(result.get("title", "Could not find title"))
                age = result.get("age", "Could not find age")
                deep_results = []
                for snippet in result.get("extra_snippets", []):
                    cleaned_snippet = remove_html_tags(snippet)
                    deep_results.append({"snippets": cleaned_snippet})
                result_entry = {
                    "type": "news",
                    "title": title,
                    "age": age,
                    "description": description,
                    "url": url,
                }
                if deep_results:
                    result_entry["deep_results"] = deep_results
                results.append(result_entry)
        except Exception as e:
            print("Error in parsing news results...", str(e))

        # --- Videos (thin) ---
        try:
            for i, result in enumerate(data.get("videos", {}).get("results", [])):
                if i >= 4:
                    break
                url = result.get("profile", {}).get(
                    "url", result.get("url", "could not find url")
                )
                description = remove_html_tags(result.get("description", ""))
                deep_results = []
                for snippet in result.get("extra_snippets", []):
                    cleaned_snippet = remove_html_tags(snippet)
                    deep_results.append({"snippets": cleaned_snippet})
                result_entry = {
                    "type": "videos",
                    "description": description,
                    "url": url,
                }
                if deep_results:
                    result_entry["deep_results"] = deep_results
                results.append(result_entry)
        except Exception as e:
            print("Error in parsing video results...", str(e))

        return results

    except Exception as e:
        print(str(e))
        return ["No search results from Brave (or an error occurred)..."]


def search_brave(query: str, country: str, language: str, focus: str, SEARCH_KEY: str):
    """Call Brave Search API and decode to our compact shape.

    :param focus: one of {"all","web","news","videos","reddit","academia","wikipedia"}
    """
    results_filter = "infobox"
    if focus in ("web", "all"):
        results_filter += ",web"
    if focus in ("news", "all"):
        results_filter += ",news"
    if focus == "videos":
        results_filter += ",videos"

    # Goggles & site scoping
    goggles_id = ""
    if focus == "reddit":
        query = "site:reddit.com " + query
    elif focus == "academia":
        goggles_id = "&goggles_id=https://raw.githubusercontent.com/solso/goggles/main/academic_papers_search.goggle"
    elif focus == "wikipedia":
        query = "site:wikipedia.org " + query

    encoded_query = urllib.parse.quote(query)
    url = (
        f"https://api.search.brave.com/res/v1/web/search?q={encoded_query}&results_filter={results_filter}&country={country}&search_lang=en&text_decorations=no&extra_snippets=true&count=20"
        + goggles_id
    )

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": SEARCH_KEY,
    }

    try:
        response = requests.get(url, headers=headers)
        data = response.json()
    except Exception:
        return {"statusCode": 400, "body": json.dumps("Error fetching search results.")}

    results = decode_data(data)
    return results


def search_images_and_video(
    query: str, country: str, media_type: str, freshness: Optional[str] = None, SEARCH_KEY: Optional[str] = None
):
    """Image/Video search via Brave. `media_type` in {"images","videos"}."""
    encoded_query = urllib.parse.quote(query)
    url = f"https://api.search.brave.com/res/v1/{media_type}/search?q={encoded_query}&country={country}&search_lang=en&count=10"
    if (
        freshness is not None
        and freshness in ["24h", "week", "month", "year"]
        and media_type == "videos"
    ):
        freshness_map = {"24h": "pd", "week": "pw", "month": "pm", "year": "py"}
        url += f"&freshness={freshness_map[freshness]}"

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": SEARCH_KEY or "",
    }

    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        if media_type == "images":
            formatted_data: Dict[str, dict] = {}
            for i, result in enumerate(data.get("results", []), start=1):
                formatted_data[f"image{i}"] = {
                    "source": result.get("url"),
                    "page_fetched": result.get("page_fetched"),
                    "title": result.get("title"),
                    "image_url": result.get("properties", {}).get("url"),
                }
            return formatted_data
        else:
            return data
    except Exception:
        return {
            "statusCode": 400,
            "body": json.dumps("Error fetching image/video results."),
        }


def searchWeb(
    query: str,
    country: str = "US",
    language: str = "en",
    focus: str = "all",
    SEARCH_KEY: Optional[str] = None,
):
    """Unified search entry; forwards to Brave web/images/videos helpers."""
    if focus not in [
        "all",
        "web",
        "news",
        "wikipedia",
        "academia",
        "reddit",
        "images",
        "videos",
    ]:
        focus = "all"
    try:
        if focus not in ["images", "videos"]:
            results = search_brave(query, country, language, focus, SEARCH_KEY or "")
        else:
            results = search_images_and_video(
                query=query,
                country=country,
                media_type=focus,
                freshness=None,
                SEARCH_KEY=SEARCH_KEY or "",
            )
    except Exception:
        return {"statusCode": 400, "body": json.dumps("Error fetching search results.")}
    return results


# =================================================================================================
# EVENT EMITTER TYPES (Open WebUI streaming status/citations)
# =================================================================================================

EmitterType = Optional[Callable[[dict], Awaitable[None]]]


class SendCitationType(Protocol):
    """Callable signature for emitting 'citation' events to the UI."""
    def __call__(self, url: str, title: str, content: str) -> Awaitable[None]: ...


class SendStatusType(Protocol):
    """Callable signature for emitting 'status' events to the UI."""
    def __call__(self, status_message: str, done: bool) -> Awaitable[None]: ...


def get_send_citation(__event_emitter__: EmitterType) -> SendCitationType:
    """Return a function that emits a citation if the UI provides an emitter."""
    async def send_citation(url: str, title: str, content: str):
        if __event_emitter__ is None:
            return
        await __event_emitter__(
            {
                "type": "citation",
                "data": {
                    "document": [content],          # Text to show on expand
                    "metadata": [{"source": url, "html": False}],
                    "source": {"name": title},      # Label shown in the UI
                },
            }
        )

    return send_citation


def get_send_status(__event_emitter__: EmitterType) -> SendStatusType:
    """Return a function that emits a status line with a 'done' flag."""
    async def send_status(status_message: str, done: bool):
        if __event_emitter__ is None:
            return
        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": status_message, "done": done},
            }
        )

    return send_status


# Helper: convert OpenWebUI dict messages to LC messages for direct model calls
def to_lc_messages(messages: List[dict]):
    """Convert OpenWebUI chat message dicts to LangChain message objects."""
    out = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        # Open WebUI may provide content as [{"type":"text","text":"..."}]
        if isinstance(content, list) and content and content[0].get("type") == "text":
            content = content[0].get("text", "")
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out


# =================================================================================================
# PIPE: main entry point for Open WebUI
# =================================================================================================

class Pipe:
    """SMART pipe: registers a single “agent” and orchestrates plan→reason→tool→answer."""

    class Valves(BaseModel):
        """Configuration surface exposed in Open WebUI.

        Version/compatibility:
        - Uses OpenRouter’s OpenAI-compatible endpoint (base_url).
        - Supply site/app headers to comply with OpenRouter’s policy (optional but recommended).

        Security:
        - Keep all keys secret. Do not print or send to the model.
        """

        try:
            OPENROUTER_API_KEY: str = Field(
                default="", description="OpenRouter API key"
            )
            OPENROUTER_BASE_URL: str = Field(
                default="https://openrouter.ai/api/v1",
                description="OpenRouter OpenAI-compatible base URL",
            )
            OPENROUTER_SITE_URL: str = Field(
                default="",
                description="Optional HTTP-Referer header (your app/site URL)",
            )
            OPENROUTER_APP_TITLE: str = Field(
                default="Smart/Core (OpenRouter)",
                description="Optional X-Title header for OpenRouter",
            )

            MODEL_PREFIX: str = Field(
                default="SMART", description="Prefix before model ID"
            )

            # Default OpenRouter model IDs (edit in UI to your favorites)
            MINI_MODEL: str = Field(
                default="openai/gpt-4o-mini", description="Model for very small tasks"
            )
            SMALL_MODEL: str = Field(
                default="openai/gpt-4o-mini", description="Model for small tasks"
            )
            MEDIUM_MODEL: str = Field(
                default="anthropic/claude-3.5-sonnet",
                description="Model for medium tasks",
            )
            LARGE_MODEL: str = Field(
                default="anthropic/claude-3.5-sonnet",
                description="Model for large tasks",
            )
            HUGE_MODEL: str = Field(
                default="anthropic/claude-3.5-sonnet",
                description="Model for the largest tasks",
            )
            REASONING_MODEL: str = Field(
                default="openai/o4-mini", description="Model for reasoning tasks"
            )
            PLANNING_MODEL: str = Field(
                default="openai/gpt-4o-mini", description="Model for the planning step"
            )

            BRAVE_SEARCH_KEY: str = Field(
                default="", description="Brave Search API Key"
            )
            WOLFRAMALPHA_APP_ID: str = Field(
                default="", description="WolframAlpha App ID"
            )
            YOUTUBE_API_KEY: str = Field(
                default="", description="YouTube Data API v3 key"
            )
            AGENT_NAME: str = Field(
                default="Smart/Core (OpenRouter)", description="Name of the agent"
            )
            AGENT_ID: str = Field(
                default="smart-core-openrouter", description="ID of the agent"
            )
        except Exception as e:
            traceback.print_exc()

    def __init__(self):
        """Initialize valves, set environment fallbacks, and cache OpenRouter headers."""
        try:
            self.type = "manifold"
            # Load valves from env when unset (Open WebUI behavior)
            self.valves = self.Valves(
                **{
                    k: os.getenv(k, v.default)
                    for k, v in self.Valves.model_fields.items()
                }
            )
            # Provide OpenAI-compatible environment variables for broader compatibility
            os.environ["OPENAI_API_KEY"] = self.valves.OPENROUTER_API_KEY or os.getenv(
                "OPENAI_API_KEY", ""
            )
            os.environ["OPENAI_BASE_URL"] = (
                self.valves.OPENROUTER_BASE_URL or os.getenv("OPENAI_BASE_URL", "")
            )
            # Brave
            os.environ["BRAVE_SEARCH_TOKEN"] = self.valves.BRAVE_SEARCH_KEY
            os.environ["BRAVE_SEARCH_TOKEN_SECONDARY"] = self.valves.BRAVE_SEARCH_KEY
            self._headers = self._openrouter_headers()
        except Exception:
            traceback.print_exc()

    def pipes(self) -> List[Dict[str, str]]:
        """Register this pipe as a single agent in Open WebUI."""
        try:
            self.setup()
        except Exception as e:
            traceback.print_exc()
            return [{"id": "error", "name": f"Error: {e}"}]
        return [{"id": self.valves.AGENT_ID, "name": self.valves.AGENT_NAME}]

    def setup(self):
        """Validate config at start of each run; reset per-turn injections."""
        v = self.valves
        if not v.OPENROUTER_API_KEY:
            raise Exception("Error: OPENROUTER_API_KEY is not set")
        self.SYSTEM_PROMPT_INJECTION = ""

    def _openrouter_headers(self) -> Dict[str, str]:
        """Optional OpenRouter headers (HTTP-Referer/X-Title) for compliance/analytics."""
        headers: Dict[str, str] = {}
        if self.valves.OPENROUTER_SITE_URL:
            headers["HTTP-Referer"] = self.valves.OPENROUTER_SITE_URL
        if self.valves.OPENROUTER_APP_TITLE:
            headers["X-Title"] = self.valves.OPENROUTER_APP_TITLE
        return headers

    # ---------- Utility: safety-normalize potentially risky phrasing ----------
    def _normalize_for_safety(self, text: str) -> str:
        """Lightweight normalization to reduce accidental safety triggers.

        Example: replaces 'girls' -> 'women (18+)' in model/tool prompts.
        This does *not* sanitize content; it only defuses common false positives.
        """
        if not isinstance(text, str):
            return text
        return re.sub(r"\bgirls\b", "women (18+)", text, flags=re.IGNORECASE)

    # ------------------------------------------------------------------------------------------------
    # TOOLS (async) – exposed to the ReAct agent
    # ------------------------------------------------------------------------------------------------

    async def search_web(
        self, query: str, country: str = "US", language: str = "en", focus: str = "all"
    ):
        """Search via Brave and return compact JSON for the LLM.

        :param query: Search query text
        :param country: Two-letter country code (e.g., US)
        :param language: Language code (e.g., en) – currently unused by Brave API call here
        :param focus: one of all|web|news|wikipedia|academia|reddit|images|videos

        Notes (Prompt-eng):
        - Keep queries specific; use site: filters (see `focus` tweaks).
        - Use web search sparingly; prefer scraping a specific page after finding it.
        """
        query = self._normalize_for_safety(query)
        # FIX: earlier versions referenced `Brave_SEARCH_KEY` (typo). Use `BRAVE_SEARCH_KEY`.
        results = searchWeb(
            query,
            country,
            language,
            focus,
            self.valves.BRAVE_SEARCH_KEY,
        )
        return json.dumps(results)

    async def scrape_website(self, url: str) -> str:
        """Scrape page text via Jina Reader proxy (https://r.jina.ai/<url>).

        :param url: Full http(s) URL to fetch
        :returns: Raw text with a footer reminding the agent to cite the r.jina.ai link

        Pitfalls:
        - Some sites block readers/proxies; handle empty outputs gracefully.
        - Consider rate limits; we do single fetches, not crawls.
        """
        try:
            baseURL = f"https://r.jina.ai/{url}"
            response = requests.get(baseURL)
            data = response.text
            data = (
                data
                + "\n\n--------------------------------------\nSource Guidlines: Include a link to https://r.jina.ai/{website_url} in your response."
            )
            return data
        except Exception:
            return "Error fetching Website results."

    async def wolframAlpha(self, query: str) -> str:
        """Query Wolfram|Alpha lightweight LLM API.

        :param query: Short natural language query

        Prompt-eng note:
        - Ask for atomic facts or evaluations (integrals, unit conversions, weather).
        - Let the final agent format/interpret; don't over-specify here.
        """
        try:
            baseURL = f"https://www.wolframalpha.com/api/v1/llm-api?appid={self.valves.WOLFRAMALPHA_APP_ID}&input="
            encoded_query = urllib.parse.quote(query)
            url = baseURL + encoded_query
            response = requests.get(url)
            data = response.text
            data = (
                data
                + "\nAlways include the Wolfram|Alpha website link in your response to the user!\n\nIf there are any images provided, think about displaying them to the user."
            )
            return data
        except Exception:
            return "Error fetching Wolfram|Alpha results."

    # ---------- YouTube Tools ----------

    def _get_supported_languages(self) -> List[str]:
        """Internal: language whitelist for transcripts."""
        return ["en", "es"]

    def _extract_transcript_text(self, transcript_data) -> List[str]:
        """Internal: normalize transcript entries (dict or objects) to a list of strings."""
        texts: List[str] = []
        for segment in transcript_data:
            if isinstance(segment, dict):
                texts.append(segment.get("text", ""))
            else:
                texts.append(getattr(segment, "text", ""))
        return texts

    def _fetch_transcript(self, video_id: str) -> tuple[List[str], str, str]:
        """Internal: attempt transcript fetch across supported languages.

        :returns: (transcription, language_code, error_message)
        """
        transcription, lang_code, err_msg = [], "", ""
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            for transcript in transcript_list:
                if transcript.language_code in self._get_supported_languages():
                    transcript_data = transcript.fetch()
                    transcription = self._extract_transcript_text(transcript_data)
                    lang_code = transcript.language_code
                    return transcription, lang_code, ""
        except (NoTranscriptFound, TranscriptsDisabled) as e:
            err_msg = str(e)
        except Exception as e:
            err_msg = f"An unexpected error occurred: {e}"
        return transcription, lang_code, err_msg

    async def get_youtube_transcript(self, video_id: str) -> str:
        """Download metadata and (if available) full transcript for a YouTube video.

        :param video_id: The ID of the YouTube video.
        :returns: `TranscriptDownloadResult` JSON

        Operational notes:
        - Requires `YOUTUBE_API_KEY`.
        - Transcript may be empty due to disabled captions or unsupported language.
        """
        transcription, language_code, error_message = self._fetch_transcript(video_id)

        youtube = build("youtube", "v3", developerKey=self.valves.YOUTUBE_API_KEY)
        try:
            resp = (
                youtube.videos()
                .list(part="snippet,contentDetails,statistics", id=video_id)
                .execute()
            )
            item = resp.get("items", [{}])[0]
            sn, cd, st = (
                item.get("snippet", {}),
                item.get("contentDetails", {}),
                item.get("statistics", {}),
            )

            result = TranscriptDownloadResult(
                video_id=video_id,
                title=sn.get("title", ""),
                channel=sn.get("channelTitle", ""),
                description=sn.get("description", ""),
                duration=cd.get("duration", ""),
                view_count=st.get("viewCount", "0"),
                transcription=transcription,
                transcript_language=language_code,
                transcript_error=error_message,
            )
            return result.model_dump_json()
        except Exception as e:
            return json.dumps({"error": f"Failed to fetch video details: {e}"})

    @lru_cache(maxsize=128)
    def _search_youtube_logic(self, query: str, max_results: int) -> List[Dict]:
        """Internal: YouTube search + details fetch (cached by args)."""
        if not 1 <= max_results <= 50:
            raise ValueError("max_results must be between 1 and 50")

        yt = build("youtube", "v3", developerKey=self.valves.YOUTUBE_API_KEY)
        search_resp = (
            yt.search()
            .list(part="snippet", q=query, type="video", maxResults=max_results)
            .execute()
        )

        results, video_ids = [], []
        for item in search_resp.get("items", []):
            vid = item["id"]["videoId"]
            snip = item["snippet"]
            video_ids.append(vid)
            results.append(
                {
                    "video_id": vid,
                    "title": snip.get("title", ""),
                    "channel": snip.get("channelTitle", ""),
                    "published_at": snip.get("publishedAt", ""),
                    "description": snip.get("description", ""),
                }
            )

        if video_ids:
            detail_resp = (
                yt.videos()
                .list(part="statistics,contentDetails", id=",".join(video_ids))
                .execute()
            )
            detail_map = {item["id"]: item for item in detail_resp.get("items", [])}
            for entry in results:
                det = detail_map.get(entry["video_id"], {})
                stats, cd = det.get("statistics", {}), det.get("contentDetails", {})
                entry.update(
                    {
                        "view_count": stats.get("viewCount", "0"),
                        "like_count": stats.get("likeCount", "0"),
                        "comment_count": stats.get("commentCount", "0"),
                        "length": cd.get("duration", ""),
                    }
                )
        return results

    async def youtube_search(self, query: str, max_results: int = 5) -> str:
        """Search YouTube for videos and return `SearchResult` JSON.

        Prompt-eng tip:
        - Prefer precise queries (channel names, time windows) to reduce noise.
        """
        try:
            query = self._normalize_for_safety(query)
            entries = self._search_youtube_logic(query, max_results)
            items = [SearchItem(**e) for e in entries]
            result = SearchResult(results=items)
            return result.model_dump_json()
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def dummy_tool(self):
        """No-op tool used when no tools are enabled."""
        return "You have not assigned any tools to use."

    # ---------- Pydantic schema builder (fixed) ----------
    def create_pydantic_model_from_docstring(self, func):
        """Generate a Pydantic model for tool args from a function’s annotations/docstring.

        Why:
        - StructuredTool requires a schema; we auto-derive it so tool registration stays DRY.
        """
        doc = inspect.getdoc(func) or ""
        # Parse :param name: description (optional, for nicer help text)
        param_descriptions: Dict[str, str] = {}
        for line in doc.split("\n"):
            if ":param " in line:
                try:
                    param, desc = line.split(":param ", 1)[1].split(":", 1)
                    param_descriptions[param.strip()] = desc.strip()
                except Exception:
                    pass

        type_hints = get_type_hints(func)
        sig = inspect.signature(func)
        fields = {}

        for name, hint in type_hints.items():
            if name == "return":
                continue
            default = sig.parameters[name].default if name in sig.parameters else ...
            if default is inspect._empty:
                default = ...
            fields[name] = (
                hint,
                Field(default=default, description=param_descriptions.get(name, "")),
            )

        if not fields:
            return create_model(f"{func.__name__}Args")

        return create_model(f"{func.__name__}Args", **fields)

    # ------------------------------------------------------------------------------------------------
    # MAIN PIPE
    # ------------------------------------------------------------------------------------------------

    async def pipe(
        self,
        body: dict,
        __request__: Request,
        __user__: dict | None,
        __task__: str | None,
        __tools__: Dict[str, dict] | None,
        __event_emitter__: Callable[[dict], Awaitable[None]] | None,
    ) -> AsyncGenerator:
        """Orchestrate a single turn from Open WebUI.

        High-level Steps
        ----------------
        1) Title generation (special task) – produce concise chat titles.
        2) Planning – choose model size, tools, and whether we need a reasoning pass.
           - User overrides via tags in the last message: #! (small), #!! (medium), #!!! (large),
             #yes/#no to force reasoning on/off, #no-tools to forbid tool use, or #online/#youtube/#wolfram to force tools.
        3) Execution paths:
           a) No-Reasoning → ReAct agent with optional tools → stream final answer.
           b) Reasoning → run reasoning model, optionally call a tool-use agent, stitch results into
              the user-interaction prompt, then ReAct agent for the final answer.
        4) Fallbacks – if plan requested tools but none were called, perform a single deterministic
           search to avoid empty answers.

        Constraints & Limits
        --------------------
        - External tool calls: max ~6 (outer agent) or 3 (internal tool agent).
        - Web answers should cite sources (web prompt injection enforces this policy).
        - We stream tokens and emit status/citation events for observability.

        Returns
        -------
        - Async generator yielding text chunks for streaming to the UI.
        """
        try:
            if __task__ == "function_calling":
                # Open WebUI internal task – nothing to do here.
                return

            self.setup()
            start_time = time.time()

            # --- Resolve model IDs from valves ---
            mini_model_id = self.valves.MINI_MODEL
            small_model_id = self.valves.SMALL_MODEL
            medium_model_id = self.valves.MEDIUM_MODEL
            large_model_id = self.valves.LARGE_MODEL
            huge_model_id = self.valves.HUGE_MODEL  # Reserved for future use
            planning_model_id = self.valves.PLANNING_MODEL

            # --- OpenRouter models via OpenAI-compatible ChatOpenAI ---
            def LLM(model_id: str) -> ChatOpenAI:
                """Factory: create a chat model bound to OpenRouter with our headers."""
                return ChatOpenAI(
                    model=model_id,
                    base_url=self.valves.OPENROUTER_BASE_URL,
                    api_key=self.valves.OPENROUTER_API_KEY,
                    default_headers=self._headers or None,
                )

            planning_model = LLM(planning_model_id)
            small_model = LLM(small_model_id)
            medium_model = LLM(medium_model_id)
            large_model = LLM(large_model_id)

            config = {}  # Placeholder for future per-call config (LLM kwargs, tags, etc.)

            # ----------------------------------------------------------------------------------------
            # SPECIAL TASK: chat title generation
            # ----------------------------------------------------------------------------------------
            if __task__ == "title_generation":
                # Build a short context from the last user/assistant messages
                last_msgs = body.get("messages", [])[-4:]
                lc_msgs = to_lc_messages(last_msgs)
                # Ask for a concise title
                title_prompt = [
                    SystemMessage(
                        content="Write a concise 3–6 word title for this conversation. No quotes."
                    ),
                ] + lc_msgs
                content = small_model.invoke(title_prompt, config=config).content
                assert isinstance(content, str)
                yield content.strip()
                return

            # Prepare UI emitters
            send_citation = get_send_citation(__event_emitter__)
            send_status = get_send_status(__event_emitter__)

            # ----------------------------------------------------------------------------------------
            # STEP 1: PLANNING
            # ----------------------------------------------------------------------------------------
            # Concatenate the conversation into a compact text with the last user message normalized
            combined_message = ""
            msgs = body["messages"]
            for idx, message in enumerate(msgs):
                role = message["role"]
                message_content = message.get("content", "")
                content_to_use = ""
                if isinstance(message_content, str):
                    text_src = message_content
                    if idx == len(msgs) - 1:  # normalize last user message
                        text_src = self._normalize_for_safety(text_src)
                    # Trim long messages (show head/tail) to control prompt size
                    if len(text_src) > 1000:
                        mssg_length = len(text_src)
                        content_to_use = (
                            text_src[:500]
                            + "\n...(Middle of message cut by $NUMBER$)...\n"
                            + text_src[-500:]
                        )
                        new_mssg_length = len(content_to_use)
                        content_to_use = content_to_use.replace(
                            "$NUMBER$", str(mssg_length - new_mssg_length)
                        )
                    else:
                        content_to_use = text_src
                elif isinstance(message_content, list):
                    # Handle Open WebUI structured content
                    for part in message_content:
                        if part.get("type") == "text":
                            text = part.get("text", "")
                            if idx == len(msgs) - 1:
                                text = self._normalize_for_safety(text)
                            if len(text) > 1000:
                                mssg_length = len(text)
                                content_to_use += (
                                    text[:500]
                                    + "\n...(Middle of message cut by $NUMBER$)...\n"
                                    + text[-500:]
                                ).replace("$NUMBER$", str(mssg_length - 1000))
                            else:
                                content_to_use += text
                        if part.get("type") == "image_url":
                            content_to_use += "\nIMAGE FROM USER CUT HERE\n"
                combined_message += f'--- NEXT MESSAGE FROM "{str(role).upper()}" ---\n{content_to_use}\n--- DONE ---\n'

            planning_messages = [
                SystemMessage(content=PLANNING_PROMPT),
                HumanMessage(content=combined_message),
            ]

            await send_status("Planning...", False)
            planning_buffer = ""
            async for chunk in planning_model.astream(planning_messages, config=config):
                content = chunk.content
                assert isinstance(content, str)
                planning_buffer += content
            content = planning_buffer

            # Extract the planning <answer> tag with CSV of hashtags
            csv_hastag_list = re.findall(r"<answer>(.*?)</answer>", content)
            csv_hastag_list = csv_hastag_list[0] if csv_hastag_list else "unknown"

            # Model selection (map #mini/#small/#medium/#large -> IDs)
            if "#mini" in csv_hastag_list:
                model_to_use_id = mini_model_id
            elif "#small" in csv_hastag_list:
                model_to_use_id = small_model_id
            elif "#medium" in csv_hastag_list:
                model_to_use_id = medium_model_id
            elif "#large" in csv_hastag_list:
                model_to_use_id = large_model_id
            else:
                model_to_use_id = small_model_id  # sensible default

            is_reasoning_needed = "YES" if "#reasoning" in csv_hastag_list else "NO"

            # Tool plan (allow #no-tools override in last message)
            tool_list: List[str] = []
            last_msg_content = body["messages"][-1]["content"]
            allow_tools = True
            if isinstance(last_msg_content, str):
                allow_tools = "#no-tools" not in last_msg_content
            elif (
                isinstance(last_msg_content, list)
                and last_msg_content
                and last_msg_content[0].get("type") == "text"
            ):
                allow_tools = "#no-tools" not in last_msg_content[0].get("text", "")

            if allow_tools:
                if (
                    "#online" in csv_hastag_list
                    or (
                        isinstance(last_msg_content, str)
                        and "#online" in last_msg_content
                    )
                    or (
                        isinstance(last_msg_content, list)
                        and "#online" in last_msg_content[0].get("text", "")
                    )
                ):
                    tool_list.append("online")
                if (
                    "#wolfram" in csv_hastag_list
                    or (
                        isinstance(last_msg_content, str)
                        and "#wolfram" in last_msg_content
                    )
                    or (
                        isinstance(last_msg_content, list)
                        and "#wolfram" in last_msg_content[0].get("text", "")
                    )
                ):
                    tool_list.append("wolfram_alpha")
                if (
                    "#youtube" in csv_hastag_list
                    or (
                        isinstance(last_msg_content, str)
                        and "#youtube" in last_msg_content
                    )
                    or (
                        isinstance(last_msg_content, list)
                        and "#youtube" in last_msg_content[0].get("text", "")
                    )
                ):
                    tool_list.append("youtube")

            # Emit planning debug info to the UI (as a “citation” block)
            await send_citation(
                url=f"SMART Planning",
                title="SMART Planning",
                content=f"{content=}",
            )

            # User overrides (#!/#!!/#!!! for model; #yes/#no for reasoning)
            if isinstance(last_msg_content, str):
                lm = last_msg_content
            else:
                lm = (
                    last_msg_content[0].get("text", "")
                    if isinstance(last_msg_content, list) and last_msg_content
                    else ""
                )

            if "#!!!" in lm or "#large" in lm:
                model_to_use_id = large_model_id
            elif "#!!" in lm or "#medium" in lm:
                model_to_use_id = medium_model_id
            elif "#!" in lm or "#small" in lm:
                model_to_use_id = small_model_id

            if "#*yes" in lm or "#yes" in lm:
                is_reasoning_needed = "YES"
            elif "#*no" in lm or "#no" in lm:
                is_reasoning_needed = "NO"

            # Ensure we register at least one tool when using large model (keeps it useful)
            if model_to_use_id == large_model_id and len(tool_list) == 0:
                tool_list.append("dummy_tool")

            await send_status(
                status_message=f"Planning complete. Using Model: {model_to_use_id}. Reasoning needed: {is_reasoning_needed}.",
                done=True,
            )

            # ----------------------------------------------------------------------------------------
            # TOOL REGISTRATION
            # ----------------------------------------------------------------------------------------
            # Collect tools: include Open WebUI-provided `__tools__` plus our built-ins.
            tools = []
            if __tools__:
                for key, value in __tools__.items():
                    tools.append(
                        StructuredTool(
                            func=None,
                            name=key,
                            coroutine=value["callable"],
                            args_schema=value["pydantic_model"],
                            description=value["spec"]["description"],
                        )
                    )

            # Build structured tools from our coroutines based on plan
            if len(tool_list) > 0:
                for tool in tool_list:
                    if tool == "online":
                        online_funcs = [
                            (self.search_web, "Search the internet for information."),
                            (self.scrape_website, "Get the contents of a website/url."),
                        ]
                        for func, desc in online_funcs:
                            tools.append(
                                StructuredTool(
                                    func=None,
                                    name=func.__name__,
                                    coroutine=func,
                                    args_schema=self.create_pydantic_model_from_docstring(
                                        func
                                    ),
                                    description=desc,
                                )
                            )
                        self.SYSTEM_PROMPT_INJECTION = (
                            self.SYSTEM_PROMPT_INJECTION + PROMPT_WebSearch
                        )
                    if tool == "wolfram_alpha":
                        wolfram_funcs = [
                            (
                                self.wolframAlpha,
                                "Query the WolframAlpha knowledge engine to answer a wide variety of questions.",
                            )
                        ]
                        for func, desc in wolfram_funcs:
                            tools.append(
                                StructuredTool(
                                    func=None,
                                    name=func.__name__,
                                    coroutine=func,
                                    args_schema=self.create_pydantic_model_from_docstring(
                                        func
                                    ),
                                    description=desc,
                                )
                            )
                    if tool == "youtube":
                        youtube_funcs = [
                            (
                                self.youtube_search,
                                "Search YouTube for videos.",
                            ),
                            (
                                self.get_youtube_transcript,
                                "Get the transcript of a YouTube video.",
                            ),
                        ]
                        for func, desc in youtube_funcs:
                            tools.append(
                                StructuredTool(
                                    func=None,
                                    name=func.__name__,
                                    coroutine=func,
                                    args_schema=self.create_pydantic_model_from_docstring(
                                        func
                                    ),
                                    description=desc,
                                )
                            )
                        self.SYSTEM_PROMPT_INJECTION += PROMPT_YouTube
                    if tool == "dummy_tool":
                        dummy_funcs = [
                            (
                                self.dummy_tool,
                                "This is a dummy tool that does nothing. It is used when the user hasn't assigned any tools.",
                            )
                        ]
                        for func, desc in dummy_funcs:
                            tools.append(
                                StructuredTool(
                                    func=None,
                                    name=func.__name__,
                                    coroutine=func,
                                    args_schema=self.create_pydantic_model_from_docstring(
                                        func
                                    ),
                                    description=desc,
                                )
                            )

            # Visibility: tell the UI which tools the agent registered
            try:
                await send_status(f"Registered tools: {[t.name for t in tools]}", False)
            except Exception:
                pass

            model_to_use = LLM(model_to_use_id)

            messages_to_use = body["messages"]
            last_message_json = isinstance(messages_to_use[-1].get("content", ""), list)

            # ----------------------------------------------------------------------------------------
            # FAST PATH: NO REASONING
            # ----------------------------------------------------------------------------------------
            if is_reasoning_needed == "NO":
                # If tools were planned, force at least one tool call; otherwise, explicitly allow no tools.
                FORCE_TOOLS = ""
                if tool_list:
                    FORCE_TOOLS = (
                        "\nYou MUST call at least one of the provided tools (#online / #youtube / #wolfram) "
                        "before answering. Do not rely on prior knowledge alone. If tools are unavailable, "
                        "reply exactly with: TOOL_UNAVAILABLE.\n"
                    )

                # Inject final-agent pre-prompts & tool pre-prompts
                messages_to_use[0]["content"] = (
                    messages_to_use[0]["content"]
                    + USER_INTERACTION_PROMPT
                    + self.SYSTEM_PROMPT_INJECTION
                    + FORCE_TOOLS
                )

                # Sanitize control tags from user input
                def strip_tags(txt: str) -> str:
                    return (
                        str(txt)
                        .replace("#*yes", "")
                        .replace("#*no", "")
                        .replace("#!!", "")
                        .replace("#!", "")
                        .replace("#!!!", "")
                        .replace("#no", "")
                        .replace("#yes", "")
                        .replace("#large", "")
                        .replace("#medium", "")
                        .replace("#small", "")
                        .replace("#online", "")
                        .replace("#wolfram", "")
                        .replace("#youtube", "")
                        .replace("#no-tools", "")
                    )

                if not last_message_json:
                    norm = self._normalize_for_safety(messages_to_use[-1]["content"])
                    messages_to_use[-1]["content"] = strip_tags(norm)
                else:
                    norm = self._normalize_for_safety(
                        messages_to_use[-1]["content"][0]["text"]
                    )
                    messages_to_use[-1]["content"][0]["text"] = strip_tags(norm)

                # Create a ReAct agent with registered tools
                graph = create_react_agent(model_to_use, tools=tools)
                inputs = {"messages": body["messages"]}

                num_tool_calls = 0
                tool_error_seen = False
                async for event in graph.astream_events(
                    inputs, version="v2", config=config
                ):
                    if num_tool_calls >= 6:
                        yield "[TOO MANY TOOL CALLS - AGENT TERMINATED]"
                        break
                    kind = event["event"]
                    data = event["data"]
                    if kind == "on_chat_model_stream":
                        if "chunk" in data and (content := data["chunk"].content):
                            yield content
                    elif kind == "on_tool_start":
                        yield "\n"
                        await send_status(f"Running tool {event['name']}", False)
                    elif kind == "on_tool_end":
                        num_tool_calls += 1
                        await send_status(
                            f"Tool '{event['name']}' returned {data.get('output')}",
                            True,
                        )
                        await send_citation(
                            url=f"Tool call {num_tool_calls}",
                            title=event["name"],
                            content=f"Tool '{event['name']}' with inputs {data.get('input')} returned {data.get('output')}",
                        )
                    elif kind == "on_tool_error":
                        tool_error_seen = True
                        await send_status(
                            f"Tool '{event['name']}' error: {data.get('error')}", True
                        )
                        await send_citation(
                            url="Tool error",
                            title=event["name"],
                            content=str(data),
                        )

                # Fallback: planner wanted tools but none were called
                if num_tool_calls == 0 and tool_list:
                    # Pull normalized, stripped user text
                    if isinstance(lm, str):
                        qtxt = lm
                    else:
                        qtxt = lm[0].get("text", "") if lm else ""
                    qtxt = self._normalize_for_safety(qtxt)
                    qtxt = (
                        qtxt.replace("#online", "")
                        .replace("#youtube", "")
                        .replace("#wolfram", "")
                        .replace("#no-tools", "")
                    )

                    if "youtube" in tool_list:
                        y = await self.youtube_search(query=qtxt, max_results=5)
                        yield "\n\n[Fallback: YouTube search]\n" + y
                    if "online" in tool_list:
                        s = await self.search_web(query=qtxt, country="AR", focus="web")
                        yield "\n\n[Fallback: Web search]\n" + s

                await send_status(
                    status_message=f"Done! Took: {round(time.time() - start_time, 1)}s. Used {model_to_use_id}. Reasoning was not used.",
                    done=True,
                )
                return

            # ----------------------------------------------------------------------------------------
            # REASONING PATH
            # ----------------------------------------------------------------------------------------
            elif is_reasoning_needed == "YES":
                reasoning_model_id = self.valves.REASONING_MODEL
                reasoning_model = LLM(reasoning_model_id)

                # Compact history for the reasoner (user and assistant snippets)
                reasoning_context = ""
                for msg in body["messages"][:-1]:
                    if msg["role"] == "user":
                        txt = (
                            msg["content"]
                            if isinstance(msg["content"], str)
                            else msg["content"][0].get("text", "")
                        )
                        text_msg = (
                            txt
                            if len(txt) <= 400
                            else txt[:250] + "\n...(Middle cut)...\n" + txt[-100:]
                        )
                        reasoning_context += f"--- NEXT MESSAGE FROM \"{msg['role'].upper()}\" ---\n{text_msg}"
                    if msg["role"] == "assistant":
                        txt = (
                            msg["content"]
                            if isinstance(msg["content"], str)
                            else msg["content"][0].get("text", "")
                        )
                        text_msg = (
                            txt
                            if len(txt) <= 250
                            else txt[:150] + "\n...(Middle cut)...\n" + txt[-50:]
                        )
                        reasoning_context += f"--- NEXT MESSAGE FROM \"{msg['role'].upper()}\" ---\n{text_msg}"

                last_msg = body["messages"][-1]
                if last_msg["role"] == "user" and not last_message_json:
                    reasoning_context += f"--- LAST USER MESSAGE/PROMPT ---\n{self._normalize_for_safety(last_msg['content'])}"
                elif last_msg["role"] == "user":
                    reasoning_context += f"--- LAST USER MESSAGE/PROMPT ---\n{self._normalize_for_safety(last_msg['content'][0]['text'])}"

                # Remove control tags from the context the reasoner sees
                for tag in [
                    "#*yes",
                    "#*no",
                    "#!!",
                    "#!",
                    "#!!!",
                    "#no",
                    "#yes",
                    "#large",
                    "#medium",
                    "#small",
                    "#online",
                    "#wolfram",
                    "#youtube",
                    "#no-tools",
                ]:
                    reasoning_context = reasoning_context.replace(tag, "")

                reasoning_messages = [
                    SystemMessage(content=REASONING_PROMPT),
                    HumanMessage(content=reasoning_context),
                ]

                await send_status("Reasoning...", False)
                reasoning_buffer = ""
                update_status = 0
                async for chunk in reasoning_model.astream(
                    reasoning_messages, config=config
                ):
                    content = chunk.content
                    assert isinstance(content, str)
                    reasoning_buffer += content
                    update_status += 1
                    if update_status >= 5:
                        update_status = 0
                        await send_status(
                            status_message=f"Reasoning ({len(reasoning_buffer)})... {reasoning_buffer[-100:]}",
                            done=False,
                        )

                await send_status(
                    status_message=f"Reasoning ({len(reasoning_buffer)})... done",
                    done=True,
                )

                reasoning_content = reasoning_buffer
                full_content = (
                    "<reasoning_agent_output>\n"
                    + reasoning_content
                    + "\n</reasoning_agent_output>"
                )

                await send_citation(
                    url=f"SMART Reasoning",
                    title="SMART Reasoning",
                    content=f"{reasoning_content=}",
                )

                # If the reasoner asked for tools, run a mini tool-use agent
                tool_agent_content = re.findall(
                    r"<ask_tool_agent>(.*?)</ask_tool_agent>",
                    reasoning_content,
                    re.DOTALL,
                )

                if len(tool_agent_content) > 0:
                    await send_status(f"Running tool-agent...", False)
                    tool_message = [
                        {"role": "system", "content": TOOL_PROMPT},
                        {
                            "role": "user",
                            "content": "<reasoning_agent_requests>\n"
                            + str(tool_agent_content)
                            + "\n</reasoning_agent_requests>",
                        },
                    ]

                    if not __tools__ and not tools:
                        tool_agent_response = "Tool agent could not use any tools because the user did not enable any."
                    else:
                        graph = create_react_agent(large_model, tools=tools)
                        inputs = {"messages": tool_message}
                        message_buffer = ""
                        num_tool_calls = 0
                        async for event in graph.astream_events(
                            inputs, version="v2", config=config
                        ):
                            if num_tool_calls > 3:
                                message_buffer += (
                                    "\n[TOO MANY TOOL CALLS - AGENT TERMINATED]"
                                )
                                break
                            kind = event["event"]
                            data = event["data"]
                            if kind == "on_chat_model_stream":
                                if "chunk" in data and (
                                    content := data["chunk"].content
                                ):
                                    message_buffer += content
                            elif kind == "on_tool_start":
                                message_buffer += "\n"
                                await send_status(
                                    f"Running tool {event['name']}", False
                                )
                            elif kind == "on_tool_end":
                                num_tool_calls += 1
                                await send_status(
                                    f"Tool '{event['name']}' returned {data.get('output')}",
                                    True,
                                )
                                await send_citation(
                                    url=f"Tool call {num_tool_calls}",
                                    title=event["name"],
                                    content=f"Tool '{event['name']}' with inputs {data.get('input')} returned {data.get('output')}",
                                )
                            elif kind == "on_tool_error":
                                await send_status(
                                    f"Tool '{event['name']}' error: {data.get('error')}",
                                    True,
                                )
                                await send_citation(
                                    url="Tool error",
                                    title=event["name"],
                                    content=str(data),
                                )

                        tool_agent_response = message_buffer

                    await send_citation(
                        url=f"SMART Tool-use",
                        title="SMART Tool-use",
                        content=f"{tool_agent_response=}",
                    )
                    full_content += (
                        "\n\n\n<tool_agent_output>\n"
                        + tool_agent_response
                        + "\n</tool_agent_output>"
                    )

                await send_status(status_message="Reasoning complete.", done=True)

                # Stitch reasoning context into final call (final agent sees both)
                def strip_tags(txt: str) -> str:
                    return (
                        str(txt)
                        .replace("#*yes", "")
                        .replace("#*no", "")
                        .replace("#!!", "")
                        .replace("#!", "")
                        .replace("#!!!", "")
                        .replace("#no", "")
                        .replace("#yes", "")
                        .replace("#large", "")
                        .replace("#medium", "")
                        .replace("#small", "")
                        .replace("#online", "")
                        .replace("#wolfram", "")
                        .replace("#youtube", "")
                        .replace("#no-tools", "")
                    )

                if not last_message_json:
                    messages_to_use[-1]["content"] = (
                        "<user_input>\n"
                        + strip_tags(
                            self._normalize_for_safety(messages_to_use[-1]["content"])
                        )
                        + "\n</user_input>\n\n"
                        + full_content
                    )
                else:
                    messages_to_use[-1]["content"][0]["text"] = (
                        "<user_input>\n"
                        + strip_tags(
                            self._normalize_for_safety(
                                messages_to_use[-1]["content"][0]["text"]
                            )
                        )
                        + "\n</user_input>\n\n"
                        + full_content
                    )

                messages_to_use[0]["content"] = (
                    messages_to_use[0]["content"]
                    + USER_INTERACTION_PROMPT
                    + self.SYSTEM_PROMPT_INJECTION
                )

                graph = create_react_agent(model_to_use, tools=tools)
                inputs = {"messages": messages_to_use}

                await send_status(
                    status_message=f"Starting answer with {model_to_use_id}...",
                    done=False,
                )

                num_tool_calls = 0
                async for event in graph.astream_events(
                    inputs, version="v2", config=config
                ):
                    if num_tool_calls >= 6:
                        await send_status(
                            status_message="Interupting due to max tool calls reached!",
                            done=True,
                        )
                        yield "[TOO MANY TOOL CALLS - AGENT TERMINATED]"
                        break
                    kind = event["event"]
                    data = event["data"]
                    if kind == "on_chat_model_stream":
                        if "chunk" in data and (content := data["chunk"].content):
                            yield content
                    elif kind == "on_tool_start":
                        yield "\n"
                        await send_status(f"Running tool {event['name']}", False)
                    elif kind == "on_tool_end":
                        num_tool_calls += 1
                        await send_status(
                            f"Tool '{event['name']}' returned {data.get('output')}",
                            True,
                        )
                        await send_citation(
                            url=f"Tool call {num_tool_calls}",
                            title=event["name"],
                            content=f"Tool '{event['name']}' with inputs {data.get('input')} returned {data.get('output')}",
                        )
                    elif kind == "on_tool_error":
                        await send_status(
                            f"Tool '{event['name']}' error: {data.get('error')}",
                            True,
                        )
                        await send_citation(
                            url="Tool error",
                            title=event["name"],
                            content=str(data),
                        )

                # Fallback if planner mandated tools but none were called
                if num_tool_calls == 0 and tool_list:
                    if isinstance(lm, str):
                        qtxt = lm
                    else:
                        qtxt = lm[0].get("text", "") if lm else ""
                    qtxt = self._normalize_for_safety(qtxt)
                    qtxt = (
                        qtxt.replace("#online", "")
                        .replace("#youtube", "")
                        .replace("#wolfram", "")
                        .replace("#no-tools", "")
                    )

                    if "youtube" in tool_list:
                        y = await self.youtube_search(query=qtxt, max_results=5)
                        yield "\n\n[Fallback: YouTube search]\n" + y
                    if "online" in tool_list:
                        s = await self.search_web(query=qtxt, country="AR", focus="web")
                        yield "\n\n[Fallback: Web search]\n" + s

                if not num_tool_calls >= 4:
                    await send_status(
                        status_message=f"Done! Took: {round(time.time() - start_time, 1)}s. Used {model_to_use_id}. Reasoning was used",
                        done=True,
                    )
                return

            else:
                yield "Error: is_reasoning_needed is not YES or NO"
                return

        except Exception as e:
            # Final catch-all (won’t crash the WebUI) – include message but not stack trace in stream
            yield "Error: " + str(e)
            traceback.print_exc()
            return

"""
title: SMART - Sequential Multi-Agent Reasoning Technique (OpenRouter edition)
author: MartianInGreen (modified by MichaelSParkin3)
author_url: https://github.com/MichaelSParkin3/Open-WebUI-SMART-Tools-OpenRouter
description: SMART is a sequential multi-agent reasoning technique. Uses OpenRouter + online/Wolfram tools.
git_url: https://github.com/you/my-super-tool
required_open_webui_version: 0.5.0
requirements: langchain-openai, langgraph==0.2.60, requests, pydantic>=2
version: 1.2
licence: MIT
"""

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
)

import requests
from pydantic import BaseModel, Field, create_model
from fastapi import Request

from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ---------------------------------------------------------------

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
        - Avalible tools are #online, #wolfram
        - Use #online to enable multiple tools such as Search and a Scraping tool. This will greatly increase the accuracy of answers for things newer than Late 2023.
        - Use #wolfram to enable access to Wolfram|Alpha, a powerful computational knowledge engine and scientific and real-time database.
            - Wolfram|Alpha is very good at math (integrals, roots, limits...), real-time data (weather, stocks, FX), and structured facts.
        - If the prompt involves math, enable #wolfram.
    
Example response:
<reasoning>
... 
(You are allowed new lines here)
</reasoning>
<answer>#medium, #online ,#no-reasoning</answer>
</system_instructions>"""

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
3. Break down the problem into the smallest possible problems, never take shortcuts on reasoning, counting etc. Everything needs to be explicitly stated. More output is better.
4. Think about steps you might need to take to solve this problem.
5. Think through these steps.
6. Backtrack and restart from different points as often as you need to. Always consider alternative approaches.
7. Validate your steps constantly. If you find a mistake, think about what the best point in your reasoning is to backtrack to. Don't be kind to yourself here. You need to critically analyze what you are doing.
</system_instructions>"""

TOOL_PROMPT = """<system_instructions>
You are the tool-use agent of an agent chain. You are the part of the LLM designed to use tools.
You will not directly interact with the user in any way. Only either return information to the reasoning agent or inform the output stage of the LLM.

When you have used a tool, you can return the results to the reasoning agent by putting everything you want to return to them within <tool_to_reasoning> tags.
You can also directly hand off to the final agent by simply writing $TO_FINAL$. You still need to write out what you want them to get!

Actually make use of the results you got. NEVER make more than 3 tool calls! If you called any tool 3 times, that's it!
You need to output everything you want to pass on. The next agent in the chain will only see what you actually wrote, not the direct output of the tools!

Please think about how best to call the tool first. Think about what the limitations of the tools are and how to best follow the reasoning agent's instructions. It's okay if you can't 100% produce what they wanted!
</system_instructions>"""

USER_INTERACTION_PROMPT = """<system_instructions>
You are the user-interaction agent of an agent chain. You are the part of the llm designed to interact with the user.

You should follow the pre-prompt given to you within <preprompt> tags.
<system_instructions>"""

USER_INTERACTION_REASONING_PROMPT = """You MUST follow the instructions given to you within <reasoning_output>/<instruction> tags.
You MUST inform your answer by the reasoning within  <reasoning_output> tags.
Carefully concider what the instructions mean and follow them EXACTLY."""

# --------------------------------------------------------------

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

PROMPT_WolframAlpha = """<wolframInstructions>
Wolfram|Alpha is an advanced computational knowledge engine and database with accurate scientific and real-time data. 
Keep queries concise (e.g., "integral of x^2", "weather Buenos Aires today"). Prefer simple inputs; do heavy plotting in Python (not available here).
</wolframInstructions>"""

# ---------------------------------------------------------------
# TOOLS
# --------------------------------------------------------------


def remove_html_tags(text):
    clean = re.compile("<.*?>")
    return re.sub(clean, "", text)


def decode_data(data):
    results = []
    try:
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
            print("Error in parsing infobox results...", str(e))

        try:
            for i, result in enumerate(data.get("web", {}).get("results", [])):
                if i >= 8:
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


def search_brave(query, country, language, focus, SEARCH_KEY):
    results_filter = "infobox"
    if focus in ("web", "all"):
        results_filter += ",web"
    if focus in ("news", "all"):
        results_filter += ",news"
    if focus == "videos":
        results_filter += ",videos"

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
    query, country, media_type, freshness=None, SEARCH_KEY=None
):
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
        "X-Subscription-Token": SEARCH_KEY,
    }

    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        if media_type == "images":
            formatted_data = {}
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
    SEARCH_KEY=None,
):
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
            results = search_brave(query, country, language, focus, SEARCH_KEY)
        else:
            # pass named params so 'freshness' stays None and the API key is passed correctly
            results = search_images_and_video(
                query=query,
                country=country,
                media_type=focus,
                freshness=None,
                SEARCH_KEY=SEARCH_KEY,
            )
    except Exception:
        return {"statusCode": 400, "body": json.dumps("Error fetching search results.")}
    return results


# ---------------------------------------------------------------

EmitterType = Optional[Callable[[dict], Awaitable[None]]]


class SendCitationType(Protocol):
    def __call__(self, url: str, title: str, content: str) -> Awaitable[None]: ...


class SendStatusType(Protocol):
    def __call__(self, status_message: str, done: bool) -> Awaitable[None]: ...


def get_send_citation(__event_emitter__: EmitterType) -> SendCitationType:
    async def send_citation(url: str, title: str, content: str):
        if __event_emitter__ is None:
            return
        await __event_emitter__(
            {
                "type": "citation",
                "data": {
                    "document": [content],
                    "metadata": [{"source": url, "html": False}],
                    "source": {"name": title},
                },
            }
        )

    return send_citation


def get_send_status(__event_emitter__: EmitterType) -> SendStatusType:
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
def to_lc_messages(messages: list[dict]):
    out = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if isinstance(content, list) and content and content[0].get("type") == "text":
            content = content[0].get("text", "")
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out


class Pipe:
    class Valves(BaseModel):
        try:
            OPENROUTER_API_KEY: str = Field(
                default="", description="OpenRouter API key"
            )
            MODEL_PREFIX: str = Field(
                default="SMART", description="Prefix before model ID"
            )
            # Default OpenRouter model IDs (edit these in the UI to your favorites)
            MINI_MODEL: str = Field(
                default="openai/gpt-4o-mini", description="Model for very small tasks"
            )
            SMALL_MODEL: str = Field(
                default="openai/gpt-4o-mini", description="Model for small tasks"
            )
            LARGE_MODEL: str = Field(
                default="openai/gpt-4o", description="Model for large tasks"
            )
            HUGE_MODEL: str = Field(
                default="anthropic/claude-3.7-sonnet",
                description="Model for the largest tasks",
            )
            REASONING_MODEL: str = Field(
                default="deepseek/deepseek-r1", description="Model for reasoning tasks"
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
            AGENT_NAME: str = Field(
                default="Smart/Core (OpenRouter)", description="Name of the agent"
            )
            AGENT_ID: str = Field(
                default="smart-core-or", description="ID of the agent"
            )
            # Optional: forwarded headers for OpenRouter analytics (referrer/title)
            HTTP_REFERER: str = Field(
                default="", description="Optional - HTTP-Referer header for OpenRouter"
            )
            X_TITLE: str = Field(
                default="SMART Pipe (OpenWebUI)",
                description="Optional - X-Title header for OpenRouter",
            )
        except Exception as e:
            traceback.print_exc()

    def __init__(self):
        try:
            self.type = "manifold"
            self.valves = self.Valves(
                **{
                    k: os.getenv(k, v.default)
                    for k, v in self.Valves.model_fields.items()
                }
            )
            os.environ["BRAVE_SEARCH_TOKEN"] = self.valves.BRAVE_SEARCH_KEY
            os.environ["BRAVE_SEARCH_TOKEN_SECONDARY"] = self.valves.BRAVE_SEARCH_KEY
        except Exception:
            traceback.print_exc()

    def pipes(self) -> list[dict[str, str]]:
        try:
            self.setup()
        except Exception as e:
            traceback.print_exc()
            return [{"id": "error", "name": f"Error: {e}"}]
        return [{"id": self.valves.AGENT_ID, "name": self.valves.AGENT_NAME}]

    def setup(self):
        v = self.valves
        if not v.OPENROUTER_API_KEY:
            raise Exception("Error: OPENROUTER_API_KEY is not set")
        # LangChain’s ChatOpenAI can target OpenRouter via base_url+api_key
        self.llm_kwargs = {
            "api_key": v.OPENROUTER_API_KEY,
            "base_url": "https://openrouter.ai/api/v1",
        }
        # Optional headers for OpenRouter metrics
        default_headers = {}
        if v.HTTP_REFERER:
            default_headers["HTTP-Referer"] = v.HTTP_REFERER
        if v.X_TITLE:
            default_headers["X-Title"] = v.X_TITLE
        if default_headers:
            self.llm_kwargs["default_headers"] = default_headers
        self.SYSTEM_PROMPT_INJECTION = ""

    # ---------- Tools (async) ----------

    async def search_web(
        self, query: str, country: str = "US", language: str = "en", focus: str = "all"
    ):
        """Search via Brave.
        :param query: Search query text
        :param country: Two-letter country code (e.g., US)
        :param language: Language code (e.g., en)
        :param focus: one of all|web|news|wikipedia|academia|reddit|images|videos
        """
        results = searchWeb(
            query, country, language, focus, self.valves.BRAVE_SEARCH_KEY
        )
        return json.dumps(results)

    async def scrape_website(self, url: str) -> str:
        """Scrape page text via Jina Reader.
        :param url: Full http(s) URL to fetch
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

    async def dummy_tool(self):
        """No-op tool used when no tools are enabled."""
        return "You have not assigned any tools to use."

    # ---------- Pydantic schema builder (fixed) ----------
    def create_pydantic_model_from_docstring(self, func):
        doc = inspect.getdoc(func) or ""
        # Parse :param name: description (optional, for nicer help text)
        param_descriptions = {}
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

    # -----------------------------------------------------

    async def pipe(
        self,
        body: dict,
        __request__: Request,
        __user__: dict | None,
        __task__: str | None,
        __tools__: dict[str, dict] | None,
        __event_emitter__: Callable[[dict], Awaitable[None]] | None,
    ) -> AsyncGenerator:
        try:
            if __task__ == "function_calling":
                return

            self.setup()
            start_time = time.time()

            mini_model_id = self.valves.MINI_MODEL
            small_model_id = self.valves.SMALL_MODEL
            large_model_id = self.valves.LARGE_MODEL
            huge_model_id = self.valves.HUGE_MODEL
            planning_model_id = self.valves.PLANNING_MODEL

            # --- OpenRouter models via ChatOpenAI ---
            planning_model = ChatOpenAI(model=planning_model_id, **self.llm_kwargs)  # type: ignore
            small_model = ChatOpenAI(model=small_model_id, **self.llm_kwargs)  # type: ignore
            large_model = ChatOpenAI(model=large_model_id, **self.llm_kwargs)  # type: ignore

            config = {}

            if __task__ == "title_generation":
                # Build a short context from the last user/assistant messages
                last_msgs = body.get("messages", [])[-4:]
                lc_msgs = to_lc_messages(last_msgs)
                # Simple heuristic: ask for a concise chat title
                title_prompt = [
                    SystemMessage(
                        content="Write a concise 3–6 word title for this conversation. No quotes."
                    ),
                ] + lc_msgs
                content = small_model.invoke(title_prompt, config=config).content
                assert isinstance(content, str)
                yield content.strip()
                return

            send_citation = get_send_citation(__event_emitter__)
            send_status = get_send_status(__event_emitter__)

            #
            # STEP 1: Planning
            #
            combined_message = ""
            for message in body["messages"]:
                role = message["role"]
                message_content = message.get("content", "")
                content_to_use = ""
                if isinstance(message_content, str):
                    if len(message_content) > 1000:
                        mssg_length = len(message_content)
                        content_to_use = (
                            message_content[:500]
                            + "\n...(Middle of message cut by $NUMBER$)...\n"
                            + message_content[-500:]
                        )
                        new_mssg_length = len(content_to_use)
                        content_to_use = content_to_use.replace(
                            "$NUMBER$", str(mssg_length - new_mssg_length)
                        )
                    else:
                        content_to_use = message_content
                elif isinstance(message_content, list):
                    for part in message_content:
                        if part.get("type") == "text":
                            text = part.get("text", "")
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

            csv_hastag_list = re.findall(r"<answer>(.*?)</answer>", content)
            csv_hastag_list = csv_hastag_list[0] if csv_hastag_list else "unknown"

            # model selection
            if "#mini" in csv_hastag_list:
                model_to_use_id = mini_model_id
            elif "#small" in csv_hastag_list:
                model_to_use_id = small_model_id
            elif "#medium" in csv_hastag_list:
                model_to_use_id = large_model_id
            elif "#large" in csv_hastag_list:
                model_to_use_id = huge_model_id
            else:
                model_to_use_id = small_model_id

            is_reasoning_needed = "YES" if "#reasoning" in csv_hastag_list else "NO"

            tool_list = []
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

            await send_citation(
                url=f"SMART Planning",
                title="SMART Planning",
                content=f"{content=}",
            )

            # user overrides
            if isinstance(last_msg_content, str):
                lm = last_msg_content
            else:
                lm = (
                    last_msg_content[0].get("text", "")
                    if isinstance(last_msg_content, list) and last_msg_content
                    else ""
                )

            if "#!!!" in lm or "#large" in lm:
                model_to_use_id = huge_model_id
            elif "#!!" in lm or "#medium" in lm:
                model_to_use_id = large_model_id
            elif "#!" in lm or "#small" in lm:
                model_to_use_id = small_model_id

            if "#*yes" in lm or "#yes" in lm:
                is_reasoning_needed = "YES"
            elif "#*no" in lm or "#no" in lm:
                is_reasoning_needed = "NO"

            if model_to_use_id == huge_model_id and len(tool_list) == 0:
                tool_list.append("dummy_tool")

            await send_status(
                status_message=f"Planning complete. Using Model: {model_to_use_id}. Reasoning needed: {is_reasoning_needed}.",
                done=True,
            )

            # Collect tools: include any OpenWebUI-provided __tools__ plus our built-ins
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

            model_to_use = ChatOpenAI(model=model_to_use_id, **self.llm_kwargs)  # type: ignore

            messages_to_use = body["messages"]
            last_message_json = isinstance(messages_to_use[-1].get("content", ""), list)

            # Fast path: NO reasoning
            if is_reasoning_needed == "NO":
                messages_to_use[0]["content"] = (
                    messages_to_use[0]["content"]
                    + USER_INTERACTION_PROMPT
                    + self.SYSTEM_PROMPT_INJECTION
                )

                # sanitize control tags from final user message
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
                        .replace("#no-tools", "")
                    )

                if not last_message_json:
                    messages_to_use[-1]["content"] = strip_tags(
                        messages_to_use[-1]["content"]
                    )
                else:
                    messages_to_use[-1]["content"][0]["text"] = strip_tags(
                        messages_to_use[-1]["content"][0]["text"]
                    )

                graph = create_react_agent(model_to_use, tools=tools)
                inputs = {"messages": body["messages"]}

                num_tool_calls = 0
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

                await send_status(
                    status_message=f"Done! Took: {round(time.time() - start_time, 1)}s. Used {model_to_use_id}. Reasoning was not used.",
                    done=True,
                )
                return

            # Reasoning path
            elif is_reasoning_needed == "YES":
                reasoning_model_id = self.valves.REASONING_MODEL
                reasoning_model = ChatOpenAI(model=reasoning_model_id, **self.llm_kwargs)  # type: ignore

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
                    reasoning_context += (
                        f"--- LAST USER MESSAGE/PROMPT ---\n{last_msg['content']}"
                    )
                elif last_msg["role"] == "user":
                    reasoning_context += f"--- LAST USER MESSAGE/PROMPT ---\n{last_msg['content'][0]['text']}"

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
                    + "\n<reasoning_agent_output>"
                )

                await send_citation(
                    url=f"SMART Reasoning",
                    title="SMART Reasoning",
                    content=f"{reasoning_content=}",
                )

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
                        async for event in graph.astream_events(inputs, version="v2", config=config):  # type: ignore
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

                        tool_agent_response = message_buffer

                    await send_citation(
                        url=f"SMART Tool-use",
                        title="SMART Tool-use",
                        content=f"{tool_agent_response=}",
                    )
                    full_content += (
                        "\n\n\n<tool_agent_output>\n"
                        + tool_agent_response
                        + "\n<tool_agent_output>"
                    )

                await send_status(status_message="Reasoning complete.", done=True)

                # Stitch reasoning context into final call
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
                        .replace("#no-tools", "")
                    )

                if not last_message_json:
                    messages_to_use[-1]["content"] = (
                        "<user_input>\n"
                        + strip_tags(messages_to_use[-1]["content"])
                        + "\n</user_input>\n\n"
                        + full_content
                    )
                else:
                    messages_to_use[-1]["content"][0]["text"] = (
                        "<user_input>\n"
                        + strip_tags(messages_to_use[-1]["content"][0]["text"])
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
                async for event in graph.astream_events(inputs, version="v2", config=config):  # type: ignore
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
            yield "Error: " + str(e)
            traceback.print_exc()
            return

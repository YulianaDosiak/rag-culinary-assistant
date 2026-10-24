"""One interface, two backends: Gemini (default) or Lightning AI.

    from llm_client import LLMClient
    llm = LLMClient()                  # reads LLM_PROVIDER from .env, defaults to "gemini"
    llm = LLMClient(provider="lightning")   # or force one explicitly

`context_lab.py` and `prompt_lab.py` are both built on this, so flipping

    LLM_PROVIDER=lightning

in `.env` re-points every notebook at Lightning AI without touching a single
notebook cell. This exists because of the Gemini free tier's ~6 requests/minute
ceiling (§11.1) -- Lightning is the fallback, not a replacement; Week 2's
material is written and measured against Gemini.

Capability matrix -- verified 2026-09-13, gemini-3.5-flash vs openai/gpt-5-nano
--------------------------------------------------------------------------------
                        Gemini                      Lightning AI
system instruction     native                       native (system role)
temperature            native                       native
structured output      response_schema, incl.       response_format json_schema.
                       bare `list[Model]`            Top-level MUST be an object
                                                      (verified: bare array schema
                                                      -> HTTP 400). This module
                                                      wraps/unwraps lists so the
                                                      call site is identical on
                                                      both backends -- see _schema.
                                                      Sent with strict: true
                                                      (2026-10-04); without it the
                                                      schema is not enforced.
max output tokens      max_output_tokens             max_completion_tokens
                                                      (verified: `max_tokens` ->
                                                      HTTP 500 naming the field)
thinking / reasoning    thinking_level: MINIMAL/      reasoning_effort accepted
control                LOW/MEDIUM/HIGH, effect       (none/low/medium/high) but
                       reliably measured (see the    ITS EFFECT WAS NOT CONFIRMED:
                       Week 2 deck)                   three back-to-back calls at
                                                      none/low and unset all
                                                      returned reasoning_tokens=0,
                                                      while separate identical
                                                      calls with no parameter at
                                                      all varied between 0 and 64.
                                                      Treat `thinking=` on this
                                                      backend as passed-through,
                                                      not as a verified dial.
free token counting    count_tokens(), verified       NO equivalent found. counttokens()
                       free of the 6/min budget       raises on this backend rather
                       (10 calls in 2.1s)              than silently spending a
                                                      real, billed call and calling
                                                      it free.
three-way usage split  prompt/thoughts/candidates     prompt_tokens / completion_
                       _token_count                   tokens_details.reasoning_
                                                      tokens / (completion -
                                                      reasoning) -- same shape,
                                                      confirmed present on every
                                                      call, including reasoning
                                                      variance run to run: two
                                                      identical prompts produced
                                                      0 and 64 reasoning tokens.
"""

import os
import time
import typing
from dataclasses import dataclass
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Usage:
    prompt_tokens: int
    thinking_tokens: int
    answer_tokens: int


@dataclass
class ChatResult:
    text: str
    usage: Usage
    parsed: Any = None       # populated only when a schema was requested
    raw: Any = None          # the untouched provider response, for debugging


class LLMClient:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = (provider or os.environ.get("LLM_PROVIDER", "gemini")).lower()
        self.calls = 0

        if self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "lightning":
            self._init_lightning()
        else:
            raise ValueError(f"Unknown LLM_PROVIDER {self.provider!r}. Use 'gemini' or 'lightning'.")
        if model:
            self.model = model   # overrides the provider's pinned model

    # ------------------------------------------------------------ gemini
    def _init_gemini(self) -> None:
        from google import genai

        self.model = "gemini-3.5-flash"
        self._genai = genai
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def _chat_gemini(self, prompt, *, system, temperature, schema, max_tokens, thinking):
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json" if schema else None,
            response_schema=schema,
            thinking_config=types.ThinkingConfig(thinking_level=thinking.upper()) if thinking else None,
        )

        for attempt in range(1, 7):
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                self.calls += 1
                break
            except errors.ClientError as exc:
                if exc.code != 429:
                    raise
                print(f"  [rate limited: ~6 requests/minute. waiting 25s, attempt {attempt}]")
                time.sleep(25)
            except errors.ServerError:
                wait = min(4 * 2 ** (attempt - 1), 60)
                print(f"  [503, model busy. waiting {wait}s, attempt {attempt}/6]")
                time.sleep(wait)
        else:
            raise RuntimeError(f"{self.model} stayed unavailable across 6 attempts.")

        usage = response.usage_metadata
        return ChatResult(
            text=(response.text or "").strip(),
            usage=Usage(
                prompt_tokens=usage.prompt_token_count or 0,
                thinking_tokens=usage.thoughts_token_count or 0,
                answer_tokens=usage.candidates_token_count or 0,
            ),
            parsed=getattr(response, "parsed", None),
            raw=response,
        )

    def _count_tokens_gemini(self, text: str) -> int:
        return self._client.models.count_tokens(model=self.model, contents=text).total_tokens

    # ---------------------------------------------------------- lightning
    def _init_lightning(self) -> None:
        key = os.environ.get("LIGHTNING_API_KEY")
        if not key:
            raise SystemExit("No LIGHTNING_API_KEY in .env  (Lightning AI -> Global Settings -> Keys)")
        self.model = "openai/gpt-5-nano"
        self._headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self._url = "https://lightning.ai/api/v1/chat/completions"

    def _chat_lightning(self, prompt, *, system, temperature, schema, max_tokens, thinking):
        import requests

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict = {"model": self.model, "messages": messages}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_completion_tokens"] = max_tokens
        if thinking is not None:
            # Accepted by the API; its effect was NOT confirmed in testing -- see
            # the module docstring. Passed through rather than promised.
            body["reasoning_effort"] = {"minimal": "none"}.get(thinking.lower(), thinking.lower())

        wrapped_list = False
        if schema is not None:
            json_schema, wrapped_list = _to_json_schema(schema)
            # strict: True is what makes the schema binding. Without it the schema
            # is only a hint: verified 2026-10-04 on openai/gpt-5-nano, a Literal
            # "Topic" came back as "topic", and one reply echoed the schema itself.
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True,
                                "schema": _strict(json_schema)},
            }

        response = requests.post(self._url, headers=self._headers, json=body)
        response.raise_for_status()
        self.calls += 1
        result = response.json()

        content = result["choices"][0]["message"].get("content") or ""
        usage = result["usage"]
        reasoning = usage["completion_tokens_details"]["reasoning_tokens"]

        parsed = None
        if schema is not None and content:
            parsed = _parse_schema(schema, content, wrapped_list)

        return ChatResult(
            text=content.strip(),
            usage=Usage(
                prompt_tokens=usage["prompt_tokens"],
                thinking_tokens=reasoning,
                answer_tokens=usage["completion_tokens"] - reasoning,
            ),
            parsed=parsed,
            raw=result,
        )

    # ---------------------------------------------------------------- api
    def chat(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        schema=None,
        max_tokens: Optional[int] = None,
        thinking: Optional[str] = None,
    ) -> ChatResult:
        """thinking accepts "minimal"/"low"/"medium"/"high" on either backend.
        See the module docstring for what is and is not verified per backend.
        """
        if self.provider == "gemini":
            return self._chat_gemini(
                prompt, system=system, temperature=temperature, schema=schema,
                max_tokens=max_tokens, thinking=thinking,
            )
        return self._chat_lightning(
            prompt, system=system, temperature=temperature, schema=schema,
            max_tokens=max_tokens, thinking=thinking,
        )

    def count_tokens(self, text: str) -> int:
        """Free on Gemini. No equivalent exists on Lightning -- see the module
        docstring. Raises rather than silently making a real, billed call and
        calling it free."""
        if self.provider != "gemini":
            raise NotImplementedError(
                "count_tokens() has no free equivalent on the lightning backend. "
                "Switch LLM_PROVIDER=gemini for this cell, or accept a real "
                "(billed) call -- there is no free tokenizer endpoint to call instead."
            )
        return self._count_tokens_gemini(text)


# --------------------------------------------------------------- schema helpers
def _to_json_schema(schema):
    """pydantic model or list[model] -> (json_schema dict, was_top_level_a_list).

    Lightning's response_format requires a top-level object (verified: a bare
    array schema returns HTTP 400, "schema must be a JSON Schema of type
    object"). Gemini has no such restriction. So a `list[Model]` schema is
    wrapped here as {"items": [...]} for the Lightning call, and unwrapped
    again in _parse_schema -- the call site (context_lab.py) asks for
    `list[Risk]` either way and gets a `list[Risk]` back either way.
    """
    from pydantic import TypeAdapter

    origin = typing.get_origin(schema)
    is_list = origin in (list, typing.List)
    adapter = TypeAdapter(schema)
    json_schema = adapter.json_schema()

    if is_list:
        # A $ref such as "#/$defs/Risk" resolves from the ROOT of the schema, so the
        # definitions must move up to the wrapper rather than stay inside "items".
        defs = json_schema.pop("$defs", None)
        wrapper = {"type": "object", "properties": {"items": json_schema}, "required": ["items"]}
        if defs:
            wrapper["$defs"] = defs
        return wrapper, True
    return json_schema, False


def _strict(node):
    """Rewrite a JSON schema into the form OpenAI-compatible strict mode accepts.

    Strict mode requires every object to be closed (additionalProperties: false)
    and to list every property as required, and it rejects `default`. Pydantic's
    schemas have neither property and may carry defaults, so they are adjusted here.

    Pydantic also writes a one-value Literal["Topic"] as {"const": "Topic"}, and
    the Lightning backend does not enforce `const`: it returned "topic"
    (2026-10-04). An `enum` of one value says the same and is enforced.
    """
    if isinstance(node, dict):
        node.pop("default", None)
        if "const" in node:
            node["enum"] = [node.pop("const")]
        if node.get("type") == "object":
            node["additionalProperties"] = False
            node["required"] = list(node.get("properties", {}))
        for value in node.values():
            _strict(value)
    elif isinstance(node, list):
        for value in node:
            _strict(value)
    return node


def _parse_schema(schema, content: str, wrapped_list: bool):
    import json

    from pydantic import TypeAdapter

    data = json.loads(content)
    if wrapped_list:
        data = data["items"]
    return TypeAdapter(schema).validate_python(data)

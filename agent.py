import json
import time

from pydantic import BaseModel, Field

from agent_tools import TOOLS, TOOL_DESCRIPTIONS, run_tool
from memory_store import MemoryStore

LOG_FILE = "agent_log.jsonl"

SYSTEM = """You are a culinary assistant agent working over two cookbooks (a knowledge graph plus text chunks).
Each step you return JSON: either call ONE tool, or give the final answer.

{tools}

RULES
- Answer ONLY from tool results. Cite chunk ids like [recipes.txt_chunk_4] when tool results show them.
- If the tools do not give the answer, the final answer must be exactly: "I don't have this information in the provided sources."
- For "which recipes contain/use/need X" use find_recipes_by_entity first. For quantities changes use scale_recipe. Never compute amounts yourself.
- Do not repeat an identical tool call. Do not call a tool when you already have enough to answer.
- Follow-up questions ("it", "that recipe", "double it"): resolve them from the conversation history.
- Memory: when the user states a lasting personal fact (diet, allergies, disliked food, equipment they own), call save_memory with a short stable key (diet, allergies, equipment...) and mention in the final answer that you noted it. If a fact changes, call save_memory with the SAME key; if it is no longer true, call delete_memory. Never store questions or temporary things.
- Use the known facts about the user below to personalize answers (e.g. avoid allergens).
- Keep answers short and in English.

KNOWN FACTS ABOUT THE USER:
{memory}"""


class AgentStep(BaseModel):
    thought: str = Field(description="One short sentence: what you need next and why.")
    action: str = Field(description="Either 'call_tool' or 'final_answer'.")
    tool: str = Field(description="Tool name when action is call_tool, otherwise an empty string.")
    args_json: str = Field(description='JSON object with the tool arguments, e.g. {"query": "quinoa"}. Use {} for final_answer.')
    answer: str = Field(description="The final answer when action is final_answer, otherwise an empty string.")


class Agent:
    def __init__(self, store, llm, memory: MemoryStore, max_model_calls=5, max_calls_per_tool=2,
                 history_turns=6, pause=0.0):
        self.store, self.llm, self.memory = store, llm, memory
        self.max_model_calls = max_model_calls
        self.max_calls_per_tool = max_calls_per_tool
        self.history_turns = history_turns
        self.pause = pause
        self.history: list[tuple[str, str]] = []

    # ---------------------------------------------------------------- conversation
    def reset_conversation(self):
        """New conversation: history is cleared, long-term memory stays."""
        self.history = []

    def _history_text(self) -> str:
        if not self.history:
            return "(this is the first message)"
        return "\n".join(f"User: {q}\nAssistant: {a}" for q, a in self.history[-self.history_turns:])

    def _remember_turn(self, q, a):
        self.history.append((q, a if len(a) <= 800 else a[:800] + "..."))
        self.history = self.history[-self.history_turns:]   # older turns are dropped

    # ---------------------------------------------------------------- one request
    def _prompt(self, question, scratch, last_warning):
        parts = [f"CONVERSATION SO FAR:\n{self._history_text()}", f"CURRENT QUESTION:\n{question}"]
        if scratch:
            parts.append("STEPS ALREADY DONE IN THIS REQUEST:\n" + "\n\n".join(scratch))
        if last_warning:
            parts.append("This is your LAST allowed step. You must choose final_answer now, using what you have.")
        parts.append("Decide the next step.")
        return "\n\n".join(parts)

    @staticmethod
    def _parse(res) -> AgentStep:
        parsed = getattr(res, "parsed", None)
        if isinstance(parsed, AgentStep):
            return parsed
        if isinstance(parsed, dict):
            return AgentStep(**parsed)
        return AgentStep(**json.loads(res.text))

    def run(self, question: str) -> dict:
        system = SYSTEM.format(tools=TOOL_DESCRIPTIONS, memory=self.memory.as_prompt())
        trace, scratch, tool_counts = [], [], {}
        model_calls, status, answer = 0, "ok", None

        for call_no in range(1, self.max_model_calls + 1):
            is_last = call_no == self.max_model_calls
            prompt = self._prompt(question, scratch, last_warning=is_last and call_no > 1)
            try:
                res = self.llm.chat(prompt=prompt, system=system, temperature=0.0, schema=AgentStep)
                model_calls += 1
                step = self._parse(res)
            except Exception as e:
                status = "error"
                answer = f"Model error: {e}"
                trace.append({"call": call_no, "thought": "", "tool": "", "args": "", "observation": f"ERROR: {e}"})
                break

            if step.action == "final_answer":
                answer = step.answer.strip() or "I don't have this information in the provided sources."
                trace.append({"call": call_no, "thought": step.thought, "tool": "", "args": "", "observation": ""})
                break

            name, raw_args = step.tool.strip(), step.args_json
            if is_last:
                trace.append({"call": call_no, "thought": step.thought, "tool": name, "args": raw_args,
                              "observation": "LIMIT: no model calls left, tool was NOT executed."})
                status = "stopped_by_limit"
                break

            if name not in TOOLS:
                obs = f"ERROR: unknown tool '{name}'. Available: {', '.join(TOOLS)}"
            else:
                try:
                    args = json.loads(raw_args or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("args must be a JSON object")
                except Exception as e:
                    args, obs = None, f"ERROR: args_json is not a valid JSON object ({e})"
                else:
                    if tool_counts.get(name, 0) >= self.max_calls_per_tool:
                        obs = (f"LIMIT: tool '{name}' was already called {self.max_calls_per_tool} times "
                               f"in this request. Use other tools or answer with what you have.")
                        status = "tool_limit_hit"
                    else:
                        tool_counts[name] = tool_counts.get(name, 0) + 1
                        try:
                            obs = run_tool(name, args, self.store, self.memory)
                        except Exception as e:
                            obs = f"ERROR while running tool: {e}"

            trace.append({"call": call_no, "thought": step.thought, "tool": name, "args": raw_args,
                          "observation": obs[:1500]})
            scratch.append(f"Step {call_no}: tool={name} args={raw_args}\nResult:\n{obs}")
            if self.pause:
                time.sleep(self.pause)

        if answer is None:
            status = "stopped_by_limit"
            answer = (f"Stopped: the limit of {self.max_model_calls} model call(s) was reached "
                      f"before a final answer.")

        self._remember_turn(question, answer)
        result = {"question": question, "answer": answer, "status": status,
                  "model_calls": model_calls, "tool_calls": dict(tool_counts), "trace": trace}
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **result}, ensure_ascii=False) + "\n")
        return result
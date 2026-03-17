from __future__ import annotations

import html,json,os,re,sqlite3,sys
from dataclasses import dataclass,field
from datetime import datetime,timezone
from functools import partial
from pathlib import Path
from typing import Callable

from fastcore.xdg import xdg_config_home
from IPython import get_ipython
from IPython.core.inputtransformer2 import leading_empty_lines
from IPython.core.magic import Magics, line_cell_magic, magics_class
from lisette.core import Chat,StreamFormatter
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_THINK = "l"
DEFAULT_SEARCH = "l"
DEFAULT_CODE_THEME = "monokai"
DEFAULT_LOG_EXACT = False
DEFAULT_SYSTEM_PROMPT = """You are an AI assistant running inside IPython.

The user interacts with you through `ipyai`, an IPython extension that turns input starting with a backtick into an AI prompt.

You may receive:
- a `<context>` XML block containing recent IPython code and optional outputs
- a `<user-request>` XML block containing the user's actual request

You can respond in Markdown. Your final visible output in terminal IPython will be rendered with Rich, so normal Markdown formatting, fenced code blocks, lists, and tables are appropriate when useful.

When the user mentions `&`-backtick tool references such as `&`tool_name``, the corresponding callable from the active IPython namespace may be available to you as a tool. Use tools when they will materially improve correctness or completeness; otherwise answer directly.

Assume you are helping an interactive Python user. Prefer concise, accurate, practical responses. When writing code, default to Python unless the user asks for something else.
"""
AI_MAGIC_NAME = "ipyai"
AI_LAST_PROMPT = "_ai_last_prompt"
AI_LAST_RESPONSE = "_ai_last_response"
AI_EXTENSION_NS = "_ipyai"
AI_EXTENSION_ATTR = "_ipyai_extension"
AI_RESET_LINE_NS = "_ipyai_reset_line"
AI_STARTUP_APPLIED_NS = "_ipyai_startup_applied"
AI_PROMPTS_TABLE = "ai_prompts"
AI_PROMPTS_COLS = ["id", "session", "prompt", "response", "history_line"]

__all__ = "AI_EXTENSION_ATTR AI_EXTENSION_NS AI_LAST_PROMPT AI_LAST_RESPONSE AI_MAGIC_NAME AI_PROMPTS_TABLE AI_RESET_LINE_NS DEFAULT_MODEL " \
          "IPyAIExtension create_extension default_config_path default_log_path default_startup_path default_sysp_path is_backtick_prompt load_ipython_extension prompt_from_lines " \
          "stream_to_stdout transform_backticks unload_ipython_extension".split()

_prompt_template = """{context}<user-request>{prompt}</user-request>"""
_tool_re = re.compile(r"&`(\w+)`")
_tool_block_re = re.compile(r"<details class='tool-usage-details'>\s*<summary>(.*?)</summary>\s*```json\s*(.*?)\s*```\s*</details>", flags=re.DOTALL)


def is_backtick_prompt(lines: list[str]) -> bool: return bool(lines) and lines[0].startswith("`")


def _drop_continuation_backslashes(lines: list[str]) -> list[str]:
    res = []
    for i,line in enumerate(lines):
        if i < len(lines)-1 and line.endswith("\\\n"): line = line[:-2] + "\n"
        res.append(line)
    return res


def prompt_from_lines(lines: list[str]) -> str | None:
    if not is_backtick_prompt(lines): return None
    first,*rest = lines
    return "".join(_drop_continuation_backslashes([first[1:], *rest]))


def transform_backticks(lines: list[str], magic: str=AI_MAGIC_NAME) -> list[str]:
    prompt = prompt_from_lines(lines)
    if prompt is None: return lines
    return [f"get_ipython().run_cell_magic({magic!r}, '', {prompt!r})\n"]


def _xml_attr(o) -> str: return html.escape(str(o), quote=True)


def _xml_text(o) -> str: return html.escape(str(o), quote=False)


def _tag(name: str, content="", **attrs) -> str:
    ats = "".join(f' {k}="{_xml_attr(v)}"' for k,v in attrs.items())
    return f"<{name}{ats}>{content}</{name}>"


def _is_ipyai_input(source: str) -> bool:
    src = source.lstrip()
    return src.startswith("`") or src.startswith("%ipyai") or src.startswith("%%ipyai")


def _tool_names(text: str) -> set[str]: return set(_tool_re.findall(text or ""))


def _tool_refs(prompt: str, hist: list[dict]) -> set[str]:
    names = _tool_names(prompt)
    for o in hist: names |= _tool_names(o["prompt"])
    return names


def _single_line(s: str) -> str: return re.sub(r"\s+", " ", s.strip())


def _truncate_short(s: str, mx: int=100) -> str:
    s = _single_line(s)
    return s if len(s) <= mx else s[:mx-3] + "..."


def compact_tool_display(text: str, result_chars: int=100) -> str:
    def _repl(m):
        summary,payload = m.groups()
        try: result = json.loads(payload).get("result", "")
        except Exception: return m.group(0)
        return f"🔧 {_single_line(summary)} => {_truncate_short(str(result), mx=result_chars)}"
    return _tool_block_re.sub(_repl, text)


def _with_trailing_newline(text: str) -> str: return text if not text or text.endswith("\n") else text + "\n"


def _display_line_count(out, shown: str) -> int:
    if not shown: return 0
    console = Console(file=out, force_terminal=getattr(out, "isatty", lambda: False)(), soft_wrap=True)
    lines = console.render_lines(Text(shown), pad=False)
    return len(lines)


def _clear_terminal_block(out, shown: str):
    nlines = _display_line_count(out, shown)
    if nlines > 1: out.write(f"\x1b[{nlines-1}F")
    out.write("\x1b[J")
    out.flush()


def _render_markdown(out, text: str, code_theme: str, console_cls=Console, markdown_cls=Markdown):
    md = markdown_cls(text, code_theme=code_theme, inline_code_theme=code_theme, inline_code_lexer="python")
    console = console_cls(file=out, force_terminal=getattr(out, "isatty", lambda: False)(), soft_wrap=True)
    console.print(md)


def _rewrite_terminal_output(out, shown: str, rewritten: str, code_theme: str, console_cls=Console, markdown_cls=Markdown):
    if not getattr(out, "isatty", lambda: False)(): return
    _clear_terminal_block(out, shown)
    _render_markdown(out, rewritten, code_theme, console_cls=console_cls, markdown_cls=markdown_cls)


def stream_to_stdout(stream, formatter_cls: Callable[..., StreamFormatter]=StreamFormatter, out=None, code_theme: str=DEFAULT_CODE_THEME,
                     console_cls=Console, markdown_cls=Markdown) -> str:
    out = sys.stdout if out is None else out
    fmt = formatter_cls()
    res = []
    for chunk in fmt.format_stream(stream):
        if not chunk: continue
        out.write(chunk)
        out.flush()
        res.append(chunk)
    text = "".join(res)
    shown = _with_trailing_newline(text)
    if shown != text:
        out.write("\n")
        out.flush()
    rewritten = compact_tool_display(text)
    if getattr(out, "isatty", lambda: False)(): _rewrite_terminal_output(out, shown, rewritten, code_theme, console_cls=console_cls, markdown_cls=markdown_cls)
    return text


def default_config_path() -> Path: return xdg_config_home()/"ipyai"/"config.json"


def default_sysp_path() -> Path: return xdg_config_home()/"ipyai"/"sysp.txt"


def default_startup_path() -> Path: return xdg_config_home()/"ipyai"/"startup.json"


def default_log_path() -> Path: return xdg_config_home()/"ipyai"/"exact-log.jsonl"


def _validate_level(name: str, value: str, default: str) -> str:
    value = (value or default).strip().lower()
    if value not in {"l", "m", "h"}: raise ValueError(f"{name} must be one of h/m/l, got {value!r}")
    return value


def _validate_bool(name: str, value, default: bool) -> bool:
    if value is None: return default
    if isinstance(value, bool): return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "yes", "on"}: return True
        if value in {"0", "false", "no", "off"}: return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _default_config():
    return dict(model=os.environ.get("IPYAI_MODEL", DEFAULT_MODEL), think=DEFAULT_THINK, search=DEFAULT_SEARCH,
                code_theme=DEFAULT_CODE_THEME, log_exact=DEFAULT_LOG_EXACT)


def load_config(path=None) -> dict:
    path = Path(path or default_config_path())
    cfg = _default_config()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        data = json.loads(path.read_text())
        if not isinstance(data, dict): raise ValueError(f"Invalid config format in {path}")
        cfg.update({k:v for k,v in data.items() if k in cfg})
    else: path.write_text(json.dumps(cfg, indent=2) + "\n")
    cfg["model"] = str(cfg["model"]).strip() or DEFAULT_MODEL
    cfg["think"] = _validate_level("think", cfg["think"], DEFAULT_THINK)
    cfg["search"] = _validate_level("search", cfg["search"], DEFAULT_SEARCH)
    cfg["code_theme"] = str(cfg["code_theme"]).strip() or DEFAULT_CODE_THEME
    cfg["log_exact"] = _validate_bool("log_exact", cfg["log_exact"], DEFAULT_LOG_EXACT)
    return cfg


def load_sysp(path=None) -> str:
    path = Path(path or default_sysp_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists(): path.write_text(DEFAULT_SYSTEM_PROMPT)
    return path.read_text()


def _default_startup(): return dict(version=1, events=[])


def load_startup(path=None) -> dict:
    path = Path(path or default_startup_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        data = json.loads(path.read_text())
        if not isinstance(data, dict): raise ValueError(f"Invalid startup format in {path}")
        events = data.get("events", [])
        if not isinstance(events, list): raise ValueError(f"Invalid startup events in {path}")
        return dict(version=int(data.get("version", 1)), events=events)
    data = _default_startup()
    path.write_text(json.dumps(data, indent=2) + "\n")
    return data


@magics_class
class AIMagics(Magics):
    def __init__(self, shell, ext):
        super().__init__(shell)
        self.ext = ext

    @line_cell_magic("ipyai")
    def ipyai(self, line: str="", cell: str | None=None):
        if cell is None: return self.ext.handle_line(line)
        return self.ext.run_prompt(cell)


@dataclass
class IPyAIExtension:
    shell: object
    model: str|None=None
    think: str|None=None
    search: str|None=None
    code_theme: str|None=None
    log_exact: bool|None=None
    system_prompt: str|None=None
    chat_cls: Callable[..., Chat]=Chat
    formatter_cls: Callable[..., StreamFormatter]=StreamFormatter
    magic_name: str=AI_MAGIC_NAME
    config_path: Path|None=None
    sysp_path: Path|None=None
    startup_path: Path|None=None
    log_path: Path|None=None
    loaded: bool=False
    transformer: Callable = field(init=False)

    def __post_init__(self):
        self.config_path = Path(self.config_path or default_config_path())
        self.sysp_path = Path(self.sysp_path or default_sysp_path())
        self.startup_path = Path(self.startup_path or default_startup_path())
        self.log_path = Path(self.log_path or default_log_path())
        cfg = load_config(self.config_path)
        self.model = self.model or cfg["model"]
        self.think = _validate_level("think", self.think if self.think is not None else cfg["think"], DEFAULT_THINK)
        self.search = _validate_level("search", self.search if self.search is not None else cfg["search"], DEFAULT_SEARCH)
        self.code_theme = str(self.code_theme or cfg["code_theme"]).strip() or DEFAULT_CODE_THEME
        self.log_exact = _validate_bool("log_exact", self.log_exact if self.log_exact is not None else cfg["log_exact"], DEFAULT_LOG_EXACT)
        self.system_prompt = self.system_prompt if self.system_prompt is not None else load_sysp(self.sysp_path)
        load_startup(self.startup_path)
        self.transformer = partial(transform_backticks, magic=self.magic_name)

    @property
    def history_manager(self): return getattr(self.shell, "history_manager", None)

    @property
    def session_number(self): return getattr(self.history_manager, "session_number", 0)

    @property
    def reset_line(self): return self.shell.user_ns.get(AI_RESET_LINE_NS, 0)

    @property
    def startup_applied(self): return bool(self.shell.user_ns.get(AI_STARTUP_APPLIED_NS, False))

    @property
    def db(self):
        hm = self.history_manager
        return None if hm is None else hm.db

    def ensure_prompt_table(self):
        if self.db is None: return
        with self.db:
            self.db.execute(
                f"""CREATE TABLE IF NOT EXISTS {AI_PROMPTS_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session INTEGER NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                history_line INTEGER NOT NULL DEFAULT 0
            )"""
            )
            cols = [o[1] for o in self.db.execute(f"PRAGMA table_info({AI_PROMPTS_TABLE})")]
            if cols != AI_PROMPTS_COLS:
                self.db.execute(f"DROP TABLE {AI_PROMPTS_TABLE}")
                self.db.execute(
                    f"""CREATE TABLE {AI_PROMPTS_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session INTEGER NOT NULL,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    history_line INTEGER NOT NULL DEFAULT 0
                )"""
                )
            self.db.execute(f"CREATE INDEX IF NOT EXISTS idx_{AI_PROMPTS_TABLE}_session_id ON {AI_PROMPTS_TABLE} (session, id)")

    def prompt_records(self, session: int | None=None) -> list:
        if self.db is None: return []
        self.ensure_prompt_table()
        session = self.session_number if session is None else session
        cur = self.db.execute(f"SELECT id, prompt, response, history_line FROM {AI_PROMPTS_TABLE} WHERE session=? ORDER BY id", (session,))
        return cur.fetchall()

    def prompt_rows(self, session: int | None=None) -> list: return [(p, r) for _,p,r,_ in self.prompt_records(session=session)]

    def last_prompt_line(self, session: int | None=None) -> int:
        rows = self.prompt_records(session=session)
        return rows[-1][3] if rows else self.reset_line

    def current_prompt_line(self) -> int:
        c = getattr(self.shell, "execution_count", 1)
        return max(c-1, 0)

    def current_input_line(self) -> int: return max(getattr(self.shell, "execution_count", 1), 1)

    def code_history(self, start: int, stop: int) -> list:
        hm = self.history_manager
        if hm is None or stop <= start: return []
        return list(hm.get_range(session=0, start=start, stop=stop, raw=True, output=True))

    def full_history(self) -> list: return self.code_history(1, self.current_input_line()+1)

    def code_context(self, start: int, stop: int) -> str:
        entries = self.code_history(start, stop)
        parts = []
        for _,line,pair in entries:
            source,output = pair
            if not source or _is_ipyai_input(source): continue
            parts.append(_tag("code", _xml_text(source)))
            if output is not None: parts.append(_tag("output", _xml_text(output)))
        if not parts: return ""
        return _tag("context", "".join(parts)) + "\n"

    def format_prompt(self, prompt: str, start: int, stop: int) -> str:
        ctx = self.code_context(start, stop)
        return _prompt_template.format(context=ctx, prompt=prompt.strip())

    def dialog_history(self, current_prompt_line: int) -> list:
        hist,res = [],[]
        prev_line = self.reset_line
        for pid,prompt,response,history_line in self.prompt_records():
            hist += [self.format_prompt(prompt, prev_line+1, history_line), response]
            res.append(dict(id=pid, prompt=prompt, response=response, history_line=history_line))
            prev_line = history_line
        return hist,res

    def resolve_tools(self, prompt: str, hist: list[dict]) -> list:
        ns = self.shell.user_ns
        names = _tool_refs(prompt, hist)
        missing = [o for o in _tool_names(prompt) if o not in ns]
        if missing: raise NameError(f"Missing tool(s) in user_ns: {', '.join(sorted(missing))}")
        bad = [o for o in _tool_names(prompt) if o in ns and not callable(ns[o])]
        if bad: raise TypeError(f"Non-callable tool(s): {', '.join(sorted(bad))}")
        return [ns[o] for o in sorted(names) if o in ns and callable(ns[o])]

    def save_prompt(self, prompt: str, response: str, history_line: int):
        if self.db is None: return
        self.ensure_prompt_table()
        with self.db: self.db.execute(f"INSERT INTO {AI_PROMPTS_TABLE} (session, prompt, response, history_line) VALUES (?, ?, ?, ?)",
                                      (self.session_number, prompt, response, history_line))

    def startup_events(self) -> list[dict]:
        events = []
        for _,line,pair in self.full_history():
            source,_ = pair
            if not source or _is_ipyai_input(source): continue
            events.append(dict(kind="code", line=line, source=source))
        for pid,prompt,response,history_line in self.prompt_records():
            events.append(dict(kind="prompt", id=pid, line=history_line+1, history_line=history_line, prompt=prompt, response=response))
        return sorted(events, key=lambda o: (o["line"], 0 if o["kind"] == "code" else 1))

    def save_startup(self) -> tuple[int,int]:
        data = dict(version=1, events=[{k:v for k,v in o.items() if k != "id"} for o in self.startup_events()])
        self.startup_path.parent.mkdir(parents=True, exist_ok=True)
        self.startup_path.write_text(json.dumps(data, indent=2) + "\n")
        return sum(o["kind"] == "code" for o in data["events"]), sum(o["kind"] == "prompt" for o in data["events"])

    def _advance_execution_count(self):
        if hasattr(self.shell, "execution_count"): self.shell.execution_count += 1

    def apply_startup(self) -> tuple[int,int]:
        if self.startup_applied: return 0,0
        self.shell.user_ns[AI_STARTUP_APPLIED_NS] = True
        if self.current_prompt_line() > 0 or self.prompt_records(): return 0,0
        events = load_startup(self.startup_path)["events"]
        ncode = nprompt = 0
        for o in sorted(events, key=lambda x: (x.get("line", 0), 0 if x.get("kind") == "code" else 1)):
            if o.get("kind") == "code":
                source = o.get("source", "")
                if not source: continue
                res = self.shell.run_cell(source, store_history=True)
                ncode += 1
                if getattr(res, "success", True) is False: break
            elif o.get("kind") == "prompt":
                self.save_prompt(o.get("prompt", ""), o.get("response", ""), self.current_prompt_line())
                self._advance_execution_count()
                nprompt += 1
        return ncode,nprompt

    def log_exact_exchange(self, prompt: str, response: str):
        if not self.log_exact: return
        rec = dict(ts=datetime.now(timezone.utc).isoformat(), session=self.session_number, prompt=prompt, response=response)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def reset_session_history(self) -> int:
        if self.db is None: return 0
        self.ensure_prompt_table()
        with self.db: cur = self.db.execute(f"DELETE FROM {AI_PROMPTS_TABLE} WHERE session=?", (self.session_number,))
        self.shell.user_ns.pop(AI_LAST_PROMPT, None)
        self.shell.user_ns.pop(AI_LAST_RESPONSE, None)
        self.shell.user_ns[AI_RESET_LINE_NS] = self.current_prompt_line()
        return cur.rowcount or 0

    def load(self):
        if self.loaded: return self
        self.ensure_prompt_table()
        cts = self.shell.input_transformer_manager.cleanup_transforms
        if self.transformer not in cts:
            idx = 1 if cts and cts[0] is leading_empty_lines else 0
            cts.insert(idx, self.transformer)
        self.shell.register_magics(AIMagics(self.shell, self))
        self.shell.user_ns[AI_EXTENSION_NS] = self
        self.shell.user_ns.setdefault(AI_RESET_LINE_NS, 0)
        setattr(self.shell, AI_EXTENSION_ATTR, self)
        self.apply_startup()
        self.loaded = True
        return self

    def unload(self):
        if not self.loaded: return self
        cts = self.shell.input_transformer_manager.cleanup_transforms
        if self.transformer in cts: cts.remove(self.transformer)
        if self.shell.user_ns.get(AI_EXTENSION_NS) is self: self.shell.user_ns.pop(AI_EXTENSION_NS, None)
        if getattr(self.shell, AI_EXTENSION_ATTR, None) is self: delattr(self.shell, AI_EXTENSION_ATTR)
        self.loaded = False
        return self

    def handle_line(self, line: str):
        line = line.strip()
        if not line:
            print(f"Model: {self.model}")
            print(f"Think: {self.think}")
            print(f"Search: {self.search}")
            print(f"Code theme: {self.code_theme}")
            print(f"Log exact: {self.log_exact}")
            print(f"Config: {self.config_path}")
            print(f"System prompt: {self.sysp_path}")
            print(f"Startup: {self.startup_path}")
            print(f"Exact log: {self.log_path}")
            return None
        if line == "model":
            print(f"Model: {self.model}")
            return None
        if line == "think":
            print(f"Think: {self.think}")
            return None
        if line == "search":
            print(f"Search: {self.search}")
            return None
        if line == "code_theme":
            print(f"Code theme: {self.code_theme}")
            return None
        if line == "log_exact":
            print(f"Log exact: {self.log_exact}")
            return None
        if line == "reset":
            n = self.reset_session_history()
            print(f"Deleted {n} AI prompts from session {self.session_number}.")
            return None
        if line == "save":
            ncode,nprompt = self.save_startup()
            print(f"Saved {ncode} code cells and {nprompt} prompts to {self.startup_path}.")
            return None
        if line.startswith("model "):
            self.model = line.split(None, 1)[1].strip()
            print(f"Model: {self.model}")
            return None
        if line.startswith("think "):
            self.think = _validate_level("think", line.split(None, 1)[1], self.think)
            print(f"Think: {self.think}")
            return None
        if line.startswith("search "):
            self.search = _validate_level("search", line.split(None, 1)[1], self.search)
            print(f"Search: {self.search}")
            return None
        if line.startswith("code_theme "):
            self.code_theme = line.split(None, 1)[1].strip() or DEFAULT_CODE_THEME
            print(f"Code theme: {self.code_theme}")
            return None
        return self.run_prompt(line)

    def run_prompt(self, prompt: str):
        prompt = (prompt or "").rstrip("\n")
        if not prompt.strip(): return None
        history_line = self.current_prompt_line()
        hist,recs = self.dialog_history(history_line)
        tools = self.resolve_tools(prompt, recs)
        full_prompt = self.format_prompt(prompt, self.last_prompt_line()+1, history_line)
        self.shell.user_ns[AI_LAST_PROMPT] = prompt
        chat = self.chat_cls(model=self.model, sp=self.system_prompt, ns=self.shell.user_ns, hist=hist, tools=tools or None)
        text = stream_to_stdout(chat(full_prompt, stream=True, think=self.think, search=self.search),
                                formatter_cls=self.formatter_cls, code_theme=self.code_theme)
        self.shell.user_ns[AI_LAST_RESPONSE] = text
        self.log_exact_exchange(full_prompt, text)
        self.save_prompt(prompt, text, history_line)
        return None


def create_extension(shell=None, **kwargs):
    shell = shell or get_ipython()
    if shell is None: raise RuntimeError("No active IPython shell found")
    ext = getattr(shell, AI_EXTENSION_ATTR, None)
    if ext is None: ext = IPyAIExtension(shell=shell, **kwargs)
    if not ext.loaded: ext.load()
    return ext


def load_ipython_extension(ipython): return create_extension(ipython)


def unload_ipython_extension(ipython):
    ext = getattr(ipython, AI_EXTENSION_ATTR, None)
    if ext is None: return
    ext.unload()

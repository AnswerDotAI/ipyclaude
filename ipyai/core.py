import argparse,ast,atexit,html,inspect,json,os,re,sqlite3,sys,uuid
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
from typing import Callable

from fastcore.basics import patch,patch_to
from fastcore.xdg import xdg_config_home
from fastcore.xtras import frontmatter
from IPython import get_ipython
from IPython.core.inputtransformer2 import leading_empty_lines
from IPython.core.magic import Magics, line_cell_magic, magics_class
from IPython.core.ultratb import SyntaxTB
from lisette.core import AsyncChat,AsyncStreamFormatter,FullResponse,contents
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from toolslm.funccall import get_schema_nm
from IPython.core.interactiveshell import InteractiveShell

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_THINK = "l"
DEFAULT_SEARCH = "l"
DEFAULT_CODE_THEME = "monokai"
DEFAULT_LOG_EXACT = False
DEFAULT_COMPLETION_MODEL = "claude-haiku-4-5-20251001"
_COMPLETION_SP = "You are a code completion engine for IPython. Return ONLY the completion text that should be inserted at the cursor position. No explanation, no markdown, no code fences, no prefix repetition."
DEFAULT_SYSTEM_PROMPT = """You are an AI assistant running inside IPython.

The user interacts with you through `ipyai`, an IPython extension that turns input starting with a period into an AI prompt.

You may receive:
- a `<context>` XML block containing recent IPython code, outputs, and notes
- a `<user-request>` XML block containing the user's actual request

Inside `<context>`, entries tagged `<code>` are executed Python cells. Entries tagged `<note>` are user-written notes (cells whose only content is a string literal). Notes provide context and intent but are not executable code.

Earlier user turns in the chat history may also contain their own `<context>` blocks. When answering questions about what you have seen in the IPython session, consider the full chat history, not only the latest `<context>` block.

You can respond in Markdown. Your final visible output in terminal IPython will be rendered with Rich, so normal Markdown formatting, fenced code blocks, lists, and tables are appropriate when useful.

When the user mentions `&`-backtick tool references such as `&`tool_name``, the corresponding callable from the active IPython namespace may be available to you as a tool. Use tools when they will materially improve correctness or completeness; otherwise answer directly.

If a `<skills>` section is appended to this system prompt, it lists available skills. When a user's request matches a skill description, call the `load_skill` tool with the skill's path to load its full instructions before responding.

Assume you are helping an interactive Python user. Prefer concise, accurate, practical responses. When writing code, default to Python unless the user asks for something else.
"""
MAGIC_NAME = "ipyai"
LAST_PROMPT = "_ai_last_prompt"
LAST_RESPONSE = "_ai_last_response"
EXTENSION_NS = "_ipyai"
EXTENSION_ATTR = "_ipyai_extension"
RESET_LINE_NS = "_ipyai_reset_line"
STARTUP_APPLIED_NS = "_ipyai_startup_applied"
PROMPTS_TABLE = "ai_prompts"
PROMPTS_COLS = ["id", "session", "prompt", "response", "history_line"]
_PROMPTS_SQL = f"""CREATE TABLE IF NOT EXISTS {PROMPTS_TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session INTEGER NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    history_line INTEGER NOT NULL DEFAULT 0)"""

def _ensure_prompts_table(db):
    if db is None: return
    with db:
        db.execute(_PROMPTS_SQL)
        cols = [o[1] for o in db.execute(f"PRAGMA table_info({PROMPTS_TABLE})")]
        if cols != PROMPTS_COLS:
            db.execute(f"DROP TABLE {PROMPTS_TABLE}")
            db.execute(_PROMPTS_SQL)
        db.execute(f"CREATE INDEX IF NOT EXISTS idx_{PROMPTS_TABLE}_session_id ON {PROMPTS_TABLE} (session, id)")
CONFIG_DIR = xdg_config_home()/"ipyai"
CONFIG_PATH = CONFIG_DIR/"config.json"
SYSP_PATH = CONFIG_DIR/"sysp.txt"
STARTUP_PATH = CONFIG_DIR/"startup.ipynb"
LOG_PATH = CONFIG_DIR/"exact-log.jsonl"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

__all__ = """EXTENSION_ATTR EXTENSION_NS LAST_PROMPT LAST_RESPONSE MAGIC_NAME PROMPTS_TABLE RESET_LINE_NS DEFAULT_MODEL DEFAULT_COMPLETION_MODEL IPyAIExtension
create_extension CONFIG_PATH SYSP_PATH STARTUP_PATH LOG_PATH is_dot_prompt load_ipython_extension
prompt_from_lines astream_to_stdout transform_dots unload_ipython_extension""".split()

_prompt_template = """{context}<user-request>{prompt}</user-request>"""
_tool_re = re.compile(r"&`(\w+)`")
_tool_block_re = re.compile(
    r"<details class='tool-usage-details'>\s*<summary>(.*?)</summary>\s*```json\s*(.*?)\s*```\s*</details>", flags=re.DOTALL)
_status_attrs = "model completion_model think search code_theme log_exact".split()


def _extract_code_blocks(text):
    from mistletoe import Document
    from mistletoe.block_token import CodeFence
    return [child.children[0].content.strip() for child in Document(text).children
            if isinstance(child, CodeFence) and child.language in ('python', 'py') and child.children and child.children[0].content.strip()]


def is_dot_prompt(lines: list[str]) -> bool: return bool(lines) and lines[0].startswith(".")


def prompt_from_lines(lines: list[str]) -> str | None:
    if not is_dot_prompt(lines): return None
    first,*rest = lines
    return "".join([first[1:], *rest]).replace("\\\n", "\n")


def transform_dots(lines: list[str], magic: str=MAGIC_NAME) -> list[str]:
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
    return src.startswith(".") or src.startswith("%ipyai") or src.startswith("%%ipyai")


def _is_note(source):
    try: tree = ast.parse(source)
    except SyntaxError: return False
    return (len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str))


def _note_str(source): return ast.parse(source).body[0].value.value


def _tool_names(text: str) -> set[str]: return set(_tool_re.findall(text or ""))


def _allowed_tools(text):
    "Extract tool names from frontmatter allowed-tools and &`tool` mentions."
    fm, body = frontmatter(text)
    names = _tool_names(text)
    if fm:
        at = fm.get('allowed-tools', '')
        if at: names |= set(str(at).split())
    return names


def _tool_results(response):
    "Extract tool names from qualifying tool results in a stored AI response."
    names = set()
    for m in _tool_block_re.finditer(response or ""):
        try: result = str(json.loads(m.group(2)).get("result", ""))
        except Exception: continue
        fm, _ = frontmatter(result)
        if fm and (fm.get('allowed-tools') or fm.get('eval')): names |= _allowed_tools(result)
    return names


def _tool_refs(prompt, hist, skills=None, notes=None, responses=None):
    names = _tool_names(prompt)
    for o in hist: names |= _tool_names(o["prompt"])
    if skills:
        names.add("load_skill")
        for s in skills: names |= set(s.get("tools") or [])
    for n in (notes or []): names |= _allowed_tools(n)
    for r in (responses or []): names |= _tool_results(r)
    return names


def _event_sort_key(o): return o.get("line", 0), 0 if o.get("kind") == "code" else 1


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


def _strip_thinking(text):
    cleaned = re.sub(r'🧠+\n*', '', text).lstrip('\n')
    return cleaned if cleaned else text


def _display_text(text): return _strip_thinking(compact_tool_display(text))


def _markdown_renderable(text: str, code_theme: str, markdown_cls=Markdown):
    return markdown_cls(text, code_theme=code_theme, inline_code_theme=code_theme, inline_code_lexer="python")


async def _astream_to_live_markdown(chunks, out, code_theme: str, console_cls=Console, markdown_cls=Markdown, live_cls=Live) -> str:
    first = None
    async for chunk in chunks:
        if chunk:
            first = chunk
            break
    if first is None: return ""
    console = console_cls(file=out, force_terminal=True)
    text = first
    with live_cls(_markdown_renderable(_display_text(text), code_theme, markdown_cls), console=console,
        auto_refresh=False, transient=False, redirect_stdout=False, redirect_stderr=False) as live:
        async for chunk in chunks:
            if not chunk: continue
            text += chunk
            live.update(_markdown_renderable(_display_text(text), code_theme, markdown_cls), refresh=True)
    return text


async def astream_to_stdout(stream, formatter_cls: Callable[..., AsyncStreamFormatter]=AsyncStreamFormatter, out=None,
                            code_theme: str=DEFAULT_CODE_THEME, console_cls=Console, markdown_cls=Markdown, live_cls=Live) -> str:
    out = sys.stdout if out is None else out
    fmt = formatter_cls()
    chunks = fmt.format_stream(stream)
    if getattr(out, "isatty", lambda: False)(): return await _astream_to_live_markdown(chunks, out, code_theme, console_cls=console_cls,
        markdown_cls=markdown_cls, live_cls=live_cls)
    res = []
    async for chunk in chunks:
        if not chunk: continue
        out.write(chunk)
        out.flush()
        res.append(chunk)
    text = "".join(res)
    if text and not text.endswith("\n"):
        out.write("\n")
        out.flush()
    return text


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


@contextmanager
def _suppress_output_history(shell):
    pub = getattr(shell, "display_pub", None)
    if pub is None or not hasattr(pub, "_is_publishing"):
        yield
        return
    old = pub._is_publishing
    pub._is_publishing = True
    try: yield
    finally: pub._is_publishing = old


def _default_config():
    return dict(model=os.environ.get("IPYAI_MODEL", DEFAULT_MODEL), completion_model=DEFAULT_COMPLETION_MODEL,
                think=DEFAULT_THINK, search=DEFAULT_SEARCH, code_theme=DEFAULT_CODE_THEME, log_exact=DEFAULT_LOG_EXACT)


def load_config(path=None) -> dict:
    path = Path(path or CONFIG_PATH)
    cfg = _default_config()
    if path.exists():
        data = json.loads(path.read_text())
        if not isinstance(data, dict): raise ValueError(f"Invalid config format in {path}")
        cfg.update({k:v for k,v in data.items() if k in cfg})
    else: path.write_text(json.dumps(cfg, indent=2) + "\n")
    cfg["model"] = str(cfg["model"]).strip() or DEFAULT_MODEL
    cfg["completion_model"] = str(cfg["completion_model"]).strip() or DEFAULT_COMPLETION_MODEL
    cfg["think"] = _validate_level("think", cfg["think"], DEFAULT_THINK)
    cfg["search"] = _validate_level("search", cfg["search"], DEFAULT_SEARCH)
    cfg["code_theme"] = str(cfg["code_theme"]).strip() or DEFAULT_CODE_THEME
    cfg["log_exact"] = _validate_bool("log_exact", cfg["log_exact"], DEFAULT_LOG_EXACT)
    return cfg


def load_sysp(path=None) -> str:
    path = Path(path or SYSP_PATH)
    if not path.exists(): path.write_text(DEFAULT_SYSTEM_PROMPT)
    return path.read_text()


def _cell_id(): return uuid.uuid4().hex[:8]


def _event_to_cell(o):
    if o.get("kind") == "code":
        source = o.get("source", "")
        if _is_note(source):
            return dict(id=_cell_id(), cell_type="markdown", source=_note_str(source),
                        metadata=dict(ipyai=dict(kind="code", line=o.get("line", 0), source=source)))
        return dict(id=_cell_id(), cell_type="code", source=source, metadata=dict(ipyai=dict(kind="code", line=o.get("line", 0))),
                    outputs=[], execution_count=None)
    if o.get("kind") == "prompt":
        meta = dict(kind="prompt", line=o.get("line", 0), history_line=o.get("history_line", 0), prompt=o.get("prompt", ""))
        return dict(id=_cell_id(), cell_type="markdown", source=o.get("response", ""), metadata=dict(ipyai=meta))


def _cell_to_event(cell):
    meta = cell.get("metadata", {}).get("ipyai", {})
    kind = meta.get("kind")
    if kind == "code":
        source = meta.get("source") or cell.get("source", "")
        return dict(kind="code", line=meta.get("line", 0), source=source)
    if kind == "prompt":
        return dict(kind="prompt", line=meta.get("line", 0), history_line=meta.get("history_line", 0),
                    prompt=meta.get("prompt", ""), response=cell.get("source", ""))


def _default_startup(): return dict(cells=[], metadata=dict(ipyai_version=1), nbformat=4, nbformat_minor=5)


def load_startup(path=None) -> dict:
    path = Path(path or STARTUP_PATH)
    if path.exists():
        data = json.loads(path.read_text())
        if not isinstance(data, dict): raise ValueError(f"Invalid startup format in {path}")
        if "nbformat" in data:
            events = [e for c in data.get("cells", []) if (e := _cell_to_event(c)) is not None]
            return dict(version=int(data.get("metadata", {}).get("ipyai_version", 1)), events=events)
        events = data.get("events", [])
        if not isinstance(events, list): raise ValueError(f"Invalid startup events in {path}")
        return dict(version=int(data.get("version", 1)), events=events)
    nb = _default_startup()
    path.write_text(json.dumps(nb, indent=2) + "\n")
    return dict(version=1, events=[])


def _parse_skill(path):
    skill_md = Path(path) / "SKILL.md"
    if not skill_md.exists(): return None
    text = skill_md.read_text()
    fm, body = frontmatter(text)
    if not fm: return None
    name = fm.get('name', '')
    if not name: return None
    tools = list(_allowed_tools(text))
    return dict(name=name, path=str(path), description=fm.get('description', ''), tools=tools)


def _discover_skills(cwd=None):
    skills,seen = [],set()
    def _scan(skills_dir):
        if not skills_dir.is_dir(): return
        for p in sorted(skills_dir.iterdir()):
            rp = str(p.resolve())
            if not p.is_dir() or rp in seen: continue
            skill = _parse_skill(p)
            if skill:
                seen.add(rp)
                skills.append(skill)
    d = Path(cwd) if cwd else Path.cwd()
    while True:
        _scan(d / '.agents' / 'skills')
        if d.parent == d: break
        d = d.parent
    _scan(Path.home() / '.config' / 'agents' / 'skills')
    return skills


def _skills_xml(skills):
    if not skills: return ""
    parts = ["The following skills are available. To activate a skill and read its full instructions, call the load_skill tool with its path."]
    for s in skills:
        parts.append(f'<skill name="{_xml_attr(s["name"])}" path="{_xml_attr(s["path"])}">{_xml_text(s["description"])}</skill>')
    return "\n" + _tag("skills", "\n".join(parts))


_eval_re = re.compile(r'^#\|\s*eval:\s*true\s*$', re.MULTILINE)

def _eval_code_blocks(text, shell):
    "Run python code blocks starting with `#| eval: true` via `shell.run_cell`."
    for block in _extract_code_blocks(text):
        if _eval_re.match(block.split('\n', 1)[0]): shell.run_cell(block, store_history=False)

def load_skill(path:str):  # path: Path to the skill directory
    "Load a skill's full instructions from its SKILL.md file."
    p = Path(path) / "SKILL.md"
    if not p.exists(): return FullResponse(f"Error: SKILL.md not found at {p}")
    text = p.read_text()
    shell = get_ipython()
    if shell: _eval_code_blocks(text, shell)
    return FullResponse(text)


@patch_to(inspect, nm="getfile")
def _getfile(obj): return str(inspect._orig_getfile(obj))

@patch()
def structured_traceback(self:SyntaxTB, etype, evalue, etb, tb_offset=None, context=5):
    if hasattr(evalue, "msg") and not isinstance(evalue.msg, str): evalue.msg = str(evalue.msg)
    return self._orig_structured_traceback(etype, evalue, etb, tb_offset=tb_offset, context=context)

def _git_repo_root(path):
    "Walk up from `path` looking for `.git`, return repo root or None."
    p = Path(path).resolve()
    for d in [p] + list(p.parents):
        if (d / ".git").exists(): return str(d)
    return None

_LIST_SQL = """SELECT s.session, s.start, s.end, s.num_cmds, s.remark,
    (SELECT prompt FROM ai_prompts WHERE session=s.session ORDER BY id DESC LIMIT 1)
    FROM sessions s WHERE s.remark{w} ORDER BY s.session DESC LIMIT 20"""

def _list_sessions(db, cwd):
    "Return recent sessions for `cwd`, falling back to git repo root exact match."
    rows = db.execute(_LIST_SQL.format(w="=?"), (cwd,)).fetchall()
    if not rows:
        repo = _git_repo_root(cwd)
        if repo and repo != cwd: rows = db.execute(_LIST_SQL.format(w="=?"), (repo,)).fetchall()
    return rows

def _fmt_session(sid, start, ncmds, last_prompt, max_prompt=60):
    "Format a session row as a display string."
    p = (last_prompt or '').replace('\n', ' ')[:max_prompt]
    if last_prompt and len(last_prompt) > max_prompt: p += '...'
    return f"{sid:>6}  {str(start or '')[:19]:20}  {ncmds or 0:>5}  {p}"

def _pick_session(rows):
    "Show an interactive session picker, return chosen session ID or None."
    from prompt_toolkit.shortcuts import radiolist_dialog
    values = [(sid, _fmt_session(sid, start, ncmds, lp)) for sid,start,end,ncmds,remark,lp in rows]
    return radiolist_dialog(title="Resume session", text="Select a session to resume:", values=values, default=values[0][0]).run()

def resume_session(shell, session_id):
    "Replace the current fresh session with an existing one."
    hm = shell.history_manager
    fresh_id = hm.session_number
    row = hm.db.execute("SELECT session FROM sessions WHERE session=?", (session_id,)).fetchone()
    if not row: raise ValueError(f"Session {session_id} not found")
    with hm.db:
        hm.db.execute("DELETE FROM sessions WHERE session=?", (fresh_id,))
        hm.db.execute("UPDATE sessions SET end=NULL WHERE session=?", (session_id,))
    hm.session_number = session_id
    max_line = hm.db.execute("SELECT MAX(line) FROM history WHERE session=?", (session_id,)).fetchone()[0]
    shell.execution_count = (max_line or 0) + 1
    hm.input_hist_parsed.extend([""] * (shell.execution_count - 1))
    hm.input_hist_raw.extend([""] * (shell.execution_count - 1))

@magics_class
class AIMagics(Magics):
    def __init__(self, shell, ext):
        super().__init__(shell)
        self.ext = ext

    @line_cell_magic("ipyai")
    def ipyai(self, line: str="", cell: str | None=None):
        if cell is None: return self.ext.handle_line(line)
        return self.ext._run_prompt(cell)


class IPyAIExtension:
    def __init__(self, shell, model=None, completion_model=None, think=None, search=None, code_theme=None, log_exact=None, system_prompt=None):
        self.shell,self.loaded = shell,False
        cfg = load_config(CONFIG_PATH)
        self.model = model or cfg["model"]
        self.completion_model = completion_model or cfg["completion_model"]
        self.think = _validate_level("think", think if think is not None else cfg["think"], DEFAULT_THINK)
        self.search = _validate_level("search", search if search is not None else cfg["search"], DEFAULT_SEARCH)
        self.code_theme = str(code_theme or cfg["code_theme"]).strip() or DEFAULT_CODE_THEME
        self.log_exact = _validate_bool("log_exact", log_exact if log_exact is not None else cfg["log_exact"], DEFAULT_LOG_EXACT)
        self.system_prompt = system_prompt if system_prompt is not None else load_sysp(SYSP_PATH)
        self.skills = _discover_skills()
        if self.skills: shell.user_ns["load_skill"] = load_skill
        load_startup(STARTUP_PATH)

    @property
    def history_manager(self): return getattr(self.shell, "history_manager", None)

    @property
    def session_number(self): return getattr(self.history_manager, "session_number", 0)

    @property
    def reset_line(self): return self.shell.user_ns.get(RESET_LINE_NS, 0)

    @property
    def startup_applied(self): return bool(self.shell.user_ns.get(STARTUP_APPLIED_NS, False))

    @property
    def db(self):
        hm = self.history_manager
        return None if hm is None else hm.db

    def ensure_prompt_table(self): _ensure_prompts_table(self.db)

    def prompt_records(self, session: int | None=None) -> list:
        if self.db is None: return []
        self.ensure_prompt_table()
        session = self.session_number if session is None else session
        cur = self.db.execute(f"SELECT id, prompt, response, history_line FROM {PROMPTS_TABLE} WHERE session=? ORDER BY id", (session,))
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
            if _is_note(source): parts.append(_tag("note", _xml_text(_note_str(source))))
            else:
                parts.append(_tag("code", _xml_text(source)))
                if output is not None: parts.append(_tag("output", _xml_text(output)))
        if not parts: return ""
        return _tag("context", "".join(parts)) + "\n"

    def format_prompt(self, prompt: str, start: int, stop: int) -> str:
        ctx = self.code_context(start, stop)
        return _prompt_template.format(context=ctx, prompt=prompt.strip())

    def dialog_history(self) -> list:
        hist,res = [],[]
        prev_line = self.reset_line
        for pid,prompt,response,history_line in self.prompt_records():
            hist += [self.format_prompt(prompt, prev_line+1, history_line), response]
            res.append(dict(id=pid, prompt=prompt, response=response, history_line=history_line))
            prev_line = history_line
        return hist,res

    def note_strings(self, start, stop):
        "Return note string values from code history in range."
        return [_note_str(src) for _,_,pair in self.code_history(start, stop) if (src := pair[0]) and _is_note(src)]

    def resolve_tools(self, prompt, hist, skills=None, notes=None, responses=None):
        ns = self.shell.user_ns
        prompt_names = _tool_names(prompt)
        missing = [o for o in prompt_names if o not in ns]
        if missing: raise NameError(f"Missing tool(s) in user_ns: {', '.join(sorted(missing))}")
        bad = [o for o in prompt_names if not callable(ns[o])]
        if bad: raise TypeError(f"Non-callable tool(s): {', '.join(sorted(bad))}")
        all_refs = _tool_refs(prompt, hist, skills=skills, notes=notes, responses=responses)
        return [dict(type="function", function=get_schema_nm(o, ns, pname="parameters")) for o in sorted(all_refs) if callable(ns.get(o))]

    def save_prompt(self, prompt: str, response: str, history_line: int):
        if self.db is None: return
        self.ensure_prompt_table()
        with self.db:
            self.db.execute(f"INSERT INTO {PROMPTS_TABLE} (session, prompt, response, history_line) VALUES (?, ?, ?, ?)",
                (self.session_number, prompt, response, history_line))

    def startup_events(self) -> list[dict]:
        events = []
        for _,line,pair in self.full_history():
            source,_ = pair
            if not source or _is_ipyai_input(source): continue
            events.append(dict(kind="code", line=line, source=source))
        for pid,prompt,response,history_line in self.prompt_records():
            events.append(dict(kind="prompt", id=pid, line=history_line+1, history_line=history_line, prompt=prompt, response=response))
        return sorted(events, key=_event_sort_key)

    def save_startup(self) -> tuple[int,int]:
        events = [{k:v for k,v in o.items() if k != "id"} for o in self.startup_events()]
        nb = dict(cells=[_event_to_cell(e) for e in events], metadata=dict(ipyai_version=1), nbformat=4, nbformat_minor=5)
        STARTUP_PATH.write_text(json.dumps(nb, indent=2) + "\n")
        return sum(o["kind"] == "code" for o in events), sum(o["kind"] == "prompt" for o in events)

    def _advance_execution_count(self):
        if hasattr(self.shell, "execution_count"): self.shell.execution_count += 1

    def apply_startup(self) -> tuple[int,int]:
        if self.startup_applied: return 0,0
        self.shell.user_ns[STARTUP_APPLIED_NS] = True
        if self.current_prompt_line() > 0 or self.prompt_records(): return 0,0
        events = load_startup(STARTUP_PATH)["events"]
        ncode = nprompt = 0
        for o in sorted(events, key=_event_sort_key):
            if o.get("kind") == "code":
                source = o.get("source", "")
                if not source: continue
                res = self.shell.run_cell(source, store_history=True)
                ncode += 1
                if getattr(res, "success", True) is False: break
            elif o.get("kind") == "prompt":
                history_line = int(o.get("history_line", max(o.get("line", 1)-1, 0)))
                self.save_prompt(o.get("prompt", ""), o.get("response", ""), history_line)
                self._advance_execution_count()
                nprompt += 1
        return ncode,nprompt

    def log_exact_exchange(self, prompt: str, response: str):
        if not self.log_exact: return
        rec = dict(ts=datetime.now(timezone.utc).isoformat(), session=self.session_number, prompt=prompt, response=response)
        with LOG_PATH.open("a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def reset_session_history(self) -> int:
        if self.db is None: return 0
        self.ensure_prompt_table()
        with self.db: cur = self.db.execute(f"DELETE FROM {PROMPTS_TABLE} WHERE session=?", (self.session_number,))
        self.shell.user_ns.pop(LAST_PROMPT, None)
        self.shell.user_ns.pop(LAST_RESPONSE, None)
        self.shell.user_ns[RESET_LINE_NS] = self.current_prompt_line()
        return cur.rowcount or 0

    def _register_keybindings(self):
        pt_app = getattr(self.shell, 'pt_app', None)
        if pt_app is None: return
        # Wrap existing auto-suggest so AI completions survive partial accepts (M-f)
        # Patch existing auto-suggest so AI completions survive partial accepts (M-f)
        auto_suggest = pt_app.auto_suggest
        if auto_suggest:
            auto_suggest._ai_full_text = None
            _orig_get = auto_suggest.get_suggestion
            def _patched_get(buffer, document):
                from prompt_toolkit.auto_suggest import Suggestion
                text,ft = document.text,auto_suggest._ai_full_text
                if ft and ft.startswith(text) and len(ft) > len(text): return Suggestion(ft[len(text):])
                auto_suggest._ai_full_text = None
                return _orig_get(buffer, document)
            auto_suggest.get_suggestion = _patched_get
        ns = self.shell.user_ns
        def _get_blocks(): return _extract_code_blocks(ns.get(LAST_RESPONSE, ''))
        @pt_app.key_bindings.add('escape', 'W')
        def _paste_all(event):
            blocks = _get_blocks()
            if blocks: event.current_buffer.insert_text('\n'.join(blocks))
        for i,ch in enumerate('!@#$%^&*(', 1):
            @pt_app.key_bindings.add('escape', ch)
            def _paste_nth(event, n=i):
                blocks = _get_blocks()
                if len(blocks) >= n: event.current_buffer.insert_text(blocks[n-1])
        cycle = dict(idx=-1, resp='')
        def _cycle(event, delta):
            resp = ns.get(LAST_RESPONSE, '')
            blocks = _get_blocks()
            if not blocks: return
            if resp != cycle['resp']: cycle.update(idx=-1, resp=resp)
            cycle['idx'] = (cycle['idx'] + delta) % len(blocks)
            from prompt_toolkit.document import Document
            event.current_buffer.document = Document(blocks[cycle['idx']])
        # prompt_toolkit swaps A/B for modifier-4 (Alt+Shift) arrows
        @pt_app.key_bindings.add('escape', 's-up')   # physical Alt-Shift-Down
        def _cycle_down(event): _cycle(event, 1)
        @pt_app.key_bindings.add('escape', 's-down')  # physical Alt-Shift-Up
        def _cycle_up(event): _cycle(event, -1)
        # Alt-Up/Down: jump through complete history entries (skips line-by-line)
        @pt_app.key_bindings.add('escape', 'up')
        def _hist_back(event): event.current_buffer.history_backward()
        @pt_app.key_bindings.add('escape', 'down')
        def _hist_fwd(event): event.current_buffer.history_forward()
        # Alt-.: AI completion via haiku
        @pt_app.key_bindings.add('escape', '.')
        def _ai_suggest(event):
            buf = event.current_buffer
            doc = buf.document
            if not doc.text.strip(): return
            app = event.app
            async def _do_complete():
                try:
                    text = await self._ai_complete(doc)
                    if text and buf.document == doc:
                        from prompt_toolkit.auto_suggest import Suggestion
                        if auto_suggest: auto_suggest._ai_full_text = doc.text + text
                        buf.suggestion = Suggestion(text)
                        app.invalidate()
                except Exception: pass
            app.create_background_task(_do_complete())

    async def _ai_complete(self, document):
        prefix,suffix = document.text_before_cursor,document.text_after_cursor
        ctx = self.code_context(self.last_prompt_line()+1, self.current_prompt_line())
        parts = []
        if ctx: parts.append(ctx)
        parts.append(f"<current-input>\n<prefix>{_xml_text(prefix)}</prefix>")
        if suffix.strip(): parts.append(f"<suffix>{_xml_text(suffix)}</suffix>")
        parts.append("</current-input>")
        parts.append("\nReturn ONLY the completion text to insert immediately after the prefix."
                     " Do not repeat the prefix or include any explanation.")
        chat = AsyncChat(model=self.completion_model, sp=_COMPLETION_SP)
        res = await chat("\n".join(parts))
        return (contents(res).content or "").strip()

    @staticmethod
    def _patch_lexer():
        from IPython.terminal.ptutils import IPythonPTLexer
        from prompt_toolkit.lexers import SimpleLexer
        _plain = SimpleLexer()
        _orig = IPythonPTLexer.lex_document
        def _lex_document(self, document):
            text = document.text.lstrip()
            if text.startswith('.') or text.startswith('%%ipyai'): return _plain.lex_document(document)
            return _orig(self, document)
        IPythonPTLexer.lex_document = _lex_document

    def load(self):
        if self.loaded: return self
        self.ensure_prompt_table()
        cts = self.shell.input_transformer_manager.cleanup_transforms
        if transform_dots not in cts:
            idx = 1 if cts and cts[0] is leading_empty_lines else 0
            cts.insert(idx, transform_dots)
        self.shell.register_magics(AIMagics(self.shell, self))
        self.shell.user_ns[EXTENSION_NS] = self
        self.shell.user_ns.setdefault(RESET_LINE_NS, 0)
        setattr(self.shell, EXTENSION_ATTR, self)
        self._register_keybindings()
        self._patch_lexer()
        self.apply_startup()
        self.loaded = True
        return self

    def unload(self):
        if not self.loaded: return self
        cts = self.shell.input_transformer_manager.cleanup_transforms
        if transform_dots in cts: cts.remove(transform_dots)
        if self.shell.user_ns.get(EXTENSION_NS) is self: self.shell.user_ns.pop(EXTENSION_NS, None)
        if getattr(self.shell, EXTENSION_ATTR, None) is self: delattr(self.shell, EXTENSION_ATTR)
        self.loaded = False
        return self

    def _show(self, attr): return print(f"self.{attr}={getattr(self, attr)!r}")

    def _set(self, attr, value):
        setattr(self, attr, value)
        return self._show(attr)

    def handle_line(self, line: str):
        line = line.strip()
        if not line:
            for o in _status_attrs: self._show(o)
            print(f"{CONFIG_PATH=}")
            print(f"{SYSP_PATH=}")
            print(f"{STARTUP_PATH=}")
            return print(f"{LOG_PATH=}")
        if line in _status_attrs: return self._show(line)
        if line == "reset":
            n = self.reset_session_history()
            return print(f"Deleted {n} AI prompts from session {self.session_number}.")
        if line == "save":
            ncode,nprompt = self.save_startup()
            return print(f"Saved {ncode} code cells and {nprompt} prompts to {STARTUP_PATH}.")
        if line == "sessions":
            rows = _list_sessions(self.db, os.getcwd())
            if not rows: return print("No sessions found for this directory.")
            print(f"{'ID':>6}  {'Start':20}  {'Cmds':>5}  {'Last prompt'}")
            for sid,start,end,ncmds,remark,lp in rows: print(_fmt_session(sid, start, ncmds, lp))
            return
        cmd,_,arg = line.partition(" ")
        if arg:
            clean = arg.strip()
            vals = dict(model=lambda: clean, completion_model=lambda: clean or DEFAULT_COMPLETION_MODEL, code_theme=lambda: clean or DEFAULT_CODE_THEME,
                        think=lambda: _validate_level("think", clean, self.think), search=lambda: _validate_level("search", clean, self.search),
                        log_exact=lambda: _validate_bool("log_exact", clean, self.log_exact))
            if cmd in vals: return self._set(cmd, vals[cmd]())
        return self.run_prompt(line)

    async def _run_prompt(self, prompt: str):
        prompt = (prompt or "").rstrip("\n")
        if not prompt.strip(): return None
        history_line = self.current_prompt_line()
        hist,recs = self.dialog_history()
        # Collect notes and responses for tool resolution
        notes = []
        prev_line = self.reset_line
        for o in recs:
            notes += self.note_strings(prev_line+1, o["history_line"])
            prev_line = o["history_line"]
        notes += self.note_strings(self.last_prompt_line()+1, history_line)
        responses = [o["response"] for o in recs]
        tools = self.resolve_tools(prompt, recs, skills=self.skills, notes=notes, responses=responses)
        full_prompt = self.format_prompt(prompt, self.last_prompt_line()+1, history_line)
        self.shell.user_ns[LAST_PROMPT] = prompt
        sp = self.system_prompt
        if self.skills: sp += _skills_xml(self.skills)
        chat = AsyncChat(model=self.model, sp=sp, ns=self.shell.user_ns, hist=hist, tools=tools or None)
        stream = await chat(full_prompt, stream=True, think=self.think, search=self.search)
        with _suppress_output_history(self.shell): text = await astream_to_stdout(stream, code_theme=self.code_theme)
        self.shell.user_ns[LAST_RESPONSE] = text
        ng = getattr(self.shell, '_ipythonng_extension', None)
        if ng: ng._pty_output = _strip_thinking(text)
        self.log_exact_exchange(full_prompt, text)
        self.save_prompt(prompt, text, history_line)
        return None

    def run_prompt(self, prompt: str): return self.shell.loop_runner(self._run_prompt(prompt))


@patch()
async def run_cell_magic(self:InteractiveShell, magic_name, line, cell):
    result = self._orig_run_cell_magic(magic_name, line, cell)
    return await result if inspect.iscoroutine(result) else result

def _await_cell_magic(lines):
    if lines and 'get_ipython().run_cell_magic(' in lines[0]: lines = ['await ' + lines[0]] + lines[1:]
    return lines

def create_extension(shell=None, resume=None, **kwargs):
    shell = shell or get_ipython()
    if shell is None: raise RuntimeError("No active IPython shell found")
    _ensure_prompts_table(shell.history_manager.db)
    if resume is not None:
        if resume == -1:
            rows = _list_sessions(shell.history_manager.db, os.getcwd())
            if rows and (chosen := _pick_session(rows)): resume_session(shell, chosen)
            else: print("No sessions found for this directory.")
        else: resume_session(shell, resume)
    ext = getattr(shell, EXTENSION_ATTR, None)
    if ext is None: ext = IPyAIExtension(shell=shell, **kwargs)
    if not ext.loaded: ext.load()
    lts = shell.input_transformer_manager.line_transforms
    if not any(getattr(f, '__name__', None) == '_await_cell_magic' for f in lts): lts.append(_await_cell_magic)
    hm = shell.history_manager
    with hm.db: hm.db.execute("UPDATE sessions SET remark=? WHERE session=?", (os.getcwd(), hm.session_number))
    if not getattr(shell, '_ipyai_atexit', False):
        sid = hm.session_number
        atexit.register(lambda: print(f"\nTo resume: ipyai -r {sid}"))
        shell._ipyai_atexit = True
    return ext


_ng_parser = argparse.ArgumentParser(add_help=False)
_ng_parser.add_argument('-r', type=int, nargs='?', const=-1, default=None)

def _parse_ng_flags():
    "Parse IPYTHONNG_FLAGS env var via argparse."
    raw = os.environ.pop("IPYTHONNG_FLAGS", "")
    if not raw: return _ng_parser.parse_args([])
    return _ng_parser.parse_args(raw.split())

def load_ipython_extension(ipython):
    flags = _parse_ng_flags()
    return create_extension(ipython, resume=flags.r)


def unload_ipython_extension(ipython):
    ext = getattr(ipython, EXTENSION_ATTR, None)
    if ext is None: return
    ext.unload()

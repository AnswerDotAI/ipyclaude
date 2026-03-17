import asyncio,inspect,io,json,sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from IPython.core.inputtransformer2 import TransformerManager
from IPython.core.ultratb import SyntaxTB

import ipyai.core as core
from ipyai.core import (EXTENSION_NS, LAST_PROMPT, LAST_RESPONSE, RESET_LINE_NS,
    DEFAULT_CODE_THEME, DEFAULT_LOG_EXACT, DEFAULT_SEARCH, DEFAULT_SYSTEM_PROMPT, DEFAULT_THINK,
    IPyAIExtension, astream_to_stdout, compact_tool_display, prompt_from_lines, transform_backticks)

class DummyAsyncFormatter:
    async def format_stream(self, stream):
        async for o in stream: yield o

class TTYStringIO(io.StringIO):
    def isatty(self): return True

class DummyMarkdown:
    def __init__(self, text, **kwargs): self.text,self.kwargs = text,kwargs

class DummyConsole:
    instances = []
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.printed = []
        type(self).instances.append(self)

    def print(self, obj):
        self.printed.append(obj)
        self.kwargs["file"].write(f"RICH:{obj.text}")

class DummyLive:
    instances = []
    def __init__(self, renderable, **kwargs):
        self.kwargs = kwargs
        self.renderables = [renderable]
        type(self).instances.append(self)

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): self.kwargs["console"].print(self.renderables[-1])
    def update(self, renderable, refresh=False): self.renderables.append(renderable)

class DummyAsyncChat:
    instances = []
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        type(self).instances.append(self)

    async def __call__(self, prompt, stream=False, **kwargs):
        self.calls.append((prompt, stream, kwargs))
        async def _stream():
            yield "first "
            yield "second"
        return _stream()

class DummyHistory:
    def __init__(self, session_number=1):
        self.session_number = session_number
        self.db = sqlite3.connect(":memory:")
        self.entries = {}

    def add(self, line, source, output=None): self.entries[line] = (source, output)
    def get_range(self, session=0, start=1, stop=None, raw=True, output=False):
        if stop is None: stop = max(self.entries, default=0) + 1
        for i in range(start, stop):
            if i not in self.entries: continue
            src,out = self.entries[i]
            yield (0, i, (src, out) if output else src)

class DummyInputTransformerManager:
    def __init__(self): self.cleanup_transforms = []

class DummyShell:
    def __init__(self):
        self.input_transformer_manager = DummyInputTransformerManager()
        self.user_ns = {}
        self.magics = []
        self.history_manager = DummyHistory()
        self.execution_count = 2
        self.ran_cells = []
        self.loop_runner = asyncio.run

    def register_magics(self, magics): self.magics.append(magics)

    def run_cell(self, source, store_history=False):
        self.ran_cells.append((source, store_history))
        if store_history:
            self.history_manager.add(self.execution_count, source)
            self.execution_count += 1
        return SimpleNamespace(success=True)


@pytest.fixture(autouse=True)
def _config_paths(monkeypatch, tmp_path):
    cfg_dir = tmp_path/"ipyai"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(core, "CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(core, "CONFIG_PATH", cfg_dir/"config.json")
    monkeypatch.setattr(core, "SYSP_PATH", cfg_dir/"sysp.txt")
    monkeypatch.setattr(core, "STARTUP_PATH", cfg_dir/"startup.json")
    monkeypatch.setattr(core, "LOG_PATH", cfg_dir/"exact-log.jsonl")


@pytest.fixture
def dummy_ai(monkeypatch):
    DummyAsyncChat.instances = []
    monkeypatch.setattr(core, "AsyncChat", DummyAsyncChat)

    async def _fake_astream_to_stdout(stream, **kwargs): return "".join([o async for o in stream])
    monkeypatch.setattr(core, "astream_to_stdout", _fake_astream_to_stdout)
    return DummyAsyncChat


def test_prompt_from_lines_drops_continuation_backslashes():
    lines = ["`plan this work\\\n", "with two lines\n"]
    assert prompt_from_lines(lines) == "plan this work\nwith two lines\n"


def test_transform_backticks_executes_ai_magic_call():
    seen = {}
    class DummyIPython:
        def run_cell_magic(self, magic, line, cell): seen.update(magic=magic, line=line, cell=cell)
    code = "".join(transform_backticks(["`hello\n", "world\n"]))
    exec(code, {"get_ipython": lambda: DummyIPython()})
    assert seen == dict(magic="ipyai", line="", cell="hello\nworld\n")


async def _chunks(*items):
    for o in items: yield o


def run_stream(*items, **kwargs): return asyncio.run(astream_to_stdout(_chunks(*items), formatter_cls=DummyAsyncFormatter, **kwargs))


def mk_ext(load=True, **kwargs):
    shell = DummyShell()
    ext = IPyAIExtension(shell=shell, **kwargs)
    return shell, ext.load() if load else ext


def test_astream_to_stdout_collects_streamed_text():
    out = io.StringIO()
    text = run_stream("a", "b", out=out)
    assert text == "ab"
    assert out.getvalue() == "ab\n"


def test_compact_tool_display_uses_summary_and_truncates_result():
    text = """Before

<details class='tool-usage-details'>
<summary>f(x=1)</summary>

```json
{"id":"toolu_1","call":{"function":"f","arguments":{"x":"1"}},"result":"%s"}
```

</details>

After""" % ("a" * 120)

    res = compact_tool_display(text)
    assert "🔧 f(x=1) => " in res
    assert "a" * 120 not in res
    assert "..." in res
    assert "\n\n\n🔧" not in res
    assert "🔧 f(x=1)" in res


def test_astream_to_stdout_uses_live_markdown_for_tty_and_returns_full_text():
    tool_block = """<details class='tool-usage-details'>
<summary>f(x=1)</summary>

```json
{"id":"toolu_1","call":{"function":"f","arguments":{"x":"1"}},"result":"2"}
```

    </details>"""
    DummyConsole.instances = []
    DummyLive.instances = []
    out = TTYStringIO()
    text = run_stream(tool_block, out=out, console_cls=DummyConsole, markdown_cls=DummyMarkdown, live_cls=DummyLive)
    assert text == tool_block
    assert out.getvalue() == "RICH:🔧 f(x=1) => 2"
    assert DummyLive.instances[-1].renderables[-1].text == "🔧 f(x=1) => 2"


def test_astream_to_stdout_uses_rich_markdown_options_for_live_updates():
    DummyConsole.instances = []
    DummyLive.instances = []
    out = TTYStringIO()
    text = run_stream("`x`", out=out, code_theme="github-dark", console_cls=DummyConsole, markdown_cls=DummyMarkdown, live_cls=DummyLive)

    assert text == "`x`"
    md = DummyLive.instances[-1].renderables[-1]
    assert md.text == "`x`"
    assert md.kwargs == dict(code_theme="github-dark", inline_code_theme="github-dark", inline_code_lexer="python")


def test_astream_to_stdout_updates_live_markdown_as_chunks_arrive():
    DummyConsole.instances = []
    DummyLive.instances = []
    out = TTYStringIO()
    text = run_stream("a", "b", out=out, console_cls=DummyConsole, markdown_cls=DummyMarkdown, live_cls=DummyLive)

    assert text == "ab"
    assert [o.text for o in DummyLive.instances[-1].renderables] == ["a", "ab"]
    assert out.getvalue() == "RICH:ab"


def test_patch_inspect_getfile_coerces_pathlike_results_to_str(monkeypatch):
    monkeypatch.setattr(inspect, "_orig_getfile", lambda obj: Path("/tmp/demo.py"))
    assert inspect.getfile(object()) == "/tmp/demo.py"


def test_patch_syntax_tb_coerces_non_string_msg():
    calls = []

    class WeirdMsg:
        def __str__(self): return "coerced"

    tb = SyntaxTB(theme_name="linux")
    tb._orig_structured_traceback = lambda etype,evalue,etb,tb_offset=None,context=5: calls.append((etype, evalue.msg, etb, tb_offset, context)) or ["ok"]
    err = SimpleNamespace(msg=WeirdMsg())

    assert tb.structured_traceback(ValueError, err, None, tb_offset=2, context=7) == ["ok"]
    assert err.msg == "coerced"
    assert calls == [(ValueError, "coerced", None, 2, 7)]


def test_extension_load_is_idempotent_and_tracks_last_response(dummy_ai):
    shell,ext = mk_ext()
    ext.load()
    assert shell.input_transformer_manager.cleanup_transforms == [transform_backticks]
    assert len(shell.magics) == 1
    assert shell.user_ns[EXTENSION_NS] is ext

    ext.run_prompt("tell me something")

    assert dummy_ai.instances[-1].kwargs["hist"] == []
    assert dummy_ai.instances[-1].calls == [(
        "<user-request>tell me something</user-request>",
        True,
        dict(search=DEFAULT_SEARCH, think=DEFAULT_THINK),
    )]
    assert dummy_ai.instances[-1].kwargs["model"] == ext.model
    assert dummy_ai.instances[-1].kwargs["sp"] == DEFAULT_SYSTEM_PROMPT
    assert shell.user_ns[LAST_PROMPT] == "tell me something"
    assert shell.user_ns[LAST_RESPONSE] == "first second"
    assert ext.prompt_rows() == [("tell me something", "first second")]
    assert ext.prompt_records()[0][3] == 1


def test_unexpected_prompt_table_schema_is_recreated():
    shell = DummyShell()
    with shell.history_manager.db:
        shell.history_manager.db.execute("CREATE TABLE ai_prompts (id INTEGER PRIMARY KEY AUTOINCREMENT, session INTEGER NOT NULL, "
                                         "prompt TEXT NOT NULL, response TEXT NOT NULL, history_line INTEGER NOT NULL DEFAULT 0, "
                                         "prompt_line INTEGER NOT NULL DEFAULT 0)")
        shell.history_manager.db.execute("INSERT INTO ai_prompts (session, prompt, response, history_line, prompt_line) VALUES "
                                         "(1, 'p', 'r', 1, 2)")

    ext = IPyAIExtension(shell=shell)

    assert ext.prompt_records() == []
    cols = [o[1] for o in shell.history_manager.db.execute("PRAGMA table_info(ai_prompts)")]
    assert cols == ["id", "session", "prompt", "response", "history_line"]


def test_config_file_is_created_and_loaded():
    _,ext = mk_ext(load=False)

    assert core.CONFIG_PATH.exists()
    assert core.SYSP_PATH.exists()
    assert core.STARTUP_PATH.exists()
    data = json.loads(core.CONFIG_PATH.read_text())
    assert data["model"] == ext.model
    assert data["think"] == DEFAULT_THINK
    assert data["search"] == DEFAULT_SEARCH
    assert data["code_theme"] == DEFAULT_CODE_THEME
    assert data["log_exact"] == DEFAULT_LOG_EXACT
    assert core.SYSP_PATH.read_text() == DEFAULT_SYSTEM_PROMPT
    assert json.loads(core.STARTUP_PATH.read_text()) == dict(version=1, events=[])
    assert ext.system_prompt == DEFAULT_SYSTEM_PROMPT


def test_existing_sysp_file_is_loaded():
    sysp_path = core.SYSP_PATH
    sysp_path.write_text("custom sysp")
    _,ext = mk_ext(load=False)

    assert ext.system_prompt == "custom sysp"


def test_config_values_drive_model_think_and_search(dummy_ai):
    core.CONFIG_PATH.write_text(json.dumps(dict(model="cfg-model", think="m", search="h", log_exact=True)))
    shell,ext = mk_ext()

    ext.run_prompt("tell me something")

    assert ext.model == "cfg-model"
    assert ext.think == "m"
    assert ext.search == "h"
    assert ext.log_exact is True
    assert dummy_ai.instances[-1].calls == [(
        "<user-request>tell me something</user-request>",
        True,
        dict(search="h", think="m"),
    )]


def test_handle_line_can_report_and_set_model(capsys):
    _,ext = mk_ext(load=False, model="old-model", think="m", search="h", code_theme="github-dark", log_exact=True)

    ext.handle_line("")
    assert capsys.readouterr().out == (
        f"self.model='old-model'\nself.think='m'\nself.search='h'\nself.code_theme='github-dark'\nself.log_exact=True\n"
        f"CONFIG_PATH={core.CONFIG_PATH!r}\nSYSP_PATH={core.SYSP_PATH!r}\nSTARTUP_PATH={core.STARTUP_PATH!r}\n"
        f"LOG_PATH={core.LOG_PATH!r}\n"
    )

    ext.handle_line("model new-model")
    assert ext.model == "new-model"
    assert capsys.readouterr().out == "self.model='new-model'\n"

    ext.handle_line("think l")
    assert ext.think == "l"
    assert capsys.readouterr().out == "self.think='l'\n"

    ext.handle_line("search m")
    assert ext.search == "m"
    assert capsys.readouterr().out == "self.search='m'\n"

    ext.handle_line("code_theme ansi_dark")
    assert ext.code_theme == "ansi_dark"
    assert capsys.readouterr().out == "self.code_theme='ansi_dark'\n"

    ext.handle_line("log_exact false")
    assert ext.log_exact is False
    assert capsys.readouterr().out == "self.log_exact=False\n"


def test_second_prompt_uses_sqlite_prompt_history(dummy_ai):
    shell,ext = mk_ext()

    ext.run_prompt("first prompt")
    shell.execution_count = 3
    ext.run_prompt("second prompt")

    assert dummy_ai.instances[1].kwargs["hist"] == [
        "<user-request>first prompt</user-request>",
        "first second",
    ]
    assert ext.prompt_rows() == [
        ("first prompt", "first second"),
        ("second prompt", "first second"),
    ]


def test_reset_only_deletes_current_session_history(capsys):
    shell,ext = mk_ext()

    ext.save_prompt("s1 prompt", "s1 response", 1)
    shell.history_manager.session_number = 2
    shell.execution_count = 8
    ext.save_prompt("s2 prompt", "s2 response", 7)

    ext.handle_line("reset")

    assert capsys.readouterr().out == "Deleted 1 AI prompts from session 2.\n"
    assert ext.prompt_rows(session=1) == [("s1 prompt", "s1 response")]
    assert ext.prompt_rows(session=2) == []
    assert shell.user_ns[RESET_LINE_NS] == 7


def test_tools_resolve_from_ampersand_backticks():
    def demo():
        "Demo tool."
        return "ok"

    shell = DummyShell()
    shell.user_ns["demo"] = demo
    ext = IPyAIExtension(shell=shell).load()

    tools = ext.resolve_tools("please call &`demo` now", [])
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "demo"


def test_tools_resolve_callable_objects_by_namespace_name():
    class Demo:
        def __call__(self):
            "Demo tool."
            return "ok"

    shell = DummyShell()
    shell.user_ns["demo"] = Demo()
    ext = IPyAIExtension(shell=shell).load()

    tools = ext.resolve_tools("please call &`demo` now", [])
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "demo"


def test_context_xml_includes_code_and_outputs_since_last_prompt():
    shell = DummyShell()
    shell.history_manager.add(1, "a = 1")
    shell.history_manager.add(2, "a", "1")
    ext = IPyAIExtension(shell=shell).load()

    ctx = ext.code_context(1, 3)
    assert "<context><code>a = 1</code><code>a</code><output>1</output></context>\n" == ctx


def test_history_context_uses_lines_since_last_prompt_only():
    shell = DummyShell()
    shell.history_manager.add(1, "before = 1")
    shell.history_manager.add(2, "`first prompt")
    shell.history_manager.add(3, "after = 2")
    shell.execution_count = 3
    ext = IPyAIExtension(shell=shell).load()
    ext.save_prompt("first prompt", "first response", 2)

    prompt = ext.format_prompt("second prompt", ext.last_prompt_line()+1, 4)
    assert "before = 1" not in prompt
    assert "after = 2" in prompt


def test_startup_replays_code_and_restores_prompts():
    startup_path = core.STARTUP_PATH
    events = [dict(kind="code", line=1, source="import math"), dict(kind="prompt", line=3, history_line=2, prompt="hi", response="hello"),
              dict(kind="code", line=3, source="x = 1")]
    data = dict(version=1, events=events)
    startup_path.write_text(json.dumps(data))
    shell = DummyShell()
    shell.execution_count = 1
    ext = IPyAIExtension(shell=shell).load()

    assert shell.ran_cells == [("import math", True), ("x = 1", True)]
    assert ext.prompt_rows() == [("hi", "hello")]
    assert ext.prompt_records()[0][3] == 2
    assert ext.dialog_history()[0][0] == "<context><code>import math</code></context>\n<user-request>hi</user-request>"
    assert shell.execution_count == 4


def test_save_writes_startup_snapshot(capsys):
    startup_path = core.STARTUP_PATH
    shell = DummyShell()
    shell.history_manager.add(1, "import math")
    shell.history_manager.add(2, "`first prompt")
    shell.history_manager.add(3, "x = 1")
    shell.execution_count = 4
    ext = IPyAIExtension(shell=shell).load()
    ext.save_prompt("first prompt", "first response", 1)

    ext.handle_line("save")

    assert capsys.readouterr().out == f"Saved 2 code cells and 1 prompts to {startup_path}.\n"
    assert json.loads(startup_path.read_text()) == dict(version=1, events=[
        dict(kind="code", line=1, source="import math"),
        dict(kind="prompt", line=2, history_line=1, prompt="first prompt", response="first response"),
        dict(kind="code", line=3, source="x = 1"),
    ])


def test_log_exact_writes_full_prompt_and_response(dummy_ai):
    log_path = core.LOG_PATH
    shell = DummyShell()
    shell.history_manager.add(1, "a = 1")
    shell.execution_count = 3
    ext = IPyAIExtension(shell=shell, log_exact=True).load()

    ext.run_prompt("tell me something")

    rec = json.loads(log_path.read_text().strip())
    assert rec["session"] == 1
    assert rec["prompt"] == "<context><code>a = 1</code></context>\n<user-request>tell me something</user-request>"
    assert rec["response"] == "first second"


def test_cleanup_transform_prevents_help_syntax_interference():
    tm = TransformerManager()
    tm.cleanup_transforms.insert(1, transform_backticks)

    code = tm.transform_cell("`I am testing my new AI prompt system.\\\nTell me do you see a newline in this prompt?")
    assert code == "get_ipython().run_cell_magic('ipyai', '', 'I am testing my new AI prompt system.\\nTell me do you see a newline in this prompt?\\n')\n"
    assert tm.check_complete("`I am testing my new AI prompt system.\\") == ("incomplete", 0)
    assert tm.check_complete("`I am testing my new AI prompt system.\\\nTell me do you see a newline in this prompt?") == ("complete", None)

# DEV

This project is small. Nearly all runtime behavior lives in [ipyai/core.py](ipyai/core.py), so getting productive mainly means understanding that file and the tests in [tests/test_core.py](tests/test_core.py).

## Setup

Install in editable mode:

```bash
pip install -e .[dev]
```

Run tests:

```bash
pytest
```

This repo is configured for fastship releases:

```bash
ship-changelog
ship-release
```

## Current Scope

Implemented:

- backtick-to-magic rewriting using IPython cleanup transforms
- multiline prompts
- session-scoped prompt persistence in SQLite
- startup snapshot save/replay through `startup.json`
- dynamic code/output context reconstruction
- ampersand-backtick tool exposure from `user_ns`
- streaming responses
- live Rich markdown rendering in terminal IPython
- XDG-backed config, startup, and system prompt files
- optional exact raw prompt/response logging

## File Map

- [ipyai/core.py](ipyai/core.py): extension logic, config loading, prompt/history building, tool resolution, streaming, Rich rendering
- [ipyai/__init__.py](ipyai/__init__.py): package exports and version
- [tests/test_core.py](tests/test_core.py): focused unit tests for the transformation, history, config, tools, and rendering behavior
- [pyproject.toml](pyproject.toml): packaging and fastship configuration

## Prompt History And Context

Each AI prompt is saved in an `ai_prompts` table inside IPython's history SQLite database. Rows are keyed by the current IPython `session_number` and include:

- `prompt`
- `response`
- `history_line`

Stored rows contain only the user prompt, full AI response, and the line where the code context for that prompt stops.

Example:

```python
In [1]: import math
In [2]: `first prompt
In [3]: x = 1
In [4]: `second prompt
```

The stored rows are roughly:

- first prompt: `history_line=1`
- second prompt: `history_line=3`

So for the second prompt, `ipyai` knows:

- the code context before it should include `x = 1`, but not `import math`
- the prompt itself happened immediately after line 3

For each new prompt, `ipyai` reconstructs chat history as alternating user / assistant entries:

- the user entry is `<context>...</context><user-request>...</user-request>`
- the assistant entry is the stored full response

The `<context>` block contains all non-`ipyai` code run since the previous AI prompt in the current session, plus `Out[...]` history when IPython has it. The XML is intentionally simple:

```xml
<context><code>a = 1</code><code>a</code><output>1</output></context>
```

## Runtime Flow

The extension lifecycle is:

1. `%load_ext ipyai` calls `load_ipython_extension`, which delegates to `create_extension`.
2. `IPyAIExtension.load()` registers `%ipyai` / `%%ipyai`, inserts a cleanup transform into IPython's `input_transformer_manager.cleanup_transforms`, and applies `startup.json` if the session is still fresh.
3. Any cell whose first character is `` ` `` is rewritten by `transform_backticks()` into `get_ipython().run_cell_magic('ipyai', '', prompt)`.
4. `AIMagics.ipyai()` routes line input to `handle_line()` and cell input to `run_prompt()`.
5. `run_prompt()` reconstructs conversation history, resolves tools, calls `lisette.Chat`, streams the response, optionally writes an exact raw prompt/response log entry, stores the full response, and returns `None`.

## Why Cleanup Transforms

The backtick rewrite happens in `cleanup_transforms`, not in a later input transformer. That matters because IPython's own parsing for help syntax and similar features can interfere with raw backtick prompts if the rewrite happens too late.

This is the mechanism that makes these cases work correctly:

- multiline pasted prompts
- prompts containing `?`
- backslash-Enter continuation

## Prompt Construction

The stored prompt text is not the exact user message sent to the model. The actual user entry is built dynamically with:

```xml
{context}<user-request>{prompt}</user-request>
```

`context` is empty when there has been no intervening code. Otherwise it is:

```xml
<context><code>...</code><output>...</output>...</context>
```

Important detail: only the raw prompt and raw response are stored in SQLite. Context is regenerated on each run from normal IPython history. That keeps the table small and avoids baking transient context into stored rows.

## SQLite Storage

`ipyai` uses IPython's existing history database connection at `shell.history_manager.db`.

Table schema:

```sql
CREATE TABLE IF NOT EXISTS ai_prompts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session INTEGER NOT NULL,
  prompt TEXT NOT NULL,
  response TEXT NOT NULL,
  history_line INTEGER NOT NULL DEFAULT 0
)
```

Notes:

- rows are scoped by IPython `session_number`
- `history_line` is used to decide which code cells belong in the next prompt's generated `<context>` block
- if `ai_prompts` does not match the expected schema, `ipyai` drops and recreates it instead of migrating it
- `%ipyai reset` deletes only current-session rows and sets a reset baseline in `user_ns`

## Startup Snapshot

`startup.json` is stored next to the other XDG files.

`%ipyai save` writes a merged event stream for the current session:

- code events store the source to replay
- prompt events store the prompt, saved response, and prompt position

On a fresh load:

- code events are replayed with `run_cell(..., store_history=True)`
- prompt events are inserted directly into `ai_prompts`
- `execution_count` is advanced for restored prompt events so later saves preserve ordering

This lets a new terminal session pick up imports, helper functions, tool definitions, and prior AI chat history without re-running the prompts themselves.

## Code Context Reconstruction

`code_context(start, stop)` pulls normal IPython history with:

```python
history_manager.get_range(session=0, start=start, stop=stop, raw=True, output=True)
```

Rules:

- inputs that look like `ipyai` commands are skipped
- normal code becomes `<code>...</code>`
- output history, when present, becomes `<output>...</output>`
- there is no notebook markdown-cell concept here; this is terminal IPython

If `output_history` is disabled or a cell has no stored output, context still works; there is just no `<output>` tag for that entry.

## Tool Resolution

Tool references are written in prompts as `&\`name\``.

`resolve_tools()`:

- finds tool names in the current prompt and prior prompts in the rebuilt dialog history
- looks them up in `shell.user_ns`
- raises `NameError` for missing tools
- raises `TypeError` for non-callables
- passes the resolved callables to `lisette.Chat(..., tools=...)`

The tool lookup is intentionally live against the active namespace, so changing a function in the IPython session changes the tool used by subsequent prompts.

## Streaming And Display

Streaming and storage are deliberately separated.

`stream_to_stdout()`:

1. uses `lisette.StreamFormatter` to iterate the response stream
2. in a TTY, updates a `rich.live.Live` view with `Markdown(...)` as chunks arrive
3. outside a TTY, writes raw chunks to stdout
4. returns the full original text for storage

The visible output is post-processed by `compact_tool_display()` so lisette tool detail blocks are shown in a shorter form like:

```text
🔧 f(x=1) => 2
```

That rewrite affects only the visible terminal output. SQLite keeps the original response.

## Config And System Prompt

XDG paths are used via `fastcore.xdg.xdg_config_home()`:

- `config.json`: model, think, search, Rich code theme, and the exact-log flag
- `sysp.txt`: system prompt passed as `sp=` to `lisette.Chat`
- `startup.json`: saved startup snapshot for fresh sessions
- `exact-log.jsonl`: optional raw prompt/response log output

Creation behavior:

- all three files are auto-created if missing
- the initial `model` defaults from `IPYAI_MODEL` if present
- runtime `%ipyai model ...` and similar commands change only the live extension object, not the config file

When `log_exact` is enabled, the log file contains the exact fully-expanded prompt passed to the model and the exact raw response returned from the stream.

## Tests

The test suite is intentionally small and uses dummy shell, history, chat, formatter, console, and markdown objects.

Coverage currently focuses on:

- backtick parsing and continuation handling
- cleanup-transform rewriting
- prompt/history persistence
- context generation
- tool resolution
- config and system prompt file creation
- startup save/replay
- raw exact logging
- final Rich render behavior
- terminal clear logic around rewritten output

When changing behavior in [ipyai/core.py](ipyai/core.py), update or add the narrowest possible test in [tests/test_core.py](tests/test_core.py).

## Common Change Points

If you want to change prompt parsing or magic routing:

- edit `is_backtick_prompt()`, `prompt_from_lines()`, or `transform_backticks()`

If you want to change the XML or history sent to the model:

- edit `_prompt_template`, `code_context()`, `format_prompt()`, or `dialog_history()`

If you want to change tool behavior:

- edit `_tool_names()`, `_tool_refs()`, or `resolve_tools()`

If you want to change terminal rendering:

- edit `compact_tool_display()`, `_clear_terminal_block()`, `_render_markdown()`, or `stream_to_stdout()`

If you want to change persistence:

- edit `ensure_prompt_table()`, `prompt_records()`, `save_prompt()`, `save_startup()`, `apply_startup()`, and `reset_session_history()`

## Working Assumptions

- the primary target is terminal IPython
- notebook markdown-cell semantics are out of scope unless explicitly added later
- prompt rows should remain compact; dynamic context generation is preferred over storing expanded prompts
- stored responses should keep full fidelity, even when terminal rendering is simplified

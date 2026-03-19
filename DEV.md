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

- period-to-magic rewriting using IPython cleanup transforms
- multiline prompts with backslash-Enter continuation
- notes: string-literal-only cells detected via `ast` and sent as `<note>` blocks in context
- session-scoped prompt persistence in SQLite
- startup snapshot save/replay through `startup.ipynb` (nbformat v4.5 with cell IDs)
- notes saved as markdown cells, code as code cells, prompts as markdown with metadata
- dynamic code/output/note context reconstruction
- ampersand-backtick tool exposure from `user_ns`
- Agent Skills discovery from `.agents/skills/` (CWD + parents) and `~/.config/agents/skills/`
- `load_skill` tool automatically available when skills are found (returns `FullResponse` to avoid truncation)
- skills list frozen at extension load time (security: prevents LLM from creating and loading skills mid-session)
- streaming responses with live Rich markdown rendering in TTY
- thinking indicator (🧠) displayed as progress and stripped from display once content arrives
- tool call display compacted to single-line `🔧 f(x=1) => 2` form
- keyboard shortcuts: Alt-Shift-W (all code blocks), Alt-Shift-1..9 (nth block) via prompt_toolkit
- code block extraction uses `mistletoe` markdown parser (not regex) for correctness
- XDG-backed config, startup, and system prompt files
- optional exact raw prompt/response logging
- minimal IPython compatibility patches for `SyntaxTB` and `inspect.getfile`

## File Map

- [ipyai/core.py](ipyai/core.py): extension logic, XDG path globals, config loading, prompt/history building, tool resolution, skill discovery, async streaming, Rich rendering, keybindings
- [ipyai/__init__.py](ipyai/__init__.py): package exports and version
- [tests/test_core.py](tests/test_core.py): focused unit tests for transformation, history, config, tools, notes, skills, rendering, and thinking display
- [pyproject.toml](pyproject.toml): packaging and fastship configuration
- [.agents/skills/](/.agents/skills/): project-local Agent Skills

## Prompt History And Context

Each AI prompt is saved in an `ai_prompts` table inside IPython's history SQLite database. Rows are keyed by the current IPython `session_number` and include:

- `prompt`
- `response`
- `history_line`

Stored rows contain only the user prompt, full AI response, and the line where the code context for that prompt stops.

Example:

```python
In [1]: import math
In [2]: .first prompt
In [3]: x = 1
In [4]: .second prompt
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

The `<context>` block contains all non-`ipyai` code run since the previous AI prompt in the current session, plus `Out[...]` history when IPython has it. String-literal-only cells are sent as `<note>` instead of `<code>` (detected via `ast`). The XML is intentionally simple:

```xml
<context><code>a = 1</code><note>This is a note</note><code>a</code><output>1</output></context>
```

## Runtime Flow

The extension lifecycle is:

1. `%load_ext ipyai` calls `load_ipython_extension`, which delegates to `create_extension`.
2. `IPyAIExtension.__init__` loads config, system prompt, discovers skills, and loads the startup file.
3. `IPyAIExtension.load()` registers `%ipyai` / `%%ipyai`, inserts a cleanup transform into IPython's `input_transformer_manager.cleanup_transforms`, registers keybindings, and applies `startup.ipynb` if the session is still fresh.
4. Any cell whose first character is `.` is rewritten by `transform_dots()` into `get_ipython().run_cell_magic('ipyai', '', prompt)`.
5. `AIMagics.ipyai()` routes line input to `handle_line()` and cell input directly to the `_run_prompt()` coroutine (returned to the async `run_cell_magic` patch for awaiting).
6. `_run_prompt()` reconstructs conversation history, resolves tools, adds skills tools/system prompt if skills were discovered, runs `lisette.AsyncChat`, streams the response, optionally writes an exact log entry, and stores the full response.

At import time, `ipyai` also applies two small global IPython bugfixes borrowed from `ipykernel_helper`:

- `SyntaxTB.structured_traceback` coerces non-string `evalue.msg` values to `str`
- `inspect.getfile` is wrapped to always return a string

## Why Cleanup Transforms

The period rewrite happens in `cleanup_transforms`, not in a later input transformer. That matters because IPython's own parsing for help syntax and similar features can interfere with raw prompts if the rewrite happens too late.

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
<context><code>...</code><note>...</note><output>...</output>...</context>
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

`startup.ipynb` is stored as a Jupyter notebook (nbformat v4.5 with cell IDs) next to the other XDG files.

`%ipyai save` writes a merged event stream for the current session as notebook cells:

- code events become code cells (with `metadata.ipyai.kind="code"`)
- string-literal-only code (notes) become markdown cells (with original source preserved in `metadata.ipyai.source` for round-trip replay)
- prompt events become markdown cells containing the AI response (with prompt text in `metadata.ipyai.prompt`)

On a fresh load:

- code cells (including notes) are replayed with `run_cell(..., store_history=True)`
- prompt cells are restored into `ai_prompts` from metadata
- `execution_count` is advanced for restored prompt events so later saves preserve ordering

Legacy `startup.json` files (pre-notebook format) are still supported for loading.

## Skills

Skills follow the [Agent Skills specification](https://agentskills.io/specification.md). Discovery happens once at extension init time via `_discover_skills()`:

1. Walk from CWD up through all parent directories, scanning `.agents/skills/` in each
2. Scan `~/.config/agents/skills/`
3. Deduplicate by resolved path; closer-to-CWD skills take priority

Each skill directory must contain a `SKILL.md` with YAML frontmatter (`name`, `description`). Frontmatter is parsed with PyYAML.

At runtime, if skills were discovered:

- the system prompt gets a `<skills>` section listing all skill names, paths, and descriptions
- a `load_skill` tool is added to the tools list (reads `SKILL.md` and returns as `FullResponse`)
- the tool namespace is a merged copy of `user_ns` (does not pollute the user's namespace)

The skills list is frozen at load time to prevent the LLM from creating and loading skills during a session.

## Code Context Reconstruction

`code_context(start, stop)` pulls normal IPython history with:

```python
history_manager.get_range(session=0, start=start, stop=stop, raw=True, output=True)
```

Rules:

- inputs that look like `ipyai` commands (starting with `.` or `%ipyai`) are skipped
- string-literal-only cells (detected by `_is_note` via `ast.parse`) become `<note>` tags containing the string value
- normal code becomes `<code>...</code>`
- output history, when present, becomes `<output>...</output>`

## Tool Resolution

Tool references are written in prompts as `&`name``.

`resolve_tools()`:

- finds tool names in the current prompt and prior prompts in the rebuilt dialog history
- looks them up in `shell.user_ns`
- raises `NameError` for missing tools
- raises `TypeError` for non-callables
- builds tool schemas with `get_schema_nm(...)` so the exposed tool name matches the namespace symbol instead of `__call__` for callable objects
- passes those schemas to `lisette.AsyncChat(..., tools=...)`

The tool lookup is intentionally live against the active namespace, so changing a function in the IPython session changes the tool used by subsequent prompts. Async callables are handled by `lisette.AsyncChat`, so tool results are awaited correctly.

## Streaming And Display

Streaming and storage are deliberately separated.

`astream_to_stdout()`:

1. uses `lisette.AsyncStreamFormatter` to iterate the response stream
2. in a TTY, updates a `rich.live.Live` view with `Markdown(...)` as chunks arrive
3. outside a TTY, writes raw chunks to stdout
4. returns the full original text for storage

Display processing (`_display_text`):

- `_strip_thinking` removes 🧠 emoji lines once actual content follows (shows them as a progress indicator during thinking, strips them from the final display)
- `compact_tool_display` rewrites lisette tool detail blocks to a short `🔧 f(x=1) => 2` form
- these affect only the visible terminal output; SQLite keeps the original response

`ipyai` wraps the streaming phase in a small guard that temporarily marks `shell.display_pub._is_publishing = True`. That keeps terminal-visible AI output out of IPython's normal stdout capture and therefore out of `output_history`, while still allowing `ipyai` to store the full response in `ai_prompts`.

## Keybindings

Registered via prompt_toolkit on `shell.pt_app.key_bindings` during `load()`:

- `escape, W` (Alt-Shift-W): insert all Python/untagged code blocks from `_ai_last_response`
- `escape, !` through `escape, (` (Alt-Shift-1 through Alt-Shift-9): insert the Nth code block

Code blocks are extracted using `mistletoe.Document` and `CodeFence` — only blocks tagged `python`, `py`, or untagged are included.

## Config And System Prompt

XDG-backed module globals are defined at import time:

- `CONFIG_PATH`: model, think, search, Rich code theme, and the exact-log flag
- `SYSP_PATH`: system prompt passed as `sp=` to `lisette.AsyncChat`
- `STARTUP_PATH`: saved startup snapshot (`.ipynb` format)
- `LOG_PATH`: optional raw prompt/response log output

Creation behavior:

- these files are created on demand when first needed
- the initial `model` defaults from `IPYAI_MODEL` if present
- runtime `%ipyai model ...` and similar commands change only the live extension object, not the config file

When `log_exact` is enabled, the log file contains the exact fully-expanded prompt passed to the model and the exact raw response returned from the stream.

## Tests

The test suite uses dummy shell, history, chat, formatter, console, and markdown objects.

Coverage currently focuses on:

- period prompt parsing and continuation handling
- cleanup-transform rewriting
- prompt/history persistence
- context generation including notes (`<note>` tags)
- tool resolution
- config and system prompt file creation
- startup save/replay in ipynb format with cell IDs
- startup round-trip for notes (markdown cells with preserved source)
- raw exact logging
- Rich live markdown rendering
- thinking emoji stripping
- skill discovery, parsing, XML generation, and `load_skill`
- skills integration in `_run_prompt`
- code block extraction

When changing behavior in [ipyai/core.py](ipyai/core.py), update or add the narrowest possible test in [tests/test_core.py](tests/test_core.py).

## Common Change Points

If you want to change prompt parsing or magic routing:

- edit `is_dot_prompt()`, `prompt_from_lines()`, or `transform_dots()`

If you want to change the XML or history sent to the model:

- edit `_prompt_template`, `code_context()`, `format_prompt()`, or `dialog_history()`

If you want to change notes behavior:

- edit `_is_note()`, `_note_str()`, and the note handling in `code_context()`

If you want to change tool behavior:

- edit `_tool_names()`, `_tool_refs()`, or `resolve_tools()`

If you want to change skills:

- edit `_parse_skill()`, `_discover_skills()`, `_skills_xml()`, `load_skill()`, and the skills block in `_run_prompt()`

If you want to change terminal rendering:

- edit `_display_text()`, `_strip_thinking()`, `compact_tool_display()`, `_astream_to_live_markdown()`, `_markdown_renderable()`, or `astream_to_stdout()`

If you want to change persistence:

- edit `ensure_prompt_table()`, `prompt_records()`, `save_prompt()`, `save_startup()`, `apply_startup()`, and `reset_session_history()`

If you want to change the startup notebook format:

- edit `_event_to_cell()`, `_cell_to_event()`, `_default_startup()`, `load_startup()`, and `save_startup()`

If you want to change keybindings:

- edit `_register_keybindings()` and `_extract_code_blocks()`

## Working Assumptions

- the primary target is terminal IPython
- prompt rows should remain compact; dynamic context generation is preferred over storing expanded prompts
- stored responses should keep full fidelity, even when terminal rendering is simplified
- skills are discovered once at load time and never re-scanned during a session

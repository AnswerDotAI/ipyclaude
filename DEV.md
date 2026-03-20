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
- unified tool discovery from prompts, skills, notes, and tool responses via `_tool_refs()`
- `_parse_frontmatter()` shared helper for extracting YAML frontmatter from skills, notes, and tool results
- `allowed-tools` frontmatter key in skills and notes for declaring tool dependencies
- tool results with qualifying frontmatter (`allowed-tools` or `eval: true`) contribute tools
- Agent Skills discovery from `.agents/skills/` (CWD + parents) and `~/.config/agents/skills/`
- `load_skill` tool added to `user_ns` at init time, resolved via normal tool mechanism (not special-cased)
- skills list frozen at extension load time (security: prevents LLM from creating and loading skills mid-session)
- streaming responses with live Rich markdown rendering in TTY
- thinking indicator (🧠) displayed as progress and stripped from display once content arrives
- tool call display compacted to single-line `🔧 f(x=1) => 2` form
- AI inline completion via Alt-. (calls `completion_model` with session context, shows as prompt_toolkit suggestion; partial accept via M-f preserves remaining suggestion)
- keyboard shortcuts: Alt-Up/Down (history jump), Alt-Shift-W (all code blocks), Alt-Shift-1..9 (nth block), Alt-Shift-Up/Down (cycle blocks) via prompt_toolkit
- code block extraction uses `mistletoe` markdown parser (not regex) for correctness
- syntax highlighting disabled for `.` prompts and `%%ipyai` cells (patches `IPythonPTLexer` at class level)
- XDG-backed config, startup, and system prompt files
- optional exact raw prompt/response logging
- skill eval blocks: `#| eval: true` python code blocks in skills are executed via `shell.run_cell` when loaded
- per-directory session persistence: CWD stored in IPython `sessions.remark`, session resume via `resume_session()`
- interactive session picker via `prompt_toolkit.radiolist_dialog` for `ipyai -r`
- `%ipyai sessions` command listing resumable sessions with last prompt preview
- `ipyai` CLI entry point (console script) launching IPython with ipythonng + ipyai + output history
- minimal IPython compatibility patches for `SyntaxTB` and `inspect.getfile` (guarded with `once=True` to coexist with ipykernel_helper)

## File Map

- [ipyai/core.py](ipyai/core.py): extension logic, XDG path globals, config loading, prompt/history building, tool resolution, skill discovery, session persistence/resume, async streaming, Rich rendering, keybindings
- [ipyai/cli.py](ipyai/cli.py): `ipyai` console script entry point — parses flags via `ipythonng.cli.parse_flags`, launches IPython with extensions and output history
- [ipyai/__init__.py](ipyai/__init__.py): package exports and version
- [tests/test_core.py](tests/test_core.py): focused unit tests for transformation, history, config, tools, notes, skills, sessions, rendering, and thinking display
- [pyproject.toml](pyproject.toml): packaging, console script (`ipyai`), and fastship configuration
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

1. `%load_ext ipyai` calls `load_ipython_extension`, which parses `IPYTHONNG_FLAGS` and delegates to `create_extension`.
2. `create_extension` ensures the `ai_prompts` table exists, optionally resumes a session (or shows the interactive picker), creates the extension, stores CWD in `sessions.remark`, and registers the atexit handler.
3. `IPyAIExtension.__init__` loads config, system prompt, discovers skills, and loads the startup file.
4. `IPyAIExtension.load()` registers `%ipyai` / `%%ipyai`, inserts a cleanup transform into IPython's `input_transformer_manager.cleanup_transforms`, registers keybindings, and applies `startup.ipynb` if the session is still fresh.
4. Any cell whose first character is `.` is rewritten by `transform_dots()` into `get_ipython().run_cell_magic('ipyai', '', prompt)`.
5. `AIMagics.ipyai()` routes line input to `handle_line()` and cell input directly to the `_run_prompt()` coroutine (returned to the async `run_cell_magic` patch for awaiting).
6. `_run_prompt()` reconstructs conversation history, resolves tools, adds skills tools/system prompt if skills were discovered, runs `lisette.AsyncChat`, streams the response, optionally writes an exact log entry, and stores the full response.

At import time, `ipyai` also applies two small global IPython bugfixes (shared with `ipykernel_helper`, guarded with `once=True` so only the first loader applies them):

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

## Session Persistence And Resume

`ipyai` stores the working directory in IPython's `sessions.remark` column (an unused TEXT field) at extension load time. This enables per-directory session listing and resume.

Key functions:

- `_list_sessions(db, cwd)` — queries sessions for the given directory, falls back to git repo root prefix match; includes the last AI prompt per session via a subquery on `ai_prompts`
- `_fmt_session()` — formats a session row for display (shared by `%ipyai sessions` and the interactive picker)
- `_pick_session(rows)` — interactive `radiolist_dialog` picker from prompt_toolkit
- `resume_session(shell, session_id)` — deletes the fresh session row, restores `session_number` and `execution_count`, pads `input_hist_parsed`/`input_hist_raw`, reopens the old session (clears `end` timestamp)

Resume is triggered by `IPYTHONNG_FLAGS` env var (set by the `ipyai` CLI when `-r` is passed). The `_ng_parser` (argparse) parses `-r <id>` or bare `-r` (const=-1 for interactive picker).

On exit, an `atexit` handler prints the session ID for easy resume.

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

Tools are discovered from multiple sources via `_tool_refs()`:

- `&`name`` in the current prompt and prior prompts in dialog history
- `allowed-tools` frontmatter and `&`name`` mentions in skills
- `&`name`` mentions and `allowed-tools` frontmatter in notes (string-literal cells)
- tool results in stored AI responses whose frontmatter contains `allowed-tools` or `eval: true`

Shared helpers:

- `_parse_frontmatter(text)` extracts YAML frontmatter from any text (reused by skills, notes, and tool results)
- `_allowed_tools(text)` combines frontmatter `allowed-tools` and `&`name`` mentions into a set of tool names
- `_tool_results(response)` scans stored response `<details>` blocks for qualifying tool results

`resolve_tools()`:

- validates tools from the current prompt (raises `NameError`/`TypeError` for missing or non-callable)
- collects all tool names from all sources via `_tool_refs()`
- silently skips tools from non-prompt sources that are missing from `user_ns`
- builds tool schemas with `get_schema_nm(...)` so the exposed tool name matches the namespace symbol instead of `__call__` for callable objects
- passes those schemas to `lisette.AsyncChat(..., tools=...)`

The `load_skill` tool is added to `user_ns` at extension init time when skills are discovered. It is resolved through the normal tool mechanism (skills always contribute `load_skill` to the tool name set) rather than being special-cased in `_run_prompt`.

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

- `escape, .` (Alt-.): AI inline completion — calls `_ai_complete()` as a background task, which builds a prompt from session context plus the current prefix/suffix and calls the configured `completion_model`. The result is set as `buffer.suggestion` (prompt_toolkit's auto-suggest display), accepted with right-arrow or word-at-a-time with M-f. IPython's existing auto-suggest `get_suggestion` is patched to remember the AI target text so partial accepts regenerate the remainder. Cancels safely if the buffer text changes before the response arrives.
- `escape, up` / `escape, down` (Alt-Up/Down): jump through complete history entries, bypassing line-by-line navigation in multiline inputs (calls `buffer.history_backward()` / `history_forward()`)
- `escape, W` (Alt-Shift-W): insert all Python code blocks from `_ai_last_response`
- `escape, !` through `escape, (` (Alt-Shift-1 through Alt-Shift-9): insert the Nth code block
- `escape, s-up` / `escape, s-down` (Alt-Shift-Up/Down): cycle through code blocks one at a time, replacing the buffer contents; prompt_toolkit swaps A/B for modifier-4 (Alt+Shift) arrows, so the bindings are intentionally inverted

Code blocks are extracted using `mistletoe.Document` and `CodeFence` — only blocks tagged `python` or `py` are included.

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

To run ipyai in isolation (no user config, startup, or history), set these environment variables:

- `XDG_CONFIG_HOME` — redirects ipyai's config files (`config.json`, `sysp.txt`, `startup.ipynb`)
- `IPYTHON_DIR` — redirects IPython's profile directory (prevents loading user `ipython_config.py` and startup scripts)
- `--HistoryManager.hist_file=<path>` — isolates the history database

The e2e test uses all three to create a fully isolated session via pexpect.

The test suite uses dummy shell, history, chat, formatter, console, and markdown objects.

Coverage currently focuses on:

- period prompt parsing and continuation handling
- cleanup-transform rewriting
- prompt/history persistence
- context generation including notes (`<note>` tags)
- tool resolution including unified discovery from skills, notes, and tool responses
- frontmatter parsing (`_parse_frontmatter`) and `allowed-tools` extraction
- config and system prompt file creation
- startup save/replay in ipynb format with cell IDs
- startup round-trip for notes (markdown cells with preserved source)
- raw exact logging
- Rich live markdown rendering
- thinking emoji stripping
- skill discovery, parsing, XML generation, `load_skill`, and eval blocks
- skills integration in `_run_prompt`
- session persistence: CWD in remark, list sessions, resume session
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

- edit `_tool_names()`, `_tool_refs()`, `_parse_frontmatter()`, `_allowed_tools()`, `_tool_results()`, or `resolve_tools()`

If you want to change skills:

- edit `_parse_skill()`, `_discover_skills()`, `_skills_xml()`, `load_skill()`, and the skills/tool collection in `_run_prompt()`

If you want to change terminal rendering:

- edit `_display_text()`, `_strip_thinking()`, `compact_tool_display()`, `_astream_to_live_markdown()`, `_markdown_renderable()`, or `astream_to_stdout()`

If you want to change persistence:

- edit `ensure_prompt_table()`, `prompt_records()`, `save_prompt()`, `save_startup()`, `apply_startup()`, and `reset_session_history()`

If you want to change the startup notebook format:

- edit `_event_to_cell()`, `_cell_to_event()`, `_default_startup()`, `load_startup()`, and `save_startup()`

If you want to change keybindings:

- edit `_register_keybindings()` and `_extract_code_blocks()`

If you want to change AI inline completion:

- edit `_ai_complete()`, `_COMPLETION_SP`, and the `escape, .` binding in `_register_keybindings()`

If you want to change syntax highlighting:

- edit `_patch_lexer()`

If you want to change session persistence or resume:

- edit `_list_sessions()`, `_fmt_session()`, `_pick_session()`, `resume_session()`, the `sessions` case in `handle_line()`, and the session handling in `create_extension()`

## Working Assumptions

- the primary target is terminal IPython
- prompt rows should remain compact; dynamic context generation is preferred over storing expanded prompts
- stored responses should keep full fidelity, even when terminal rendering is simplified
- skills are discovered once at load time and never re-scanned during a session

# ipyai

`ipyai` is an IPython extension that turns any input starting with `.` into an AI prompt.

It is aimed at terminal IPython, not notebook frontends.

## Install

```bash
pip install ipyai
```

## Load

```python
%load_ext ipyai
```

If you change the package in a running shell:

```python
%reload_ext ipyai
```

## How To Auto-Load `ipyai`

Add this to an `ipython_config.py` file used by terminal `ipython`:

```python
c.TerminalIPythonApp.extensions = ["ipyai.core"]
```

Good places for that file include:

- env-local: `{sys.prefix}/etc/ipython/ipython_config.py`
- user-local: `~/.ipython/profile_default/ipython_config.py`

In a virtualenv, the env-local path is usually `.venv/etc/ipython/ipython_config.py`.

To see which config paths your current `ipython` is searching, run:

```bash
ipython --debug -c 'exit()' 2>&1 | grep Searching
```

## Usage

Only the leading period is special. There is no closing delimiter.

Single line:

```python
.write a haiku about sqlite
```

Multiline paste:

```python
.summarize this module:
focus on state management
and persistence behavior
```

Backslash-Enter continuation in the terminal:

```python
.draft a migration plan \
with risks and rollback steps
```

`ipyai` also provides a line and cell magic named `%ipyai` / `%%ipyai`.

Note: `.01 * 3` and similar expressions starting with `.` followed by a digit will be interpreted as prompts. Write `0.01 * 3` instead.

## Notes

Any IPython cell containing only a string literal is treated as a "note". Notes provide context to the AI without being executable code:

```python
"This is a note explaining what I'm about to do"
```

Notes appear in the AI context as `<note>` blocks rather than `<code>` blocks. When saving a session, notes are stored as markdown cells in the startup notebook.

## `%ipyai` Commands

```python
%ipyai
%ipyai model claude-sonnet-4-6
%ipyai think m
%ipyai search h
%ipyai code_theme monokai
%ipyai log_exact true
%ipyai save
%ipyai reset
```

- `%ipyai` — show current settings and config file paths
- `%ipyai model ...` / `think ...` / `search ...` / `code_theme ...` / `log_exact ...` — change settings for the current session
- `%ipyai save` — save the current session (code, notes, and AI history) to `startup.ipynb`
- `%ipyai reset` — clear AI prompt history for the current session

## Tools

Expose a function from the active IPython namespace as a tool by referencing it with `&`name`` in the prompt:

```python
def weather(city): return f"Sunny in {city}"

.use &`weather` to answer the question about Brisbane
```

Callable objects and async callables are also supported.

## Skills

`ipyai` supports [Agent Skills](https://agentskills.io/) — reusable instruction sets that the AI can load on demand. Skills are discovered at extension load time from:

- `.agents/skills/` in the current directory and every parent directory
- `~/.config/agents/skills/`

Each skill is a directory containing a `SKILL.md` file with YAML frontmatter (`name`, `description`) and markdown instructions. At the start of each conversation, the AI sees a list of available skill names and descriptions. When a request matches a skill, the AI calls the `load_skill` tool to read its full instructions before responding.

See the [Agent Skills specification](https://agentskills.io/specification.md) for the full format.

## Keyboard Shortcuts

`ipyai` registers prompt_toolkit keybindings for inserting code from the last AI response into the current IPython input:

| Shortcut | Action |
|---|---|
| **Alt-Shift-W** | Insert all Python code blocks from the last response |
| **Alt-Shift-1** through **Alt-Shift-9** | Insert the Nth code block |
| **Alt-Shift-Up/Down** | Cycle through code blocks one at a time |

Code blocks are extracted from fenced markdown blocks tagged as `python` or `py`. Blocks tagged with other languages (bash, json, etc.) or untagged blocks are skipped.

## Startup Replay

`%ipyai save` snapshots the current session to `~/.config/ipyai/startup.ipynb`:

- code cells are saved as code cells (notes become markdown cells)
- AI prompts are saved with the response as markdown and the prompt in cell metadata

When `ipyai` loads into a fresh session, saved code is replayed and saved prompts are restored into the conversation history. This primes new sessions with imports, helpers, tools, and prior AI context without re-running the prompts.

## Output Rendering

Responses are streamed and rendered as markdown in the terminal via Rich. Thinking indicators (🧠) are displayed during model reasoning and removed once the response begins. Tool calls are compacted to a short form like `🔧 f(x=1) => 2`.

## Configuration

Config files live under `~/.config/ipyai/` and are created on demand:

| File | Purpose |
|---|---|
| `config.json` | Model, think/search level, code theme, log flag |
| `sysp.txt` | System prompt |
| `startup.ipynb` | Saved session snapshot |
| `exact-log.jsonl` | Raw prompt/response log (when `log_exact` is enabled) |

`config.json` supports:

```json
{
  "model": "claude-sonnet-4-6",
  "think": "l",
  "search": "l",
  "code_theme": "monokai",
  "log_exact": false
}
```

- `model` defaults from the `IPYAI_MODEL` environment variable if set when the config is first created
- `think` and `search` must be one of `l`, `m`, or `h`

## Development

See [DEV.md](DEV.md) for project layout, architecture, persistence details, and development workflow.

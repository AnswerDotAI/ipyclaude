# ipyai

`ipyai` is an IPython extension that turns any input starting with `` ` `` into an AI prompt.

It is aimed at terminal IPython, not notebook frontends. Prompts stream through `lisette`, final output is rendered with `rich`, and prompt history is stored alongside normal IPython history in the same SQLite database.

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

`ipyai` is designed for terminal IPython. To auto-load it, add this to an `ipython_config.py` file used by terminal `ipython`:

```python
c.TerminalIPythonApp.extensions = ["ipyai"]
```

Good places for that file include:

- env-local: `{sys.prefix}/etc/ipython/ipython_config.py`
- user-local: `~/.ipython/profile_default/ipython_config.py`
- system-wide IPython config directories

In a virtualenv, the env-local path is usually:

- `.venv/etc/ipython/ipython_config.py`

To see which config paths your current `ipython` is searching, run:

```bash
ipython --debug -c 'exit()' 2>&1 | grep Searching
```

## Usage

Only the leading backtick is special. There is no closing delimiter.

Single line:

```python
`write a haiku about sqlite
```

Multiline paste:

```python
`summarize this module:
focus on state management
and persistence behavior
```

Backslash-Enter continuation in the terminal:

```python
`draft a migration plan \
with risks and rollback steps
```

`ipyai` also provides a line and cell magic named `%ipyai` / `%%ipyai`.

## `%ipyai` commands

```python
%ipyai
%ipyai model claude-sonnet-4-6
%ipyai think m
%ipyai search h
%ipyai code_theme monokai
%ipyai reset
```

Behavior:

- `%ipyai` prints the active model, think level, search level, code theme, config path, and system prompt path
- `%ipyai model ...`, `%ipyai think ...`, `%ipyai search ...`, `%ipyai code_theme ...` change the current session only
- `%ipyai reset` deletes AI prompt history for the current IPython session and resets the code-context baseline

## Tools

To expose a function from the active IPython namespace as a tool for the current conversation, reference it as `&\`name\`` in the prompt:

```python
def weather(city): return f"Sunny in {city}"

`use &`weather` to answer the question about Brisbane
```

## Output Rendering

Responses are streamed directly to the terminal during generation. After streaming completes:

- the stored response remains the original full `lisette` output
- the visible terminal output is re-rendered with `rich.Markdown`
- tool call detail blocks are compacted to a short single-line form such as `🔧 f(x=1) => 2`

## Configuration

On first load, `ipyai` creates two files under the XDG config directory:

- `~/.config/ipyai/config.json`
- `~/.config/ipyai/sysp.txt`

`config.json` currently supports:

```json
{
  "model": "claude-sonnet-4-6",
  "think": "l",
  "search": "l",
  "code_theme": "monokai"
}
```

Notes:

- `model` defaults from `IPYAI_MODEL` if that environment variable is set when the config file is first created
- `think` and `search` must be one of `l`, `m`, or `h`
- `code_theme` is passed to Rich for fenced and inline code styling

`sysp.txt` is used as the system prompt passed to `lisette.Chat`.

## Development

See [DEV.md](DEV.md) for project layout, architecture, persistence details, and development workflow.

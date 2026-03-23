---
name: ipyai-help
allowed-tools: is_ipyaiskill_loaded
description: Explains ipyai usage, commands, configuration, and features. Use when the user asks how to use ipyai, what commands are available, or needs help with ipyai configuration.
---

# ipyai Help

ipyai is an IPython extension that turns any input starting with `.` into an AI prompt.

## Quick Start

Load the extension:

```python
%load_ext ipyai
```

Then type a period followed by your prompt:

```python
.write a haiku about sqlite
```

## Commands

- `%ipyai` — show current settings (model, think level, search level, code theme, log flag, config paths)
- `%ipyai model <name>` — change model for this session (e.g. `%ipyai model claude-sonnet-4-6`)
- `%ipyai think <level>` — set thinking level: `l`, `m`, or `h`
- `%ipyai search <level>` — set search level: `l`, `m`, or `h`
- `%ipyai code_theme <name>` — set Rich code theme (e.g. `monokai`, `github-dark`)
- `%ipyai log_exact <bool>` — enable/disable exact prompt/response logging
- `%ipyai save` — save current session (code + AI history) to startup file
- `%ipyai reset` — clear AI prompt history for current session

## Tools

Expose a function as a tool by referencing it with `&`name`` in your prompt:

```python
def weather(city): return f"Sunny in {city}"

. use `use &`weather` to check Brisbane`weather` to check Brisbane
```

With `#|eval: true` as the first line, the code is run when the skill is loaded. For example, after loading this, you can run `is_ipyaiskill_loaded()`:

```python
#|eval: true
def is_ipyaiskill_loaded():
    "A test tool for the ipyai skill"
    return True
```

## Notes

Any IPython cell containing only a string literal is treated as a "note". Notes appear in AI context as `<note>` blocks rather than `<code>` blocks, and are saved as markdown cells in the startup notebook.

```python
"This is a note that provides context to the AI"
```

## Configuration Files

- `~/.config/ipyai/config.json` — model, think, search, code_theme, log_exact
- `~/.config/ipyai/sysp.txt` — system prompt
- `~/.config/ipyai/startup.ipynb` — saved session snapshot
- `~/.config/ipyai/exact-log.jsonl` — raw prompt/response log (when enabled)

## Multiline Prompts

Paste multiple lines or use backslash continuation:

```python
. summarize this module:
focus on state management
and persistence behavior
```

```python
. draft a migration plan \
with risks and rollback steps
```

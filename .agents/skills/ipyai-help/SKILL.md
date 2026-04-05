---
name: ipyclaude-help
allowed-tools: is_ipyaiskill_loaded
description: Explains ipyclaude usage, commands, configuration, and features. Use when the user asks how to use ipyclaude, what commands are available, or needs help with ipyclaude configuration.
---

# ipyclaude Help

ipyclaude is an IPython extension that turns any input starting with `.` into an AI prompt.

## Quick Start

Load the extension:

```python
%load_ext ipyclaude
```

Then type a period followed by your prompt:

```python
.write a haiku about sqlite
```

## Commands

- `%ipyclaude` — show current settings (model, think level, search level, code theme, log flag, config paths)
- `%ipyclaude model <name>` — change model for this session (e.g. `%ipyclaude model claude-sonnet-4-6`)
- `%ipyclaude think <level>` — set thinking level: `l`, `m`, or `h`
- `%ipyclaude search <level>` — set search level: `l`, `m`, or `h`
- `%ipyclaude code_theme <name>` — set Rich code theme (e.g. `monokai`, `github-dark`)
- `%ipyclaude log_exact <bool>` — enable/disable exact prompt/response logging
- `%ipyclaude save` — save current session (code + AI history) to startup file
- `%ipyclaude reset` — clear AI prompt history for current session

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
    "A test tool for the ipyclaude skill"
    return True
```

## Notes

Any IPython cell containing only a string literal is treated as a "note". Notes appear in AI context as `<note>` blocks rather than `<code>` blocks, and are saved as markdown cells in the startup notebook.

```python
"This is a note that provides context to the AI"
```

## Configuration Files

- `~/.config/ipyclaude/config.json` — model, think, search, code_theme, log_exact
- `~/.config/ipyclaude/sysp.txt` — system prompt
- `~/.config/ipyclaude/startup.ipynb` — saved session snapshot
- `~/.config/ipyclaude/exact-log.jsonl` — raw prompt/response log (when enabled)

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

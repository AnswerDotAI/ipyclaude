# ipyai

Minimal IPython extension PoC for backtick-triggered AI prompts.

## Status

This extension currently does five things:

- any input whose first character is `` ` `` is rewritten to `%%ipyai`
- `%%ipyai` sends the prompt to `lisette.Chat`
- model/search/think defaults are loaded from an XDG config file
- the system prompt is loaded from an XDG `sysp.txt` file
- final terminal output is re-rendered with `rich.Markdown`
- `&\`tool\`` makes a callable in the active IPython namespace available for tool calling
- each prompt is wrapped in an XML-aware prompt template that prepends code run since the last AI prompt, including `Out[...]` history when available
- prompts and responses are stored in an `ai_prompts` table in IPython's history sqlite DB, keyed by IPython session id and prompt line

It does **not** yet do richer command summarization or stdout/stderr capture beyond what IPython already keeps in output history.

## Usage

Install the package, then load the extension:

```python
%load_ext ipyai
```

The default settings come from the XDG config file. On first load, `ipyai` creates `~/.config/ipyai/config.json` (or the path under `XDG_CONFIG_HOME`), seeding the model from `IPYAI_MODEL` if that env var is set. You can inspect or change the active runtime values with:

```python
%ipyai
%ipyai model claude-sonnet-4-5
%ipyai think m
%ipyai search h
%ipyai code_theme github-dark
%ipyai reset
```

The config file looks like:

```json
{
  "model": "claude-sonnet-4-6",
  "think": "l",
  "search": "l",
  "code_theme": "monokai"
}
```

On first load, `ipyai` also creates `~/.config/ipyai/sysp.txt` if missing and uses it as the system prompt. The default prompt explains that:

- it is running inside IPython
- responses may use Markdown, which will be formatted by Rich in the terminal
- `&\`tool\`` references may correspond to callable tools from the active IPython namespace
- prompts may include `<context>` and `<user-request>` XML blocks

Only the **leading** backtick is special. There is no closing delimiter.

To expose a function as a tool for the current prompt, reference it as `&\`name\`` inside the prompt text:

```python
def weather(city): return f"sunny in {city}"

`use &`weather` to answer what the weather is in Brisbane
```

Single-line prompt:

```python
`write a haiku about sqlite
```

Multi-line paste works because the whole pasted cell is rewritten before execution:

```python
`summarize this codebase:
focus on how prompts should be persisted
and what IPython hooks are relevant
```

For terminal entry, regular `Enter` executes the prompt. If you want to keep typing, end the line with a trailing backslash before pressing `Enter`:

```python
`draft a migration plan for AI prompt history \
including schema ideas and failure cases
```

Each AI prompt automatically rebuilds chat history from the current session's `ai_prompts` rows. Every user message is reconstructed dynamically from:

- the original prompt text
- an XML context block containing all non-`ipyai` code run since the previous AI prompt
- any available IPython output-history entries for those code cells

The latest prompt and response are stored in `_ai_last_prompt` and `_ai_last_response`. `%ipyai reset` deletes only the AI prompt history for the current IPython session and moves the code-context baseline forward to that reset point.

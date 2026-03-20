# Release notes

<!-- do not remove -->

## 0.0.7

### Bugs Squashed

- use exact match for git repo root session fallback and remove `once=True` from patches ([#19](https://github.com/AnswerDotAI/ipyai/issues/19))


## 0.0.6

### New Features

- Add CLI entry point, session persistence/resume, skill eval blocks, and ipythonng output history integration ([#18](https://github.com/AnswerDotAI/ipyai/issues/18))
- Add session persistence: store CWD in remark, list/resume sessions via -r flag and %ai sessions command ([#17](https://github.com/AnswerDotAI/ipyai/issues/17))
- Auto-run eval:true code blocks when loading skills ([#16](https://github.com/AnswerDotAI/ipyai/issues/16))
- Unify tool discovery from skills, notes, and tool responses via `_tool_refs`(); add `_parse_frontmatter`/`_allowed_tools` helpers and `completion_model` config ([#15](https://github.com/AnswerDotAI/ipyai/issues/15))
- Add AI inline completion (Alt-.), Alt-Up/Down history jump, Alt-Shift-Up/Down block cycling, and `completion_model` config ([#14](https://github.com/AnswerDotAI/ipyai/issues/14))
- Add Alt-Shift-Up/Down cycling through code blocks ([#13](https://github.com/AnswerDotAI/ipyai/issues/13))
- Add keybindings for code block insertion, and migrate from backtick/JSON to dot-prompt/ipynb format ([#12](https://github.com/AnswerDotAI/ipyai/issues/12))
- Switch prompt prefix from backtick to period, add skills discovery/loading, and strip thinking emojis from display ([#11](https://github.com/AnswerDotAI/ipyai/issues/11))
- Add skills discovery system with SKILL.md parsing, XML formatting, and `load_skill` tool integration ([#10](https://github.com/AnswerDotAI/ipyai/issues/10))
- Convert startup format from JSON events to notebook format with note/markdown cell support ([#9](https://github.com/AnswerDotAI/ipyai/issues/9))
- capture stdout and stderr ([#6](https://github.com/AnswerDotAI/ipyai/pull/6)), thanks to [@RensDimmendaal](https://github.com/RensDimmendaal)

### Bugs Squashed

- Refactor magic line handler to validate args only for known status attrs ([#7](https://github.com/AnswerDotAI/ipyai/pull/7)), thanks to [@RensDimmendaal](https://github.com/RensDimmendaal)
- fix ai response line overflow ([#5](https://github.com/AnswerDotAI/ipyai/pull/5)), thanks to [@RensDimmendaal](https://github.com/RensDimmendaal)


## 0.0.5

### New Features

- Add output-history suppression during AI streaming ([#8](https://github.com/AnswerDotAI/ipyai/issues/8))


## 0.0.4

### New Features

- Support async tools ([#4](https://github.com/AnswerDotAI/ipyai/issues/4))


## 0.0.3

### New Features

- Use rich live ([#2](https://github.com/AnswerDotAI/ipyai/issues/2))
- Add startup snapshot save/replay, exact prompt/response logging, and schema auto-recreation ([#1](https://github.com/AnswerDotAI/ipyai/issues/1))

### Bugs Squashed

- Session restore does not inject history correctly ([#3](https://github.com/AnswerDotAI/ipyai/issues/3))


## 0.0.1

- Initial `ipyai` IPython extension with backtick prompts, prompt history, tool calls, and rich terminal rendering


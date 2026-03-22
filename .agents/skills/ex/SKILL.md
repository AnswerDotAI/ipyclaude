---
name: ex
description: Guide for using the ex() tool to view and edit files. Use when the user asks about ex commands, ex syntax, or needs help with file editing via ex.
---
# ex() Guide

`ex(path, cmds='', sw=4, linenums=False, as_dict=False)`

Runs ex on a file. `x` is always appended — don't add `wq`/`x`. Multiline `cmds` work (heredoc). Use `a` on a non-existent path to create files.

## Parameters

- **`linenums=True`** — appends `%#`; use *with* edit commands to see the result
- **`sw`** — shiftwidth for `>`/`<` (default 4)
- **`as_dict`** — returns `{'success': ...}` or `{'error': ...}`

## Examples

```python
ex('f.py', linenums=True)              # view with line numbers
ex('f.py', cmds='10,20p')              # view a range
ex('f.py', cmds='10,20#')              # view a range with line numbers

ex('f.py', cmds='5s/old/new/\n4,6#')     # substitute on line 5, then check
ex('f.py', cmds='%s/old/new/g', linenums=True)        # whole-file substitute; see result immediately
ex('f.py', cmds='10,50g/DEBUG/d')     # delete matching lines in a range

ex('f.py', cmds='5,10>\n4,11#')       # indent, then check surrounding lines
ex('f.py', cmds='5,10<\n4,11#', sw=2)    # dedent, then check surrounding lines

ex('f.py', cmds='1,3co10\n10,14#')      # copy lines 1-3 after line 10, then check
ex('f.py', cmds='5,7m20\n19,23#')       # move lines 5-7 after line 20, then check

ex('f.py', cmds='''5a
    x = 1
.
4,7#''')

ex('new.py', cmds='''a
def hello():
    print("hi")
.''', linenums=True)
```

## Gotchas

- **Newlines in `s//`** — use `\r` to insert, `\n` to match. Use `\\n` to insert a literal `\n`.
- **Line numbers shift** mid-command after inserts/deletes.
- **Alternate `s` delimiters** — `#`, `@`, `+`, `;` all work (e.g. `s#/old/path#/new/path#`). `|` does not (it's the ex command separator).
- **Brackets in patterns** — escape with `\[` and `\]` in the match side. Unescaped in the replacement.
- **Always end `a`/`i` with `.`** — forgetting the `.` terminator is silent and destructive: ex swallows all remaining commands (including `x`) as input text. Nothing gets saved, no error reported.
- **`linenums=True` dumps the whole file** — on large files, skip it and check a range with `N,M#` instead.
- **`a` appends *after* the line** — if the target line ends with a closing delimiter (e.g. `"""`), the new text lands outside it. Use `s` to edit such lines in place.

## Workflow Tip

`grep -n` / `rg -n` line numbers match ex addressing, so you can find-then-fix:

```python
bash("rg -n 'pattern' file.py")   # find line numbers
ex('file.py', cmds='14s/old/new/\n13,15#')  # edit by line number, then check
```

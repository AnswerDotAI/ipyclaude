---
name: sed
description: Guide for using the sed() tool to read, search, and transform files. Use when the user asks about sed commands, sed syntax, or needs help with text processing via sed.
---
# sed() Guide

`sed(path, cmds, inplace=False, quiet=False, linenums=False, as_dict=False)`

Runs sed on a file. Non-destructive by default (prints to stdout). Use `inplace=True` to modify the file.

## Parameters

- **`quiet=True`** — like `sed -n`; essential when using `p`, otherwise matched lines are doubled
- **`inplace=True`** — like `sed -i ''`; modifies file, returns nothing
- **`as_dict`** — returns `{'success': ...}` or `{'error': ...}`
- **`linenums=True`** — numbers output lines; with `inplace`, shows the modified file with line numbers. Cannot combine with `quiet`.

## Examples

```python
sed('f.py', '3,5p', quiet=True)              # read a range
sed('f.py', '/start/,/end/p', quiet=True)    # read between patterns
sed('f.py', 's/old/new/g')                   # preview substitution
sed('f.py', 's/old/new/g', inplace=True)     # apply it
sed('f.py', 's/old/new/g', inplace=True, linenums=True)  # apply and see result
sed('f.py', '/DEBUG/d', inplace=True)        # delete matching lines
sed('f.py', r'3i\
NEW LINE')                                   # insert before line 3
sed('f.py', r's/\(.*\)/[\1]/')              # capture groups (BRE: escaped parens)
```

## Gotchas

- **`quiet=True` for reading** — without it, `p` duplicates matched lines.
- **Preview then apply** — run without `inplace` first, then re-run with `inplace=True`.
- **BRE by default** — parens must be escaped: `\(…\)`, `\1`.
- **macOS sed** — `i`/`a`/`c` require backslash-newline; `=` takes only a single address, not a range.
- **`|` works as delimiter** — unlike ex, `s|old|new|` is fine.

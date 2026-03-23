---
name: exhash
description: Guide for using the exhash verified line-addressed editor. Use when the user asks about exhash commands, exhash syntax, lnhash addressing, or needs help with hash-verified text editing.
allowed-tools: exhash_file,lnhashview_file
---

```python
#|eval: true
from exhash import exhash, exhash_file, lnhashview, lnhashview_file, lnhash, line_hash
allow('lnhashview', 'lnhashview_file', 'exhash', 'exhash_file', 'lnhash')
```
# exhash() Guide

`exhash(text, cmds, sw=4)` — Verified line-addressed editor. Like `ex`, but every line address includes a 4-char content hash that is verified before each command executes. Written in Rust with Python bindings.

**Purpose:** Safe surgical text editing, primarily for LLMs, where the hash prevents edits from landing on the wrong line due to stale line numbers.

## Core Functions

| Function | Purpose |
|---|---|
| `lnhashview(text)` | View text with `lineno\|hash\|  content` for each line |
| `lnhash(lineno, line)` | Get `lineno\|hash\|` address for one line |
| `line_hash(line)` | Get just the 4-char hex hash for a line |
| `exhash(text, cmds, sw=4)` | Apply commands to text, return result dict |
| `exhash_result(results)` | Format modified lines from result dicts |
| `lnhashview_file(path, start=None, end=None)` | View file contents with `lineno\|hash\|  content`, optionally filtered by 1-based `start`/`end` |
| `exhash_file(path, cmds, sw=4, inplace=False)` | Like `exhash` but reads from file; with `inplace=True` writes back atomically on success |

## Return Value

`exhash()` returns a result object with:
- **`lines`** — list of output lines
- **`hashes`** — lnhash for each output line
- **`modified`** — 1-based line numbers of modified/added lines
- **`deleted`** — 1-based line numbers of removed lines (in original)

`exhash_file()` returns the same result object. With `inplace=True`, the result is auto-displayed as a diff. Otherwise, use `str()` or `print()` to see the diff:

```
1|a020|  def greet(name):
-2|c748|      msg = "Hi, " + name
+2|756b|      msg = "Hey, " + name
 3|e005|      print(msg)
```

## Addressing

| Form | Meaning | Example |
|---|---|---|
| `N\|hash\|cmd` | Single line | `3\|0b26\|s/x/y/` |
| `N\|h1\|,M\|h2\|cmd` | Range | `3\|0b26\|,5\|09b9\|d` |
| `$cmd` | Last line | `$s/None/0/` |
| `%cmd` | Whole file (= `1,$`) | `%s/old/new/g` |
| `0\|0000\|` | Before line 1 (only `a`/`i`) | `0\|0000\|a\nfirst line` |

## Commands

| Command | Description | Example |
|---|---|---|
| `s/pat/rep/[flags]` | Substitute. Rust regex. `$1` for captures. Flags: `g`=all, `i`=case-insensitive. Any non-alnum delimiter: `s@pat@rep@` | `3\|0b26\|s/(\w+) = (\d+)/$1 = int($2)/` |
| `y/src/dst/` | Transliterate chars. Any non-alnum delimiter: `y@src@dst@` | `3\|0b26\|y/xyz/XYZ/` |
| `d` | Delete line(s) | `3\|0b26\|d` |
| `a` | Append text after line (remaining lines are the text block) | `6\|6e37\|a\n    new line` |
| `i` | Insert text before line | `1\|2eda\|i\n# comment` |
| `c` | Change/replace line(s) | `10\|0d04\|c\n    return False` |
| `j` | Join lines (with range, joins all) | `3\|0b26\|,5\|09b9\|j` |
| `m dest` | Move after dest address | `9\|259d\|m2\|733a\|` |
| `t dest` | Copy after dest address | `2\|733a\|t9\|259d\|` |
| `>[n]` | Indent n levels (`sw` spaces each, default 1) | `3\|0b26\|,5\|09b9\|>` |
| `<[n]` | Dedent n levels | `3\|0b26\|,5\|09b9\|<2` |
| `sort` | Sort lines alphabetically | `3\|0b26\|,5\|09b9\|sort` |
| `p` | Print (include in output without changing) | `3\|0b26\|p` |
| `g/pat/cmd` | Global: run cmd on matching lines. Any non-alnum delimiter: `g@pat@cmd` | `%g/print/d` |
| `g!/pat/cmd` or `v/pat/cmd` | Inverted global. Any non-alnum delimiter works | `%v/def/d` |

## Workflow

```python
# 1. View the text with hashed addresses
lnhashview(text)
# => ['1|2eda|  def hello():', '2|733a|      print("hello")', ...]

# 2. Edit using those addresses
result = exhash(text, ["2|733a|s/hello/hi/"])
# => {'lines': [...], 'hashes': [...], 'modified': [2], 'deleted': []}
# 3. Use the result for further edits
new_text = '\n'.join(result['lines'])
```

## File-based Functions

`lnhashview_file` and `exhash_file` are convenience wrappers for working directly with files:

```python
# View a file with hashed addresses
lnhashview_file('/tmp/myfile.py')
# => ['1|a020|  def greet(name):', '2|944e|      msg = ...', ...]

# View just lines 10-20
lnhashview_file('/tmp/myfile.py', 10, 20)

# Edit without modifying the file (dry run)
result = exhash_file('/tmp/myfile.py', ["2|944e|s/Hello/Hi/"])

# Edit and write back atomically
result = exhash_file('/tmp/myfile.py', ["2|944e|s/Hello/Hi/"], inplace=True)
```

**Atomic writes:** With `inplace=True`, the file is only written on success. If a hash verification fails, the file is left untouched.

**Tip:** Use `rg -n` or `grep -n` to find the line numbers you need, then pass that range to `lnhashview_file` to get the hashed addresses for editing.

## Multi-command Batches

Pass multiple commands as a list. Each command's hashes are verified against the **current state** (after earlier commands have executed):

```python
# After deleting line 3, original line 4 shifts to line 3.
# Use the NEW line number but the content hash stays the same.
exhash(text, ["3|0b26|d", "3|10d5|s/2/200/"])
```

## Hash Verification

Wrong or stale hashes fail loudly:
```
ValueError: stale lnhash at line 2: expected dead, got d1ac
```

This is the core safety feature — no silent mis-edits.

## Gotchas

- **Custom delimiters:** `s`, `y`, `g`, `g!`, and `v` all accept any non-alphanumeric char as delimiter instead of `/`, e.g. `s@pat@rep@`, `g@pat@cmd`. Each command in a combo picks its own delimiter independently: `g@a/b@s/old/new/`.
- **Literal newlines in pattern/replacement** are supported in `s` — this joins or splits lines as needed.
- **Regex syntax is Rust's** — use `$1`, `$2` (not `\1`) for capture groups in replacements.
- **`g/pat/cmd` needs a range prefix** — use `%g/pat/cmd` for whole-file, or `N|h|,M|h|g/pat/cmd` for a range. Bare `g/pat/cmd` without an address fails.
- **Multi-line `a`/`i`/`c`** — use `\n` in the command string to add multiple lines. No `.` terminator needed.
- **Line numbers shift in batches** — after a delete or insert, subsequent commands must use updated line numbers (but the content hash stays the same, which is what gets verified).
- **`%` and `$` don't use hashes** — these special addresses bypass hash verification since they're unambiguous.

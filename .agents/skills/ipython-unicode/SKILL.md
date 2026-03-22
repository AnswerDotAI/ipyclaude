---
name: ipython-unicode
description: Explains IPython's LaTeX/Unicode tab-completion feature — how to type \alpha, \vec, \dot, font families like \bbA, \bfA, \scrA, superscripts, subscripts, Greek letters, and math symbols as valid Python identifiers. Use when the user asks about unicode identifiers, latex symbols, tab completion for symbols, or typing Greek/math characters in IPython.
---

# IPython Unicode / LaTeX Tab Completion

IPython supports LaTeX-style tab completion: type a `\name` sequence and press **Tab** to insert the corresponding Unicode character directly into your code as a valid Python identifier.

## How It Works

```python
F\vec<Tab>    →  F⃗
v\dot<Tab>    →  v̇
x\hat<Tab>    →  x̂
\alpha<Tab>   →  α
\mu<Tab>      →  μ
```

The resulting characters are valid Python identifiers — you can use them as variable names, function parameters, etc.

## Combining Accents (place AFTER the variable name)

| LaTeX                    | Unicode | Usage example             |
|--------------------------|---------|---------------------------|
| `\vec`                   | ⃗       | `F⃗` force vector          |
| `\dot`                   | ̇       | `ẋ` velocity (dx/dt)      |
| `\ddot`                  | ̈       | `ẍ` acceleration          |
| `\dddot`                 | ⃛       | third derivative           |
| `\ddddot`                | ⃜       | fourth derivative          |
| `\hat`                   | ̂       | `x̂` unit vector/estimate  |
| `\bar`                   | ̄       | `x̄` mean value            |
| `\tilde`                 | ̃       | `x̃`                       |
| `\acute`                 | ́       | `x́`                       |
| `\grave`                 | ̀       | `x̀`                       |
| `\breve`                 | ̆       | `x̆`                       |
| `\check`                 | ̌       | `x̌`                       |
| `\overbar`               | ̅       | `x̅`                       |
| `\underbar`              | ̲       | `x̲`                       |
| `\not`                   | ̸       | `x̸` (negation overlay)    |
| `\overleftarrow`         | ⃖       |                           |
| `\overleftrightarrow`    | ⃡       |                           |

## Greek Letters

All standard Greek letters are supported:

**Lowercase:** `\alpha` α, `\beta` β, `\gamma` γ, `\delta` δ, `\epsilon` ε, `\varepsilon` ε, `\zeta` ζ, `\eta` η, `\theta` θ, `\vartheta` ϑ, `\iota` ι, `\kappa` κ, `\lambda` λ, `\mu` μ, `\nu` ν, `\xi` ξ, `\pi` π, `\varpi` ϖ, `\rho` ρ, `\varrho` ϱ, `\sigma` σ, `\varsigma` ς, `\tau` τ, `\upsilon` υ, `\phi` ϕ, `\varphi` φ, `\chi` χ, `\psi` ψ, `\omega` ω

**Uppercase:** `\Gamma` Γ, `\Delta` Δ, `\Theta` Θ, `\Lambda` Λ, `\Xi` Ξ, `\Pi` Π, `\Sigma` Σ, `\Upsilon` Υ, `\Phi` Φ, `\Psi` Ψ, `\Omega` Ω

## Superscripts `\^` and Subscripts `\_`

```python
\^a → ᵃ   \^b → ᵇ  …  \^z → ᶻ   (also A–Z and some Greek: \^alpha → ᵅ)
\_a → ₐ   \_e → ₑ  …  \_n → ₙ   (also Greek: \_beta → ᵦ, \_rho → ᵨ)
\_0 → ₀   \_1 → ₁  …  \_9 → ₉
```

## Math / Physics Symbols

```
\hbar → ħ       \hslash → ℏ    \planck → ℎ    \euler → ℯ
\ell → ℓ        \wp → ℘        \Re → ℜ        \Im → ℑ
\aleph → ℵ      \beth → ℶ      \gimel → ℷ     \daleth → ℸ
\ohm → Ω        \Angstrom → Å
```

## Font Family Alphabets (~1000 symbols)

Each covers A–Z and a–z. Type e.g. `\bfA<Tab>` to get 𝐀:

| Prefix      | Style              | Sample      |
|-------------|--------------------|-------------|
| `\bbA`      | Blackboard bold    | 𝔸𝔹ℂ𝔻𝔼      |
| `\bfA`      | Bold               | 𝐀𝐁𝐂𝐃𝐄      |
| `\biA`      | Bold italic        | 𝑨𝑩𝑪𝑫𝑬      |
| `\itA`      | Italic             | 𝐴𝐵𝐶𝐷𝐸      |
| `\scrA`     | Script             | 𝒜𝒞𝒟…       |
| `\bscrA`    | Bold script        | 𝓐𝓑𝓒𝓓𝓔      |
| `\frakA`    | Fraktur            | 𝔄𝔅ℭ𝔇𝔈      |
| `\bfrakA`   | Bold fraktur       | 𝕬𝕭𝕮𝕯𝕰      |
| `\sansA`    | Sans-serif         | 𝖠𝖡𝖢𝖣𝖤      |
| `\bsansA`   | Bold sans          | 𝗔𝗕𝗖𝗗𝗘      |
| `\isansA`   | Italic sans        | 𝘈𝘉𝘊𝘋𝘌      |
| `\bisansA`  | Bold italic sans   | 𝘼𝘽𝘾𝘿𝙀      |
| `\ttA`      | Typewriter (mono)  | 𝙰𝙱𝙲𝙳𝙴      |

## Special Latin / IPA

```
\ss → ß   \ae → æ   \AE → Æ   \aa → å   \AA → Å
\OE → Œ   \eth → ð  \schwa → ə  \DJ → Đ  \NG → Ŋ
```

## Physics / Math Variable Examples

```python
# All valid Python variable names!
α  = 0.01                    # \alpha
μ, σ = 0.0, 1.0              # \mu, \sigma
λ  = 1e-3                    # \lambda
ħ  = 1.055e-34               # \hbar
F⃗  = [1, 0, 0]               # F then \vec<Tab>
ẋ  = 5.0                     # x then \dot<Tab>
ẍ  = 9.8                     # x then \ddot<Tab>
x̄  = sum(data)/len(data)     # x then \bar<Tab>
𝐀  = np.eye(3)               # \bfA (bold matrix)
𝔽₂ = GF(2)                   # \bbF then \_2<Tab>
```

## Browsing All Symbols

```python
from IPython.core.latex_symbols import latex_symbols
len(latex_symbols)  # 1290

# Search for what you need
{k: v for k, v in latex_symbols.items() if 'arrow' in k}
{k: v for k, v in latex_symbols.items() if k.startswith('\\bb')}
```

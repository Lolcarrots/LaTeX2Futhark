from __future__ import annotations
import re
from typing import Optional, Tuple, List

_CMD_RE = re.compile(r"\\[A-Za-z]+")

def _find_matching_brace(s: str, i: int) -> int:
    assert i < len(s) and s[i] == "{", (i, s[max(0,i-10):i+10])
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j
    return -1

def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i].isspace():
        i += 1
    return i

def _read_command_at(s: str, i: int) -> Optional[Tuple[str, int]]:
    m = _CMD_RE.match(s, i)
    if m:
        return (m.group(0), m.end())
    return None

FUNCS_NEED_PARENS = {
    r"\sin", r"\cos", r"\tan",
    r"\sinh", r"\cosh", r"\tanh",
    r"\arcsin", r"\arccos", r"\arctan",
    r"\exp", r"\ln", r"\log",
    r"\min", r"\max",
    r"\det",
}

MATRIX_ENVS = {"pmatrix", "bmatrix", "matrix"}

ACCENT_ALIASES = {
    r"\overline": r"\bar",
    r"\widehat": r"\hat",
    r"\widetilde": r"\tilde",
    r"\overrightarrow": r"\vec",
}

_FRAC_CMD = re.compile(r'\\(dfrac|tfrac)\b')

_D_BEFORE_VAR = re.compile(r'\bd(?=(?:\\[A-Za-z]+|[A-Za-z]|\{))')

def _strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)

def _normalize_operatorname_det(text: str) -> str:
    return re.sub(r"\\operatorname\s*\{\s*det\s*\}", r"\\det", text)

def _normalize_mod(text: str) -> str:
    return re.sub(r"\\bmod(?![A-Za-z])", r"\\operatorname{mod}", text)

def _normalize_differential(text: str) -> str:
    return re.sub(r"\\mathrm\s*\{\s*d\s*\}", "d", text)

def _normalize_accents(text: str) -> str:

    for src, dst in ACCENT_ALIASES.items():
        text = re.sub(rf"{re.escape(src)}\s*\{{", lambda m, d=dst: d + "{", text)

    for src, dst in ACCENT_ALIASES.items():
        text = re.sub(rf"{re.escape(src)}\s+([A-Za-z])\b", lambda m, d=dst: d + "{" + m.group(1) + "}", text)
    return text

def _wrap_matrix_env_after_det(text: str) -> str:
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        cm = _read_command_at(text, i)
        if cm:
            cmd, j = cm
            if cmd == r"\det":
                k = _skip_ws(text, j)
                if text.startswith(r"\begin{", k):
                    env_name_start = k + len(r"\begin{")
                    env_name_end = text.find("}", env_name_start)
                    if env_name_end != -1:
                        env_name = text[env_name_start:env_name_end]
                        if env_name in MATRIX_ENVS:
                            begin_pat = re.compile(rf"\\begin\{{{re.escape(env_name)}\}}")
                            end_pat = re.compile(rf"\\end\{{{re.escape(env_name)}\}}")
                            p = k
                            depth = 0
                            end_idx = -1
                            while p < n:
                                mb = begin_pat.search(text, p)
                                me = end_pat.search(text, p)
                                if me is None:
                                    break
                                if mb and mb.start() < me.start():
                                    depth += 1
                                    p = mb.end()
                                else:
                                    depth -= 1
                                    p = me.end()
                                    if depth <= 0:
                                        end_idx = p
                                        break
                            if end_idx != -1:
                                out.append(text[i:j])
                                out.append("(")
                                out.append(text[k:end_idx])
                                out.append(")")
                                i = end_idx
                                continue
            out.append(text[i:j])
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)

def _brace_to_paren_after_funcs(text: str) -> str:
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        cm = _read_command_at(text, i)
        if not cm:
            out.append(text[i])
            i += 1
            continue
        cmd, j = cm
        if cmd not in FUNCS_NEED_PARENS:
            out.append(text[i:j])
            i = j
            continue

        base_seg = ""
        k = _skip_ws(text, j)
        if cmd == r"\log" and k < n and text[k] == "_":
            k += 1
            k = _skip_ws(text, k)
            if k < n and text[k] == "{":
                rb = _find_matching_brace(text, k)
                if rb != -1:
                    base_seg = text[j:rb+1]
                    k = rb + 1
            elif k < n and re.match(r"[A-Za-z]", text[k]):
                base_seg = "_{" + text[k] + "}"
                k += 1
        else:
            k = j

        k = _skip_ws(text, k)
        if k < n and text[k] == "{":
            rb = _find_matching_brace(text, k)
            if rb != -1:
                inner = text[k+1:rb]
                out.append(text[i:j])
                out.append(base_seg)
                out.append("(" + inner + ")")
                i = rb + 1
                continue
        out.append(text[i:j])
        i = j
    return "".join(out)

def _normalize_frac_and_derivative_spacing(text: str) -> str:

    text = _FRAC_CMD.sub(r'\\frac', text)


    out = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith(r'\frac', i):

            j = i + len(r'\frac')

            while j < n and text[j].isspace(): j += 1
            if j >= n or text[j] != '{':
                out.append(text[i]); i += 1; continue

            nbeg = j

            depth, k = 0, j
            while k < n:
                if text[k] == '{': depth += 1
                elif text[k] == '}':
                    depth -= 1
                    if depth == 0: break
                k += 1
            if k >= n: out.append(text[i]); i += 1; continue
            nend = k
            num = text[nbeg+1:nend]


            k += 1
            while k < n and text[k].isspace(): k += 1
            if k >= n or text[k] != '{':
                out.append(text[i]); i += 1; continue
            dbeg = k
            depth, k = 0, k
            while k < n:
                if text[k] == '{': depth += 1
                elif text[k] == '}':
                    depth -= 1
                    if depth == 0: break
                k += 1
            if k >= n: out.append(text[i]); i += 1; continue
            dend = k
            den = text[dbeg+1:dend]


            num_stripped = num.lstrip()
            if num_stripped.startswith('d') and not num_stripped.startswith('d^'):
                num = _D_BEFORE_VAR.sub('d ', num, count=1)


            den = _D_BEFORE_VAR.sub('d ', den)

            out.append(r'\frac' + '{' + num + '}' + '{' + den + '}')
            i = dend + 1
            continue

        out.append(text[i]); i += 1

    return ''.join(out)

def _normalize_dots_to_derivatives(text: str, default_var: str = "t") -> str:
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        m = re.match(r"\\(ddddot|dddot|ddot|dot)\b", text[i:])
        if not m:
            out.append(text[i])
            i += 1
            continue
        cmd = "\\" + m.group(1)
        i += m.end()
        while i < n and text[i].isspace():
            i += 1
        arg = None
        if i < n and text[i] == "{":
            rb = _find_matching_brace(text, i)
            if rb != -1:
                arg = text[i+1:rb]
                i = rb + 1
            else:
                out.append(cmd)
                continue
        elif i < n and re.match(r"[A-Za-z]", text[i]):
            arg = text[i]
            i += 1
        else:
            out.append(cmd)
            continue
        order = {"\dot": 1, "\ddot": 2, "\dddot": 3, "\ddddot": 4}[cmd]
        if order == 1:
            rep = r"\frac{d}{d " + default_var + r"} (" + arg + r")"
        else:
            o = str(order)
            rep = r"\frac{d^{" + o + r"}}{d " + default_var + r"^{" + o + r"}} (" + arg + r")"
        out.append(rep)
    return "".join(out)

def _swallow_unary_minus_before_overleftarrow(text: str) -> str:

    text = re.sub(
        r'^\s*-\s*\\overleftarrow\s*\{([^{}]+)\}',
        r'\\vec{\1}',
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r'^\s*-\s*\\overleftarrow\s+([A-Za-z])\b',
        r'\\vec{\1}',
        text,
    )
    return text

def _normalize_overleftarrow(text: str) -> str:

    text = re.sub(
        r'\\overleftarrow\s*\{([^{}]+)\}',
        r'(-\\vec{\1})',
        text,
        flags=re.DOTALL,
    )

    text = re.sub(
        r'\\overleftarrow\s+([A-Za-z])\b',
        r'(-\\vec{\1})',
        text,
    )
    return text

def _normalize_paren_bracket_sizes(text: str) -> str:


    text = re.sub(r'\\lparen\b', '(', text)
    text = re.sub(r'\\rparen\b', ')', text)
    text = re.sub(r'\\lbrack\b', '[', text)
    text = re.sub(r'\\rbrack\b', ']', text)


    size_cmd = (
        r'(?:'
        r'big|Big|bigg|Bigg|'
        r'bigl|bigr|Bigl|Bigr|'
        r'biggl|biggr|Biggl|Biggr|'
        r'bigm|Bigm|biggm|Biggm'
        r')'
    )
    text = re.sub(rf'\\{size_cmd}\*?\s*(?=[()\[\]])', '', text)


    text = re.sub(r'\\left\s*\.', '', text)
    text = re.sub(r'\\right\s*\.', '', text)
    text = re.sub(r'\\left\s*(?=[()\[\]])', '', text)
    text = re.sub(r'\\right\s*(?=[()\[\]])', '', text)

    return text

def _normalize_matrix_wrappers(text: str) -> str:

    text = re.sub(
        r'\\left\s*\[\s*\\begin\{matrix\}',
        r'\\begin{bmatrix}',
        text
    )
    text = re.sub(
        r'\\end\{matrix\}\s*\\right\s*\]',
        r'\\end{bmatrix}',
        text
    )


    text = re.sub(
        r'\\left\s*\(\s*\\begin\{matrix\}',
        r'\\begin{pmatrix}',
        text
    )
    text = re.sub(
        r'\\end\{matrix\}\s*\\right\s*\)',
        r'\\end{pmatrix}',
        text
    )


    text = re.sub(r'\\begin\{Bmatrix\}', r'\\begin{bmatrix}', text)
    text = re.sub(r'\\end\{Bmatrix\}',   r'\\end{bmatrix}',   text)


    return text

def preprocess_latex(text: str, *, dot_dt_var: str = "t") -> str:
    if not text:
        return text
    text = _strip_comments(text)
    text = _normalize_operatorname_det(text)
    text = _normalize_mod(text)
    text = _normalize_paren_bracket_sizes(text)
    text = _normalize_matrix_wrappers(text)
    text = _normalize_frac_and_derivative_spacing(text)
    text = _normalize_differential(text)
    text = _normalize_accents(text)
    text = _normalize_dots_to_derivatives(text, default_var=dot_dt_var)
    text = _wrap_matrix_env_after_det(text)
    text = _brace_to_paren_after_funcs(text)
    text = _swallow_unary_minus_before_overleftarrow(text)
    text = _normalize_overleftarrow(text)
    return text

if __name__ == "__main__":
    samples = [
        r"\dot{x}", r"\ddot{f(t)}", r"\dddot y + \ddddot{q}",
        r"\det{A}", r"\det \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}",
        r"\sin{\alpha} + \cos{(x+y)}",
        r"\log_{a}{x^2}  + \log_b{y}",
        r"\overline{x} + \overrightarrow y + \widehat{AB} + \widetilde{f}",
        r"\mathrm{d}x + \mathrm{d}\,t",
        r"a \\bmod b + \\operatorname{det}(M)",
    ]
    for s in samples:
        print("IN:  ", s)
        print("OUT: ", preprocess_latex(s))
        print("---")
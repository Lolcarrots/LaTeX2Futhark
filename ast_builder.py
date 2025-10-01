import sys
import argparse
import re
from lark import Lark, Transformer, v_args, Tree, Token


try:
    from preprocess_latex import preprocess_latex
    HAVE_PREPROCESS = True
except Exception:
    HAVE_PREPROCESS = False


class ASTNode:
    def __repr__(self):
        return self.__str__()

class NumberNode(ASTNode):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return str(self.value)

class DerivativeNode(ASTNode):
    def __init__(self, expr, var_orders, kind="partial"):


        self.expr = expr
        self.var_orders = var_orders
        if kind not in ("partial", "total"):
            raise ValueError(f"Unknown derivative kind: {kind}")
        self.kind = kind


    def __str__(self):
        op = "\\partial" if self.kind == "partial" else "d"
        den_parts = []
        total = 0
        total_is_int = True

        for var_node, order_node in self.var_orders:
            v_s = str(var_node)
            o_val = getattr(order_node, "value", None)
            if isinstance(o_val, (int, float)):
                try:
                    o_int = int(o_val)
                    if o_int == o_val:
                        total += o_int
                        den_parts.append(f"{op} {v_s}" if o_int == 1 else f"{op} {v_s}^{o_int}")
                        continue
                except Exception:
                    pass
            total_is_int = False
            den_parts.append(f"{op} {v_s}^{{{order_node}}}")

        num = op if not total_is_int or total <= 1 else f"{op}^{total}"
        expr_s = str(self.expr)
        return f"\\frac{{{num}}}{{{' '.join(den_parts)}}} {expr_s}"

class InfinityNode(ASTNode):
    def __str__(self): return "\\infty"

class SymbolNode(ASTNode):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        if self.name == 'ℝ':
            return '\\mathbb{R}'
        if self.name == 'ℕ':
            return '\\mathbb{N}_0'
        greek = {
            'alpha','beta','gamma','delta','epsilon','zeta','eta','theta','iota','kappa','lambda','mu','nu','xi','omicron',
            'pi','rho','sigma','tau','upsilon','phi','chi','psi','omega','varepsilon','vartheta','varpi','varrho','varsigma','varphi',
            'Gamma','Delta','Theta','Lambda','Xi','Pi','Sigma','Upsilon','Phi','Psi','Omega',
        }
        if self.name in greek or self.name == 'partial':
            return f"\\{self.name}"
        return str(self.name)

class SubscriptNode(ASTNode):
    def __init__(self, base, subscript):
        self.base = base
        self.subscript = subscript
    def __str__(self):
        if isinstance(self.subscript, list):
            inner = " ".join(str(s) for s in self.subscript)
            return f"{self.base}_{{{inner}}}"
        sub_s = str(self.subscript)
        if (len(sub_s) > 1 or not sub_s.isalnum()
                or '{' in sub_s or '}' in sub_s or '^' in sub_s or '+' in sub_s):
            return f"{self.base}_{{{sub_s}}}"
        return f"{self.base}_{sub_s}"

class BinOpNode(ASTNode):
    def __init__(self, op, left, right):
        self.op, self.left, self.right = op, left, right
    def __str__(self):
        if self.op == '^':
            left_str = str(self.left)
            right_str = str(self.right)
            complex_base = isinstance(self.left, (BinOpNode, FractionNode, IntegralNode, SumNode, MatrixNode, DerivativeNode))
            neg_numeric_base = isinstance(self.left, NumberNode) and isinstance(self.left.value, (int, float)) and self.left.value < 0
            if complex_base or neg_numeric_base:
                left_str = f"({left_str})"
            if not isinstance(self.right, (NumberNode, SymbolNode, SubscriptNode)):
                right_str = f"{{{right_str}}}"
            return f"{left_str}^{right_str}"
        if self.op == '-' and isinstance(self.left, NumberNode) and getattr(self.left, 'value', None) == 0:
            right_str = str(self.right)
            if isinstance(self.right, BinOpNode) and self.right.op in ['+', '-']:
                right_str = f"({right_str})"
            return f"-{right_str}"
        op_display = self.op
        if self.op == '*':
            op_display = ' \\cdot '
        elif self.op in ['+', '-']:
            op_display = f" {self.op} "
        left_str, right_str = str(self.left), str(self.right)
        if isinstance(self.left, BinOpNode) and self.op in ['*', '/', '^']:
            left_str = f"({left_str})"
        if isinstance(self.right, BinOpNode) and self.op in ['*', '/', '^', '-']:
            right_str = f"({right_str})"
        return f"{left_str}{op_display}{right_str}"

class FractionNode(ASTNode):
    def __init__(self, num, den): self.numerator, self.denominator = num, den
    def __str__(self): return f"\\frac{{{self.numerator}}}{{{self.denominator}}}"

class SqrtNode(ASTNode):
    def __init__(self, rad, idx=None): self.radicand, self.index = rad, idx
    def __str__(self):
        if self.index is None: return f"\\sqrt{{{self.radicand}}}"
        return f"\\sqrt[{self.index}]{{{self.radicand}}}"

class FunctionCallNode(ASTNode):
    def __init__(self, name, arg, power=None, base=None):
        self.func_name = name
        if isinstance(arg, list):
            self.arguments = [a if isinstance(a, ASTNode) else NumberNode(a) for a in arg]
            self.argument = self.arguments[0] if len(self.arguments) == 1 else None
        else:
            self.arguments = [arg]
            self.argument = arg
        self.power = power
        self.base = base
    def __str__(self):
        pow_s = ""
        if self.power:
            p_val_s = str(self.power)
            if not isinstance(self.power, (NumberNode, SymbolNode, SubscriptNode)):
                p_val_s = f"{{{p_val_s}}}"
            if p_val_s == "-1" and self.func_name in ["\\sin", "\\cos", "\\tan"]:
                pow_s = "^{-1}"
            else:
                pow_s = f"^{p_val_s}"
        base_s = f"_{{{self.base}}}" if self.func_name == "\\log" and self.base else ""
        arg_s = str(self.arguments[0]) if len(self.arguments) == 1 else ", ".join(str(a) for a in self.arguments)
        return f"{self.func_name}{base_s}{pow_s}({arg_s})"

class ProductNode(ASTNode):
    def __init__(self, body, var, lower, upper):
        self.body, self.var, self.lower, self.upper = body, var, lower, upper
    def __str__(self):
        body_s = str(self.body)
        if isinstance(self.body, BinOpNode) and self.body.op in ['+', '-']:
            body_s = f"\\left( {body_s} \\right)"
        return f"\\prod_{{{self.var}={self.lower}}}^{{{self.upper}}} {body_s}"

class IntegralNode(ASTNode):
    def __init__(self, integrand, var, lower=None, upper=None):
        self.integrand, self.var, self.lower, self.upper = integrand, var, lower, upper
    def __str__(self):
        lim_s = ""
        if self.lower or self.upper:
            lim_s = f"_{{{str(self.lower) if self.lower else ''}}}^{{{str(self.upper) if self.upper else ''}}}"
        int_s = str(self.integrand)
        if isinstance(self.integrand, BinOpNode) and self.integrand.op in ['+','-']: int_s = f"\\left( {int_s} \\right)"
        return f"\\int{lim_s} {int_s} \\, d{self.var}"

class MultiIntegralNode(ASTNode):
    def __init__(self, kind, integrand, diffs, lower=None, upper=None, sub=None, sup=None):
        self.kind = kind
        self.integrand = integrand
        self.diffs = diffs
        self.lower = lower
        self.upper = upper
        self.sub = sub
        self.sup = sup
    def __str__(self):
        head_map = {"iint":"\\iint","iiint":"\\iiint","oint":"\\oint","oiint":"\\oiint","oiiint":"\\oiiint"}
        head = head_map.get(self.kind, "\\iint")
        lim = ""
        if self.lower is not None or self.upper is not None:
            l = (str(self.lower) if self.lower else "")
            u = (str(self.upper) if self.upper else "")
            lim = f"_{{{l}}}^{{{u}}}"
        elif self.sub is not None:
            lim = f"_{{{self.sub}}}"
            if self.sup is not None:
                lim += f"^{{{self.sup}}}"
        body = str(self.integrand)
        diffs = "".join([f" \\, d{str(v)}" for v in self.diffs])
        return f"{head}{lim} {body}{diffs}"

class SumNode(ASTNode):
    def __init__(self, summand, var, lower, upper):
        self.summand, self.var, self.lower, self.upper = summand, var, lower, upper
    def __str__(self):
        sum_s = str(self.summand)
        if isinstance(self.summand, BinOpNode) and self.summand.op in ['+','-']: sum_s = f"\\left( {sum_s} \\right)"
        return f"\\sum_{{{self.var}={self.lower}}}^{{{self.upper}}} {sum_s}"

class MatrixNode(ASTNode):
    def __init__(self, rows, matrix_type='matrix'):
        self.rows, self.matrix_type = rows, matrix_type
    def __str__(self):
        r_s = " \\\\ ".join([" & ".join(map(str,r)) for r in self.rows])
        return f"\\begin{{{self.matrix_type}}} {r_s} \\end{{{self.matrix_type}}}"

class GradientNode(ASTNode):
    def __init__(self, expr, vars_list):
        self.expr = expr
        self.vars_list = vars_list
    def __str__(self):
        vlist = ", ".join(str(v) for v in self.vars_list)
        expr_s = str(self.expr)
        if isinstance(self.expr, BinOpNode) and self.expr.op in ["+", "-"]:
            expr_s = f"\\left( {expr_s} \\right)"
        return f"\\nabla_{{{vlist}}} {expr_s}"

class LaplacianNode(ASTNode):
    def __init__(self, expr, vars_list, power):
        self.expr = expr
        self.vars_list = vars_list
        self.power = power
    def __str__(self):
        vlist = ", ".join(str(v) for v in self.vars_list)
        expr_s = str(self.expr)
        if isinstance(self.expr, BinOpNode) and self.expr.op in ["+", "-"]:
            expr_s = f"\\left( {expr_s} \\right)"
        return f"\\nabla^{{{self.power}}}_{{{vlist}}} {expr_s}"

class ProgramNode(ASTNode):
    def __init__(self, statements): self.statements = statements
    def __str__(self): return "\n".join(map(str, [s for s in self.statements if s]))

class AbsNode(ASTNode):
    def __init__(self, expr): self.expr = expr
    def __str__(self):
        expr_s = str(self.expr)
        if isinstance(self.expr, BinOpNode) and self.expr.op in ['+','-']:
            expr_s = f"\\left( {expr_s} \\right)"
        return f"|{expr_s}|"

class NormNode(ASTNode):
    def __init__(self, expr, order=None):
        self.expr = expr
        self.order = order
    def __str__(self):
        expr_s = str(self.expr)
        if isinstance(self.expr, BinOpNode) and self.expr.op in ['+','-']:
            expr_s = f"\\left( {expr_s} \\right)"
        if self.order is None:
            return f"\\|{expr_s}\\|"
        ord_s = str(self.order)
        if not isinstance(self.order, (NumberNode, SymbolNode, SubscriptNode)):
            ord_s = f"{{{ord_s}}}"
        return f"\\|{expr_s}\\|_{ord_s}"

class InnerProductNode(ASTNode):
    def __init__(self, args):
        self.args = args
    def __str__(self):
        inside = ", ".join(str(a) for a in self.args)
        return f"\\langle {inside} \\rangle"

class EmptySetNode(ASTNode):
    def __str__(self): return "\\emptyset"

class SetNode(ASTNode):
    def __init__(self, elements): self.elements = elements
    def __str__(self):
        inner = ", ".join(str(e) for e in self.elements)
        return "{" + inner + "}"

class SetBuilderNode(ASTNode):
    def __init__(self, bounds, predicate):
        self.bounds = bounds
        self.predicate = predicate
    def __str__(self):
        def bstr(b):
            v, d = b
            return f"{v} \\in {d}" if d is not None else f"{v}"
        left = ", ".join(bstr(b) for b in self.bounds)
        return "{" + f"{left} : {self.predicate}" + "}"

class PiecewiseNode(ASTNode):
    def __init__(self, pieces): self.pieces = pieces
    def __str__(self):
        rows = []
        for e, c in self.pieces:
            rows.append(f"{e} & {c}" if c is not None else f"{e}")
        body = " \\\\ ".join(rows)
        return "\\begin{cases} " + body + " \\end{cases}"

class RelOpNode(ASTNode):
    def __init__(self, op, left, right): self.op, self.left, self.right = op, left, right
    def __str__(self): return f"{self.left} {self.op} {self.right}"

class SetOpNode(ASTNode):
    def __init__(self, op, left, right): self.op, self.left, self.right = op, left, right
    def __str__(self): return f"{self.left} {self.op} {self.right}"

class LogicOpNode(ASTNode):
    def __init__(self, op, left, right): self.op, self.left, self.right = op, left, right
    def __str__(self): return f"{self.left} {self.op} {self.right}"

class NotNode(ASTNode):
    def __init__(self, expr): self.expr = expr
    def __str__(self): return f"\\lnot {self.expr}"

class QuantifierNode(ASTNode):
    def __init__(self, kind, bounds, body): self.kind, self.bounds, self.body = kind, bounds, body
    def __str__(self):
        q = "\\forall" if self.kind == 'forall' else "\\exists"
        parts = []
        for v, d in self.bounds:
            parts.append(f"{v} \\in {d}" if d is not None else f"{v}")
        bind = ", ".join(parts)
        return f"{q} {bind} : {self.body}"

class AccentNode(ASTNode):
    def __init__(self, kind, base): self.kind, self.base = kind, base
    def __str__(self):
        cmd_map = {"hat":"\\hat","bar":"\\bar","vec":"\\vec","tilde":"\\tilde"}
        cmd = cmd_map.get(self.kind, "\\hat")
        return f"{cmd}{{{self.base}}}"

class ModNode(ASTNode):
    def __init__(self, left, right): self.left, self.right = left, right
    def __str__(self): return f"{self.left} \\operatorname{{mod}} {self.right}"

class ArgExtremumNode(ASTNode):
    def __init__(self, kind, body, sub=None, sup=None):
        self.kind = kind
        self.body = body
        self.sub = sub
        self.sup = sup
    def __str__(self):
        head = f"\\operatorname{{{self.kind}}}"
        sub = f"_{{{self.sub}}}" if self.sub is not None else ""
        sup = f"^{{{self.sup}}}" if self.sup is not None else ""
        return f"{head}{sub}{sup} {self.body}"


grammar = r"""
%import common.WS_INLINE
%import common.NEWLINE
%ignore WS_INLINE

LEFT: "\\left"
RIGHT: "\\right"
THINSPACE: "\\,"
NEGTHINSPACE: "\\!"
MEDSPACE: "\\:"
THICKSPACE: "\\;"
QUAD: "\\quad"
QQUAD: "\\qquad"
%ignore LEFT
%ignore RIGHT
%ignore THINSPACE
%ignore NEGTHINSPACE
%ignore MEDSPACE
%ignore THICKSPACE
%ignore QUAD
%ignore QQUAD

SUM: "\\sum"
INT: "\\int"
IINT: "\\iint"
IIINT: "\\iiint"
OINT: "\\oint"
OIINT: "\\oiint"
OIIINT: "\\oiiint"
FRAC: "\\frac"
SQRT: "\\sqrt"
SIN: "\\sin"
COS: "\\cos"
TAN: "\\tan"
LOG: "\\log"
LN: "\\ln"
EXP: "\\exp"
ARCSIN: "\\arcsin"
ARCCOS: "\\arccos"
ARCTAN: "\\arctan"
SINH: "\\sinh"
COSH: "\\cosh"
TANH: "\\tanh"
DET: "\\det"
MIN: "\\min"
MAX: "\\max"
PROD: "\\prod"
BEGIN: "\\begin"
END: "\\end"
CDOT: "\\cdot"
TIMES: "\\times"
ODOT: "\\odot"
EQUIV: "\\equiv"
APPROX: "\\approx"
NEQ: "\\neq" | "\\ne"
SIM: "\\sim"
D_SYMBOL: "d"
DIFF: /d[a-zA-Z][a-zA-Z0-9]*/
BACKSLASH: "\\\\"
INFINITY: "\\infty" | "\\infinity"
AMP: "&"
COMMA: ","
PARTIAL: "\\partial"
APOSTROPHE: "'"
PRIME_CMD: "\\prime"
NABLA: "\\nabla"
LANGLE: "\\langle"
RANGLE: "\\rangle"
LVERT: "\\lvert"
RVERT: "\\rvert"
LVERTD: "\\lVert"
RVERTD: "\\rVert"
VBAR: "|"
NVBAR: "\\|"
IN: "\\in"
SUBSET: "\\subset"
SUBSETEQ: "\\subseteq"
SUPSET: "\\supset"
SUPSETEQ: "\\supseteq"
UNION: "\\cup"
INTERSECT: "\\cap"
EMPTYSET: "\\emptyset"
FORALL: "\\forall"
EXISTS: "\\exists"
LAND: "\\land"
LOR: "\\lor"
NOT: "\\lnot" | "\\neg"
IMPLIES: "\\implies"
IFF: "\\iff"
COLON: ":"
HAT: "\\hat"
BAR: "\\bar"
VEC: "\\vec"
TILDE: "\\tilde"
ARGEXTCMD.2: /\\operatorname\{(?:argmax|argmin)\}/
OPMODCMD.2: /\\operatorname\{mod\}/
GEN_CMD: /(?!\\prime|\\left|\\right|\\quad|\\qquad|\\lvert|\\rvert|\\lVert|\\rVert|\\langle|\\rangle|\\times|\\odot|\\equiv|\\approx|\\neq|\\ne|\\sim)\\[a-zA-Z]+/
CONTROL_SPACE: /\\(?=\s)/

NUMBER: /\d+(\.\d+)?([eE][-+]?\d+)?/
SYMBOL: /[a-zA-Z][a-zA-Z0-9]*/

LBRACE: "{"
RBRACE: "}"
LBRACK: "["
RBRACK: "]"
LPAR: "("
RPAR: ")"
UNDERSCORE: "_"
CIRCUMFLEX: "^"
PLUS: "+"
MINUS: "-"
STAR: "*"
SLASH: "/"
EQUAL: "="

start: expr (NEWLINE+ expr)* -> start_rule

expr: term
    | expr PLUS term                  -> add
    | expr MINUS term                 -> sub
    | expr UNION term                 -> union
    | expr INTERSECT term             -> intersect
    | expr LAND term                  -> land
    | expr LOR term                   -> lor
    | expr IMPLIES term               -> implies
    | expr IFF term                   -> iff
    | expr EQUAL term                 -> eq
    | expr IN term                    -> in_rel
    | expr EQUIV term                 -> equiv
    | expr APPROX term                -> approx
    | expr NEQ term                   -> neq
    | expr SIM term                   -> sim
    | expr SUBSETEQ term              -> subseteq
    | expr SUBSET term                -> subset
    | expr SUPSETEQ term              -> supseteq
    | expr SUPSET term                -> supset

term: factor
    | term STAR factor           -> mul
    | term SLASH factor          -> div
    | term CDOT factor           -> mul
    | term TIMES factor          -> times
    | term ODOT factor           -> odot
    | term factor_no_bar         -> mul
    | term OPMODCMD factor       -> mod
    | NOT factor                 -> not_rule

power: atom CIRCUMFLEX LBRACE (expr | prime_cmds) RBRACE -> power_rule_brace
     | atom CIRCUMFLEX atom

factor: power (LPAR arg_list RPAR)+ -> power_with_calls
      | power
      | atom
      | MINUS factor -> unary_minus

power_no_bar: atom_no_bar CIRCUMFLEX LBRACE (expr | prime_cmds) RBRACE -> power_rule_brace
            | atom_no_bar CIRCUMFLEX atom_no_bar

factor_no_bar: power_no_bar (LPAR arg_list RPAR)+ -> power_with_calls
             | power_no_bar
             | atom_no_bar

atom: NUMBER -> number_lit
    | SYMBOL LPAR arg_list RPAR -> symbol_func_rule
    | SYMBOL prime_seq LPAR arg_list RPAR -> symbol_prime_rule
    | SYMBOL -> symbol_lit
    | INFINITY -> infinity_lit
    | GEN_CMD -> command_symbol
    | subscript
    | LPAR expr RPAR -> group
    | LBRACK expr RBRACK -> group
    | derivative_expr
    | derivative_inline_expr
    | total_derivative_expr
    | total_derivative_inline_expr
    | partial_sub_op_expr
    | total_sub_op_expr
    | nabla_grad_expr
    | nabla_lap_expr
    | frac
    | sum_expr
    | prod_expr
    | integral_expr
    | multi_integral_expr
    | sqrt
    | func
    | HAT LBRACE expr RBRACE                          -> hat_rule
    | HAT atom                                        -> hat_atom_rule
    | BAR LBRACE expr RBRACE                          -> bar_rule
    | BAR atom                                        -> bar_atom_rule
    | VEC LBRACE expr RBRACE                          -> vec_rule
    | VEC atom                                        -> vec_atom_rule
    | TILDE LBRACE expr RBRACE                        -> tilde_rule
    | TILDE atom                                      -> tilde_atom_rule
    | ARGEXTCMD UNDERSCORE LBRACE expr RBRACE CIRCUMFLEX LBRACE expr RBRACE factor -> argext_both_rule
    | ARGEXTCMD UNDERSCORE LBRACE expr RBRACE factor                                -> argext_sub_rule
    | ARGEXTCMD factor                                                               -> argext_plain_rule
    | gen_func
    | styled_func
    | styled_func_brack
    | matrix
    | abs_expr
    | norm_expr
    | inner_expr
    | EMPTYSET                        -> emptyset_lit
    | cases_expr
    | set_literal
    | set_builder
    | quantifier_expr

atom_no_bar: NUMBER                         -> number_lit
           | SYMBOL LPAR arg_list RPAR      -> symbol_func_rule
           | SYMBOL prime_seq LPAR arg_list RPAR -> symbol_prime_rule
           | SYMBOL                          -> symbol_lit
           | INFINITY                        -> infinity_lit
           | GEN_CMD                         -> command_symbol
           | subscript
           | LPAR expr RPAR                  -> group
           | LBRACK expr RBRACK              -> group
           | derivative_expr
           | derivative_inline_expr
           | total_derivative_expr
           | total_derivative_inline_expr
           | partial_sub_op_expr
           | total_sub_op_expr
           | nabla_grad_expr
           | nabla_lap_expr
           | frac
           | sum_expr
           | prod_expr
           | integral_expr
           | multi_integral_expr
           | sqrt
           | func
           | gen_func
           | styled_func
           | styled_func_brack
           | matrix
           | inner_expr

prime_seq: APOSTROPHE+              -> prime_seq_rule
prime_cmds: PRIME_CMD (PRIME_CMD)* -> prime_cmd_seq_rule

derivative_expr: FRAC LBRACE partial_power RBRACE LBRACE den_list RBRACE factor -> derivative_rule
derivative_inline_expr: FRAC LBRACE partial_power factor RBRACE LBRACE den_list RBRACE -> derivative_inline_rule

partial_power: PARTIAL -> partial_power_plain
             | PARTIAL CIRCUMFLEX NUMBER -> partial_power_num
             | PARTIAL CIRCUMFLEX LBRACE expr RBRACE -> partial_power_expr

den_list: den_item+ -> den_list_rule

den_item: PARTIAL SYMBOL -> den_item_plain
        | PARTIAL SYMBOL CIRCUMFLEX NUMBER -> den_item_pow_num
        | PARTIAL SYMBOL CIRCUMFLEX LBRACE expr RBRACE -> den_item_pow_expr

total_derivative_expr: FRAC LBRACE d_power RBRACE LBRACE d_den_list RBRACE factor -> tderivative_rule
total_derivative_inline_expr: FRAC LBRACE d_power factor RBRACE LBRACE d_den_list RBRACE -> tderivative_inline_rule
                             | FRAC LBRACE DIFF RBRACE           LBRACE d_den_list RBRACE -> tderivative_inline_df_rule

d_power: D_SYMBOL -> d_power_plain
       | D_SYMBOL CIRCUMFLEX NUMBER -> d_power_num
       | D_SYMBOL CIRCUMFLEX LBRACE expr RBRACE -> d_power_expr

d_den_list: d_den_item+ -> d_den_list_rule

d_den_item: D_SYMBOL SYMBOL -> d_den_item_plain
          | D_SYMBOL SYMBOL CIRCUMFLEX NUMBER -> d_den_item_pow_num
          | D_SYMBOL SYMBOL CIRCUMFLEX LBRACE expr RBRACE -> d_den_item_pow_expr
          | DIFF -> d_den_item_fused

partial_sub_op_expr: PARTIAL UNDERSCORE SYMBOL factor -> partial_sub_op_rule1
                   | PARTIAL UNDERSCORE LBRACE sub_den_list RBRACE factor -> partial_sub_op_rule

total_sub_op_expr: D_SYMBOL UNDERSCORE SYMBOL factor -> total_sub_op_rule1
                 | D_SYMBOL UNDERSCORE LBRACE sub_den_list RBRACE factor -> total_sub_op_rule

sub_den_list: sub_den_item+ -> sub_den_list_rule

sub_den_item: SYMBOL -> sub_den_item_plain
            | SYMBOL CIRCUMFLEX NUMBER -> sub_den_item_pow_num
            | SYMBOL CIRCUMFLEX LBRACE expr RBRACE -> sub_den_item_pow_expr

nabla_grad_expr: NABLA UNDERSCORE LBRACE nabla_vars RBRACE factor -> nabla_grad_rule
nabla_lap_expr: NABLA CIRCUMFLEX NUMBER UNDERSCORE LBRACE nabla_vars RBRACE factor -> nabla_lap_rule_num
              | NABLA CIRCUMFLEX LBRACE expr RBRACE UNDERSCORE LBRACE nabla_vars RBRACE factor -> nabla_lap_rule_expr

nabla_vars: SYMBOL (COMMA SYMBOL)* -> nabla_vars_rule
subscript_comma_list: sub_den_item (COMMA sub_den_item)+ -> subscript_comma_list_rule

subscript: SYMBOL UNDERSCORE NUMBER                               -> simple_num_subscript_rule
         | SYMBOL UNDERSCORE SYMBOL                               -> simple_sym_subscript_rule
         | SYMBOL UNDERSCORE LBRACE subscript_comma_list RBRACE   -> complex_subscript_list_rule
         | SYMBOL UNDERSCORE LBRACE expr RBRACE                   -> complex_subscript_expr_rule

frac: FRAC LBRACE expr RBRACE LBRACE expr RBRACE -> fraction_rule
sum_expr: SUM UNDERSCORE LBRACE SYMBOL EQUAL expr RBRACE CIRCUMFLEX LBRACE expr RBRACE expr -> sum_rule
prod_expr: PROD UNDERSCORE LBRACE SYMBOL EQUAL expr RBRACE CIRCUMFLEX LBRACE expr RBRACE expr -> product_rule

integral_expr: INT expr diff -> indef_integral_rule
             | INT UNDERSCORE LBRACE expr RBRACE CIRCUMFLEX LBRACE expr RBRACE expr diff -> def_integral_rule

diff: D_SYMBOL SYMBOL -> d_symbol_rule
    | DIFF -> dx_token_rule

sqrt: SQRT LBRACE expr RBRACE -> simple_sqrt_rule
    | SQRT LBRACK expr RBRACK LBRACE expr RBRACE -> indexed_sqrt_rule

func: simple_func
    | power_func

styled_func: GEN_CMD LBRACE SYMBOL RBRACE LPAR arg_list RPAR           -> styled_func_rule
styled_func_brack: GEN_CMD LBRACE SYMBOL RBRACE LBRACK arg_list RBRACK -> styled_func_brack_rule

simple_func: SIN LPAR expr RPAR -> sin_rule
          | COS LPAR expr RPAR -> cos_rule
          | TAN LPAR expr RPAR -> tan_rule
          | LOG UNDERSCORE LBRACE expr RBRACE LPAR expr RPAR -> log_with_base_rule
          | LOG LPAR expr RPAR -> log_rule
          | LN LPAR expr RPAR -> ln_rule
          | EXP LPAR expr RPAR -> exp_rule
          | ARCSIN LPAR expr RPAR -> arcsin_rule
          | ARCCOS LPAR expr RPAR -> arccos_rule
          | ARCTAN LPAR expr RPAR -> arctan_rule
          | SINH LPAR expr RPAR -> sinh_rule
          | COSH LPAR expr RPAR -> cosh_rule
          | TANH LPAR expr RPAR -> tanh_rule
          | DET LPAR expr RPAR -> det_rule
          | MIN LPAR arg_list RPAR -> min_rule
          | MAX LPAR arg_list RPAR -> max_rule

power_func: SIN CIRCUMFLEX atom LPAR expr RPAR -> sin_power_rule
          | COS CIRCUMFLEX atom LPAR expr RPAR -> cos_power_rule
          | TAN CIRCUMFLEX atom LPAR expr RPAR -> tan_power_rule
          | SIN CIRCUMFLEX LBRACE expr RBRACE LPAR expr RPAR -> sin_power_rule_brace
          | COS CIRCUMFLEX LBRACE expr RBRACE LPAR expr RPAR -> cos_power_rule_brace
          | TAN CIRCUMFLEX LBRACE expr RBRACE LPAR expr RPAR -> tan_power_rule_brace

gen_func: GEN_CMD LPAR arg_list RPAR -> generic_func_rule
arg_list: expr (COMMA expr)* -> arg_list_rule

matrix: BEGIN LBRACE matrix_type RBRACE matrix_row_list END LBRACE matrix_type RBRACE -> matrix_rule
matrix_type: "pmatrix" -> pmatrix_type
          | "bmatrix" -> bmatrix_type
          | "matrix" -> matrix_type_plain

matrix_row_list: matrix_row ((BACKSLASH | CONTROL_SPACE)+ matrix_row)* -> matrix_rows_rule
matrix_row: expr (AMP expr)* -> matrix_row_rule

abs_expr: VBAR expr VBAR            -> abs_rule
        | LVERT expr RVERT          -> abs_rule

norm_expr: NVBAR expr NVBAR                                 -> norm_rule
         | LVERTD expr RVERTD                               -> norm_rule
         | NVBAR expr NVBAR UNDERSCORE NUMBER               -> norm_rule_ord_num
         | NVBAR expr NVBAR UNDERSCORE LBRACE expr RBRACE   -> norm_rule_ord_expr
         | LVERTD expr RVERTD UNDERSCORE NUMBER             -> norm_rule_ord_num
         | LVERTD expr RVERTD UNDERSCORE LBRACE expr RBRACE -> norm_rule_ord_expr

inner_expr: LANGLE arg_list RANGLE -> inner_rule

cases_expr: BEGIN LBRACE "cases" RBRACE cases_row_list END LBRACE "cases" RBRACE -> cases_rule
cases_row_list: cases_row ((BACKSLASH | CONTROL_SPACE)+ cases_row)*             -> cases_rows_rule
cases_row: expr (AMP expr)?                                                     -> cases_row_rule

set_literal: LBRACE set_elems RBRACE                                            -> set_literal_rule
           | LBRACE RBRACE                                                      -> empty_set_braces_rule
set_elems: expr (COMMA expr)*                                                   -> set_elems_rule

set_builder: LBRACE sb_bound_list (VBAR | COLON) expr RBRACE                    -> set_builder_rule
sb_bound_list: sb_bound (COMMA sb_bound)*                                       -> sb_bound_list_rule
sb_bound: SYMBOL IN expr                                                        -> sb_bound_rule

quantifier_expr: FORALL qbound_list COLON expr                                  -> forall_rule
               | EXISTS qbound_list COLON expr                                  -> exists_rule
qbound_list: qbound (COMMA qbound)*                                             -> qbound_list_rule
qbound: SYMBOL IN expr                                                          -> qbound_in_rule
      | SYMBOL                                                                  -> qbound_plain_rule

multi_integral_expr: iint_like
                   | iiint_like
                   | oint_like

iint_like: (IINT | OIINT) expr diffs2 -> multi_int2_indef_rule
         | (IINT | OIINT) UNDERSCORE LBRACE expr RBRACE expr diffs2 -> multi_int2_region_rule
         | (IINT | OIINT) UNDERSCORE LBRACE expr RBRACE CIRCUMFLEX LBRACE expr RBRACE expr diffs2 -> multi_int2_def_rule

iiint_like: (IIINT | OIIINT) expr diffs3 -> multi_int3_indef_rule
          | (IIINT | OIIINT) UNDERSCORE LBRACE expr RBRACE expr diffs3 -> multi_int3_region_rule
          | (IIINT | OIIINT) UNDERSCORE LBRACE expr RBRACE CIRCUMFLEX LBRACE expr RBRACE expr diffs3 -> multi_int3_def_rule

oint_like: OINT expr diff -> contour_indef_rule
         | OINT UNDERSCORE LBRACE expr RBRACE expr diff -> contour_region_rule
         | OINT UNDERSCORE LBRACE expr RBRACE CIRCUMFLEX LBRACE expr RBRACE expr diff -> contour_def_rule

diffs2: diff diff      -> diffs2_rule
diffs3: diff diff diff -> diffs3_rule
"""


@v_args(inline=True)
class ASTTransformer(Transformer):
    def ensure_node(self, x):
        if isinstance(x, ASTNode):
            return x
        if isinstance(x, Token):
            if x.type == "NUMBER":
                v = x.value
                return NumberNode(float(v) if '.' in v or 'e' in v.lower() else int(v))
            if x.type == "SYMBOL":
                if x.value in ("infty", "oo"):
                    return InfinityNode()
                if x.value == '\\mathbb{R}':
                    return SymbolNode('ℝ')
                return SymbolNode(x.value)
            if x.type == "INFINITY":
                return InfinityNode()
            if x.type == "matrix_type":
                return x.value
            return SymbolNode(x.value)
        if isinstance(x, (int, float)):
            return NumberNode(x)
        if isinstance(x, str):
            return SymbolNode(x)
        if isinstance(x, Tree):
            rule_name = x.data
            if hasattr(self, rule_name):
                transformer_method = getattr(self, rule_name)
                return transformer_method(*x.children)
            if x.children:
                return self.ensure_node(x.children[0])
            return SymbolNode(f"UnhandledTree:{x.data}")
        if x is None:
            return None
        if isinstance(x, list):
            return [self.ensure_node(item) for item in x]
        return SymbolNode(f"UnknownTypeInEnsure({type(x)})")

    def start_rule(self, *items):
        all_expr_nodes = []
        if not items:
            return ProgramNode([])
        first_item = items[0]
        if first_item is None:
            return ProgramNode([])
        all_expr_nodes.append(self.ensure_node(first_item))
        if len(items) > 1 and items[1] is not None:
            rest_list = items[1]
            for it in rest_list:
                if not isinstance(it, Token):
                    all_expr_nodes.append(self.ensure_node(it))
        return ProgramNode(all_expr_nodes)

    def add(self, left, _op, right): return BinOpNode('+', self.ensure_node(left), self.ensure_node(right))
    def sub(self, left, _op, right): return BinOpNode('-', self.ensure_node(left), self.ensure_node(right))

    def mul(self, *args):
        if len(args) == 3:
            left, _op_tok, right = args
        elif len(args) == 2:
            left, right = args
        else:
            raise ValueError(f"Unexpected args for mul: {args}")
        left_n = self.ensure_node(left)
        right_n = self.ensure_node(right)

        if (
            isinstance(left_n, BinOpNode) and left_n.op == '^'
            and isinstance(left_n.left, SymbolNode)
            and isinstance(right_n, FunctionCallNode)
            and isinstance(right_n.argument, ASTNode)
        ):
            f_sym = left_n.left
            call_name = right_n.func_name
            if (
                call_name == f_sym.name
                and len(right_n.arguments) == 1
                and isinstance(right_n.arguments[0], SymbolNode)
            ):
                order = self._deriv_order_from_exponent(left_n.right)
                if order is not None and order >= 1:
                    return DerivativeNode(
                        right_n,
                        [(right_n.arguments[0], NumberNode(order))],
                        kind="total",
                    )
        return BinOpNode('*', left_n, right_n)

    def _deriv_order_from_exponent(self, exp_node):
        if isinstance(exp_node, NumberNode):
            try:
                return int(exp_node.value)
            except Exception:
                return None

        def count_primes(n):
            if isinstance(n, SymbolNode) and n.name == 'prime':
                return 1
            if isinstance(n, BinOpNode) and n.op == '*':
                return count_primes(n.left) + count_primes(n.right)
            return 0
        prime_count = count_primes(exp_node)
        return prime_count if prime_count > 0 else None

    def div(self, left, _op, right): return BinOpNode('/', self.ensure_node(left), self.ensure_node(right))
    def unary_minus(self, _minus, factor_node_val):
        node = self.ensure_node(factor_node_val)
        if isinstance(node, NumberNode):
            return NumberNode(-node.value)
        return BinOpNode('-', NumberNode(0), node)
    def group(self, _lp, expr, _rp): return expr
    def power(self, left, _caret, right): return BinOpNode('^', self.ensure_node(left), self.ensure_node(right))
    def power_rule_brace(self, base, _caret, _lb, exp, _rb): return BinOpNode('^', self.ensure_node(base), self.ensure_node(exp))

    def power_with_calls(self, base, *rest):
        base_n = self.ensure_node(base)
        arglists = [al for al in rest if isinstance(al, list)]
        if not arglists:
            return base_n
        args = arglists[-1]
        if (
            isinstance(base_n, BinOpNode) and base_n.op == '^'
            and isinstance(base_n.left, SymbolNode)
            and len(args) == 1 and isinstance(args[0], SymbolNode)
        ):
            order = self._deriv_order_from_exponent(base_n.right)
            if order is not None and order >= 1:
                var = args[0]
                expr_node = FunctionCallNode(base_n.left.name, var)
                return DerivativeNode(expr_node, [(var, NumberNode(order))], kind="total")
        if isinstance(base_n, SymbolNode):
            return FunctionCallNode(base_n.name, args if len(args) > 1 else args[0])
        fname = str(base_n)
        return FunctionCallNode(fname, args if len(args) > 1 else args[0])

    def number_lit(self, n_tok): return self.ensure_node(n_tok)
    def symbol_lit(self, s_tok):
        if s_tok.value == '\\mathbb{R}':
            return SymbolNode('ℝ')
        return self.ensure_node(s_tok)
    def infinity_lit(self, _): return InfinityNode()
    def command_symbol(self, cmd_tok): return SymbolNode(cmd_tok.value.lstrip('\\'))

    def simple_num_subscript_rule(self, base, _, number): return SubscriptNode(base.value, str(self.ensure_node(number)))
    def simple_sym_subscript_rule(self, base, _, symbol): return SubscriptNode(base.value, str(self.ensure_node(symbol)))

    def subscript_comma_list_rule(self, first, *rest):
        items = [first]
        for x in rest:
            if isinstance(x, Token) and x.type == 'COMMA': continue
            items.append(x)
        return items

    def complex_subscript_list_rule(self, base, _, _lb, items, _rb):
        base_name = base.value
        seq = []
        for it in items:
            if isinstance(it, tuple) and len(it) == 2:
                v_node, o_node = it
                if isinstance(o_node, NumberNode) and getattr(o_node, 'value', 1) == 1:
                    seq.append(v_node)
                else:
                    seq.append(BinOpNode('^', v_node, o_node))
            else:
                seq.append(self.ensure_node(it))
        return SubscriptNode(base_name, seq)

    def complex_subscript_expr_rule(self, base, _, _lb, expr, _rb):
        return SubscriptNode(base.value, str(self.ensure_node(expr)))

    def fraction_rule(self, _frac, _lb1, num_expr, _rb1, _lb2, den_expr, _rb2):
        return FractionNode(self.ensure_node(num_expr), self.ensure_node(den_expr))

    def sum_rule(self, _sum, _, _lb1, index_var, _eq, lower_expr, _rb1, _cf, _lb2, upper_expr, _rb2, body_expr):
        return SumNode(self.ensure_node(body_expr), self.ensure_node(index_var), self.ensure_node(lower_expr), self.ensure_node(upper_expr))

    def product_rule(self, _prod, _, _lb1, index_var, _eq, lower_expr, _rb1, _cf, _lb2, upper_expr, _rb2, body_expr):
        return ProductNode(self.ensure_node(body_expr), self.ensure_node(index_var), self.ensure_node(lower_expr), self.ensure_node(upper_expr))

    def indef_integral_rule(self, _int, expr, diff):
        return IntegralNode(self.ensure_node(expr), self.ensure_node(diff))

    def def_integral_rule(self, _int, _, _lb1, lower, _rb1, _cf, _lb2, upper, _rb2, expr, diff):
        return IntegralNode(self.ensure_node(expr), self.ensure_node(diff), self.ensure_node(lower), self.ensure_node(upper))

    def d_symbol_rule(self, _d_tok, var_tok): return self.ensure_node(var_tok)
    def dx_token_rule(self, tok): return SymbolNode(tok.value[1:])

    def derivative_rule(self, *_c):
        (_, _lb1, _num_power, _rb1, _lb2, den_list, _rb2, factor_node) = _c
        expr_node = self.ensure_node(factor_node)
        var_orders = []
        for item in (den_list if isinstance(den_list, list) else [den_list]):
            if isinstance(item, tuple) and len(item) == 2:
                v_node, o_node = item
                var_orders.append((self.ensure_node(v_node), self.ensure_node(o_node)))
            else:
                var_orders.append((self.ensure_node(item), NumberNode(1)))
        return DerivativeNode(expr_node, var_orders, kind="partial")

    def derivative_inline_rule(self, *_c):
        (_frac, _lb1, _num_power, factor_node, _rb1, _lb2, den_list, _rb2) = _c
        expr_node = self.ensure_node(factor_node)
        var_orders = []
        for item in (den_list if isinstance(den_list, list) else [den_list]):
            if isinstance(item, tuple) and len(item) == 2:
                v_node, o_node = item
                var_orders.append((self.ensure_node(v_node), self.ensure_node(o_node)))
            else:
                var_orders.append((self.ensure_node(item), NumberNode(1)))
        return DerivativeNode(expr_node, var_orders, kind="partial")

    def partial_power_plain(self, _): return NumberNode(1)
    def partial_power_num(self, _p, _c, number_tok): return self.ensure_node(number_tok)
    def partial_power_expr(self, _p, _c, _lb, expr, _rb): return self.ensure_node(expr)

    def den_list_rule(self, *items): return list(items)
    def den_item_plain(self, _p, s): return (SymbolNode(s.value), NumberNode(1))
    def den_item_pow_num(self, _p, s, _c, n): return (SymbolNode(s.value), self.ensure_node(n))
    def den_item_pow_expr(self, _p, s, _c, _lb, e, _rb): return (SymbolNode(s.value), self.ensure_node(e))

    def tderivative_rule(self, frac_tok, lbrace1, num_power, rbrace1, lbrace2, den_list, rbrace2, factor_node):
        expr_node = self.ensure_node(factor_node)
        var_orders = []
        for item in (den_list if isinstance(den_list, list) else [den_list]):
            if isinstance(item, tuple) and len(item) == 2:
                v_node, o_node = item
                var_orders.append((self.ensure_node(v_node), self.ensure_node(o_node)))
            else:
                var_orders.append((self.ensure_node(item), NumberNode(1)))
        return DerivativeNode(expr_node, var_orders, kind="total")

    def tderivative_inline_rule(self, frac_tok, lbrace1, num_power, factor_node, rbrace1, lbrace2, den_list, rbrace2):
        expr_node = self.ensure_node(factor_node)
        var_orders = []
        for item in (den_list if isinstance(den_list, list) else [den_list]):
            if isinstance(item, tuple) and len(item) == 2:
                v_node, o_node = item
                var_orders.append((self.ensure_node(v_node), self.ensure_node(o_node)))
            else:
                var_orders.append((self.ensure_node(item), NumberNode(1)))
        return DerivativeNode(expr_node, var_orders, kind="total")

    def d_den_item_fused(self, diff_tok):

        var_name = diff_tok.value[1:]
        return (SymbolNode(var_name), NumberNode(1))

    def tderivative_inline_df_rule(self, frac_tok, _lb1, diff_tok, _rb1, _lb2, den_list, _rb2):

        expr_node = SymbolNode(diff_tok.value[1:])
        var_orders = []
        items = den_list if isinstance(den_list, list) else [den_list]
        for item in items:
            if isinstance(item, tuple) and len(item) == 2:
                v_node = self.ensure_node(item[0])
                o_node = self.ensure_node(item[1])
                var_orders.append((v_node, o_node))
            else:
                var_orders.append((self.ensure_node(item), NumberNode(1)))
        return DerivativeNode(expr_node, var_orders, kind="total")

    def d_power_plain(self, _): return NumberNode(1)
    def d_power_num(self, _d, _c, n): return self.ensure_node(n)
    def d_power_expr(self, _d, _c, _lb, expr, _rb): return self.ensure_node(expr)
    def d_den_list_rule(self, *items): return list(items)
    def d_den_item_plain(self, _d, s): return (SymbolNode(s.value), NumberNode(1))
    def d_den_item_pow_num(self, _d, s, _c, n): return (SymbolNode(s.value), self.ensure_node(n))
    def d_den_item_pow_expr(self, _d, s, _c, _lb, e, _rb): return (SymbolNode(s.value), self.ensure_node(e))

    def partial_sub_op_rule1(self, _p, _u, s, factor_node):
        return DerivativeNode(self.ensure_node(factor_node), [(SymbolNode(s.value), NumberNode(1))], kind="partial")
    def partial_sub_op_rule(self, _p, _u, _lb, sub_list, _rb, factor_node):
        expr_node = self.ensure_node(factor_node)
        var_orders = []
        items = sub_list if isinstance(sub_list, list) else [sub_list]
        for item in items:
            if isinstance(item, tuple) and len(item) == 2:
                v_node, o_node = item
                var_orders.append((self.ensure_node(v_node), self.ensure_node(o_node)))
            else:
                var_orders.append((self.ensure_node(item), NumberNode(1)))
        return DerivativeNode(expr_node, var_orders, kind="partial")

    def total_sub_op_rule1(self, _d, _u, s, factor_node):
        return DerivativeNode(self.ensure_node(factor_node), [(SymbolNode(s.value), NumberNode(1))], kind="total")
    def total_sub_op_rule(self, _d, _u, _lb, sub_list, _rb, factor_node):
        expr_node = self.ensure_node(factor_node)
        var_orders = []
        items = sub_list if isinstance(sub_list, list) else [sub_list]
        for item in items:
            if isinstance(item, tuple) and len(item) == 2:
                v_node, o_node = item
                var_orders.append((self.ensure_node(v_node), self.ensure_node(o_node)))
            else:
                var_orders.append((self.ensure_node(item), NumberNode(1)))
        return DerivativeNode(expr_node, var_orders, kind="total")

    def sub_den_list_rule(self, *items): return list(items)
    def sub_den_item_plain(self, s): return (SymbolNode(s.value), NumberNode(1))
    def sub_den_item_pow_num(self, s, _c, n): return (SymbolNode(s.value), self.ensure_node(n))
    def sub_den_item_pow_expr(self, s, _c, _lb, e, _rb): return (SymbolNode(s.value), self.ensure_node(e))

    def symbol_func_rule(self, sym_tok, _lp, args, _rp):
        args_list = args if isinstance(args, list) else [args]
        args_nodes = [self.ensure_node(a) for a in args_list]
        return FunctionCallNode(sym_tok.value, args_nodes if len(args_nodes) > 1 else args_nodes[0])

    def prime_seq_rule(self, *apostrophes): return NumberNode(len(apostrophes))
    def prime_cmd_seq_rule(self, *primes): return NumberNode(len(primes))

    def symbol_prime_rule(self, sym_tok, prime_count, _lp, args, _rp):
        order_node = self.ensure_node(prime_count)
        args_list = args if isinstance(args, list) else [args]
        args_nodes = [self.ensure_node(a) for a in args_list]
        if len(args_nodes) == 1 and isinstance(args_nodes[0], SymbolNode):
            expr_node = FunctionCallNode(sym_tok.value, args_nodes)
            return DerivativeNode(expr_node, [(args_nodes[0], order_node)], kind="total")
        return FunctionCallNode(sym_tok.value + "'" * int(getattr(order_node, "value", 1)), args_nodes)

    def nabla_vars_rule(self, first_sym, *rest):
        vars_list = [SymbolNode(first_sym.value)]
        for item in rest:
            if isinstance(item, Token) and item.type == "COMMA": continue
            if isinstance(item, Token) and item.type == "SYMBOL":
                vars_list.append(SymbolNode(item.value))
            else:
                vars_list.append(self.ensure_node(item))
        return vars_list
    def nabla_grad_rule(self, _nab, _us, _lb, vars_list, _rb, factor_node):
        vlist = vars_list if isinstance(vars_list, list) else [vars_list]
        return GradientNode(self.ensure_node(factor_node), vlist)
    def nabla_lap_rule_num(self, _nab, _cf, number_tok, _us, _lb, vars_list, _rb, factor_node):
        return LaplacianNode(self.ensure_node(factor_node), (vars_list if isinstance(vars_list, list) else [vars_list]), self.ensure_node(number_tok))
    def nabla_lap_rule_expr(self, _nab, _cf, _lb, expr, _rb, _us, _lb2, vars_list, _rb2, factor_node):
        return LaplacianNode(self.ensure_node(factor_node), (vars_list if isinstance(vars_list, list) else [vars_list]), self.ensure_node(expr))

    def simple_sqrt_rule(self, _sqrt, _lb, expr, _rb):
        return SqrtNode(self.ensure_node(expr))

    def indexed_sqrt_rule(self, _sqrt, _lbrack, idx_expr, _rbrack, _lb, expr, _rb):
        return SqrtNode(self.ensure_node(expr), self.ensure_node(idx_expr))

    def sin_rule(self, *_): return FunctionCallNode("\\sin", self.ensure_node(_[-2]))
    def cos_rule(self, *_): return FunctionCallNode("\\cos", self.ensure_node(_[-2]))
    def tan_rule(self, *_): return FunctionCallNode("\\tan", self.ensure_node(_[-2]))
    def log_rule(self, *_): return FunctionCallNode("\\log", self.ensure_node(_[-2]))
    def log_with_base_rule(self, _log, _us, _lb, base, _rb, _lp, expr, _rp):
        return FunctionCallNode("\\log", self.ensure_node(expr), base=self.ensure_node(base))
    def ln_rule(self, *_):  return FunctionCallNode("\\ln",  self.ensure_node(_[-2]))
    def exp_rule(self, *_): return FunctionCallNode("\\exp", self.ensure_node(_[-2]))
    def arcsin_rule(self, *_): return FunctionCallNode("\\arcsin", self.ensure_node(_[-2]))
    def arccos_rule(self, *_): return FunctionCallNode("\\arccos", self.ensure_node(_[-2]))
    def arctan_rule(self, *_): return FunctionCallNode("\\arctan", self.ensure_node(_[-2]))
    def sinh_rule(self, *_): return FunctionCallNode("\\sinh", self.ensure_node(_[-2]))
    def cosh_rule(self, *_): return FunctionCallNode("\\cosh", self.ensure_node(_[-2]))
    def tanh_rule(self, *_): return FunctionCallNode("\\tanh", self.ensure_node(_[-2]))
    def det_rule(self, *_):  return FunctionCallNode("\\det", self.ensure_node(_[-2]))

    def _int_val(self, node):
        if isinstance(node, NumberNode):
            try:
                n = int(node.value)
                if float(node.value) == float(n):
                    return n
            except Exception:
                pass
        return None

    def _inv_trig_or_power(self, name, p_node, arg_node):
        n = self._int_val(p_node)
        if n == -1 and name in ("\\sin", "\\cos", "\\tan"):
            inv = {"\\sin":"\\arcsin", "\\cos":"\\arccos", "\\tan":"\\arctan"}[name]
            return FunctionCallNode(inv, arg_node)
        return FunctionCallNode(name, arg_node, power=p_node)

    def sin_power_rule(self, _sin, _cf, power, _lp, expr, _rp): return self._inv_trig_or_power("\\sin", self.ensure_node(power), self.ensure_node(expr))
    def cos_power_rule(self, _cos, _cf, power, _lp, expr, _rp): return self._inv_trig_or_power("\\cos", self.ensure_node(power), self.ensure_node(expr))
    def tan_power_rule(self, _tan, _cf, power, _lp, expr, _rp): return self._inv_trig_or_power("\\tan", self.ensure_node(power), self.ensure_node(expr))
    def sin_power_rule_brace(self, _sin, _cf, _lb, power, _rb, _lp, expr, _rp): return self._inv_trig_or_power("\\sin", self.ensure_node(power), self.ensure_node(expr))
    def cos_power_rule_brace(self, _cos, _cf, _lb, power, _rb, _lp, expr, _rp): return self._inv_trig_or_power("\\cos", self.ensure_node(power), self.ensure_node(expr))
    def tan_power_rule_brace(self, _tan, _cf, _lb, power, _rb, _lp, expr, _rp): return self._inv_trig_or_power("\\tan", self.ensure_node(power), self.ensure_node(expr))

    def min_rule(self, _min, _lp, args, _rp):
        args_list = args if isinstance(args, list) else [args]
        return FunctionCallNode("\\min", [self.ensure_node(a) for a in args_list])
    def max_rule(self, _max, _lp, args, _rp):
        args_list = args if isinstance(args, list) else [args]
        return FunctionCallNode("\\max", [self.ensure_node(a) for a in args_list])

    def generic_func_rule(self, cmd_tok, _lp, args, _rp):
        args_list = args if isinstance(args, list) else [args]
        return FunctionCallNode(cmd_tok.value, [self.ensure_node(a) for a in args_list])

    def arg_list_rule(self, *children):
        args = []
        for c in children:
            if isinstance(c, Token) and c.type == 'COMMA':
                continue
            args.append(self.ensure_node(c))
        return args

    def matrix_row_rule(self, *children):
        row = [self.ensure_node(c) for c in children if not (isinstance(c, Token) and c.type == 'AMP')]
        return row

    def matrix_rows_rule(self, *children):
        return [child for child in children if isinstance(child, list)]

    def matrix_rule(self, *children):
        if len(children) != 9:
            raise ValueError(f"Unexpected number of children in matrix_rule: {len(children)}")
        mtype1 = children[2]; rows = children[4]; mtype2 = children[7]
        if str(mtype1) != str(mtype2):
            raise ValueError(f"Mismatched matrix types: {mtype1} vs {mtype2}")
        return MatrixNode(rows, matrix_type=str(mtype1))
    def pmatrix_type(self): return "pmatrix"
    def bmatrix_type(self): return "bmatrix"
    def matrix_type_plain(self): return "matrix"

    def abs_rule(self, _open, expr, _close): return AbsNode(self.ensure_node(expr))
    def norm_rule(self, _open, expr, _close): return NormNode(self.ensure_node(expr), None)
    def norm_rule_ord_num(self, _open, expr, _close, _us, num): return NormNode(self.ensure_node(expr), self.ensure_node(num))
    def norm_rule_ord_expr(self, _open, expr, _close, _us, _lb, ord_expr, _rb): return NormNode(self.ensure_node(expr), self.ensure_node(ord_expr))

    def inner_rule(self, _langle, args, _rangle):
        args_list = args if isinstance(args, list) else [args]
        return InnerProductNode([self.ensure_node(a) for a in args_list])

    def union(self, left, _tok, right): return SetOpNode('\\cup', self.ensure_node(left), self.ensure_node(right))
    def intersect(self, left, _tok, right): return SetOpNode('\\cap', self.ensure_node(left), self.ensure_node(right))
    def land(self, left, _tok, right): return LogicOpNode('\\land', self.ensure_node(left), self.ensure_node(right))
    def lor(self, left, _tok, right): return LogicOpNode('\\lor', self.ensure_node(left), self.ensure_node(right))
    def implies(self, left, _tok, right): return LogicOpNode('\\implies', self.ensure_node(left), self.ensure_node(right))
    def iff(self, left, _tok, right): return LogicOpNode('\\iff', self.ensure_node(left), self.ensure_node(right))
    def eq(self, left, _tok, right): return RelOpNode('=', self.ensure_node(left), self.ensure_node(right))
    def in_rel(self, left, _tok, right): return RelOpNode('\\in', self.ensure_node(left), self.ensure_node(right))
    def subset(self, left, _tok, right): return RelOpNode('\\subset', self.ensure_node(left), self.ensure_node(right))
    def subseteq(self, left, _tok, right): return RelOpNode('\\subseteq', self.ensure_node(left), self.ensure_node(right))
    def supset(self, left, _tok, right): return RelOpNode('\\supset', self.ensure_node(left), self.ensure_node(right))
    def supseteq(self, left, _tok, right): return RelOpNode('\\supseteq', self.ensure_node(left), self.ensure_node(right))

    def not_rule(self, _not_tok, factor_node): return NotNode(self.ensure_node(factor_node))
    def emptyset_lit(self, _): return EmptySetNode()

    def empty_set_braces_rule(self, _lb, _rb): return SetNode([])
    def set_elems_rule(self, *children):
        items = []
        for c in children:
            if isinstance(c, Token) and c.type == 'COMMA': continue
            items.append(self.ensure_node(c))
        return items
    def set_literal_rule(self, _lb, elems, _rb):
        elems_list = elems if isinstance(elems, list) else [elems]
        return SetNode(elems_list)

    def sb_bound_rule(self, sym_tok, _in_tok, expr): return (SymbolNode(sym_tok.value), self.ensure_node(expr))
    def sb_bound_list_rule(self, *items):
        out = []
        for it in items:
            if isinstance(it, Token) and it.type == 'COMMA': continue
            out.append(it if isinstance(it, tuple) else self.ensure_node(it))
        return out
    def set_builder_rule(self, _lb, bounds, _sep_tok, predicate, _rb):
        blist = bounds if isinstance(bounds, list) else [bounds]
        return SetBuilderNode(blist, self.ensure_node(predicate))

    def qbound_in_rule(self, sym_tok, _in_tok, expr): return (SymbolNode(sym_tok.value), self.ensure_node(expr))
    def qbound_plain_rule(self, sym_tok): return (SymbolNode(sym_tok.value), None)
    def qbound_list_rule(self, *items):
        out = []
        for it in items:
            if isinstance(it, Token) and it.type == 'COMMA': continue
            out.append(it if isinstance(it, tuple) else self.ensure_node(it))
        return out
    def forall_rule(self, _forall, bounds, _colon, body):
        blist = bounds if isinstance(bounds, list) else [bounds]
        return QuantifierNode('forall', blist, self.ensure_node(body))
    def exists_rule(self, _exists, bounds, _colon, body):
        blist = bounds if isinstance(bounds, list) else [bounds]
        return QuantifierNode('exists', blist, self.ensure_node(body))

    def cases_rows_rule(self, *children): return [c for c in children if isinstance(c, tuple)]
    def cases_row_rule(self, expr1, *rest):
        if not rest:
            return (self.ensure_node(expr1), None)
        cond = None
        for r in rest:
            if isinstance(r, Token) and r.type == 'AMP': continue
            cond = self.ensure_node(r)
        return (self.ensure_node(expr1), cond)
    def cases_rule(self, *children):
        rows = None
        for c in children:
            if isinstance(c, list):
                rows = c
                break
        rows = rows or []
        norm = []
        for row in rows:
            if isinstance(row, tuple) and len(row) == 2:
                e, cond = row
                norm.append((self.ensure_node(e), (self.ensure_node(cond) if cond is not None else None)))
            else:
                norm.append((self.ensure_node(row), None))
        return PiecewiseNode(norm)

    def mod(self, left, _op_tok, right): return ModNode(self.ensure_node(left), self.ensure_node(right))

    def argext_plain_rule(self, cmd_tok, body):
        kind = "argmax" if "argmax" in cmd_tok.value else "argmin"
        return ArgExtremumNode(kind, self.ensure_node(body), sub=None, sup=None)
    def argext_sub_rule(self, cmd_tok, _us, _lb, sub, _rb, body):
        kind = "argmax" if "argmax" in cmd_tok.value else "argmin"
        return ArgExtremumNode(kind, self.ensure_node(body), sub=self.ensure_node(sub), sup=None)
    def argext_both_rule(self, cmd_tok, _us, _lb, sub, _rb, _cf, _lb2, sup, _rb2, body):
        kind = "argmax" if "argmax" in cmd_tok.value else "argmin"
        return ArgExtremumNode(kind, self.ensure_node(body), sub=self.ensure_node(sub), sup=self.ensure_node(sup))

    def hat_rule(self, _hat, _lb, expr, _rb): return AccentNode('hat', self.ensure_node(expr))
    def hat_atom_rule(self, _hat, atom): return AccentNode('hat', self.ensure_node(atom))
    def bar_rule(self, _bar, _lb, expr, _rb): return AccentNode('bar', self.ensure_node(expr))
    def bar_atom_rule(self, _bar, atom): return AccentNode('bar', self.ensure_node(atom))
    def vec_rule(self, _vec, _lb, expr, _rb): return AccentNode('vec', self.ensure_node(expr))
    def vec_atom_rule(self, _vec, atom): return AccentNode('vec', self.ensure_node(atom))
    def tilde_rule(self, _tilde, _lb, expr, _rb): return AccentNode('tilde', self.ensure_node(expr))
    def tilde_atom_rule(self, _tilde, atom): return AccentNode('tilde', self.ensure_node(atom))

    def times(self, left, _tok, right): return BinOpNode('\\times', self.ensure_node(left), self.ensure_node(right))
    def odot(self, left, _tok, right): return BinOpNode('\\odot', self.ensure_node(left), self.ensure_node(right))
    def equiv(self, left, _tok, right): return RelOpNode('\\equiv', self.ensure_node(left), self.ensure_node(right))
    def approx(self, left, _tok, right): return RelOpNode('\\approx', self.ensure_node(left), self.ensure_node(right))
    def neq(self, left, _tok, right): return RelOpNode('\\neq', self.ensure_node(left), self.ensure_node(right))
    def sim(self, left, _tok, right): return RelOpNode('\\sim', self.ensure_node(left), self.ensure_node(right))

    def _int_kind_from_token(self, tok):
        t = tok.type
        if t == "IINT": return "iint"
        if t == "IIINT": return "iiint"
        if t == "OINT": return "oint"
        if t == "OIINT": return "oiint"
        if t == "OIIINT": return "oiiint"
        return "iint"

    def diffs2_rule(self, d1, d2): return [self.ensure_node(d1), self.ensure_node(d2)]
    def diffs3_rule(self, d1, d2, d3): return [self.ensure_node(d1), self.ensure_node(d2), self.ensure_node(d3)]

    def multi_int2_indef_rule(self, tok, expr, dlist): return MultiIntegralNode(self._int_kind_from_token(tok), self.ensure_node(expr), dlist)
    def multi_int2_region_rule(self, tok, _us, _lb, subexpr, _rb, expr, dlist): return MultiIntegralNode(self._int_kind_from_token(tok), self.ensure_node(expr), dlist, sub=self.ensure_node(subexpr))
    def multi_int2_def_rule(self, tok, _us, _lb, lower, _rb, _cf, _lb2, upper, _rb2, expr, dlist):
        return MultiIntegralNode(self._int_kind_from_token(tok), self.ensure_node(expr), dlist, lower=self.ensure_node(lower), upper=self.ensure_node(upper))
    def multi_int3_indef_rule(self, tok, expr, dlist): return MultiIntegralNode(self._int_kind_from_token(tok), self.ensure_node(expr), dlist)
    def multi_int3_region_rule(self, tok, _us, _lb, subexpr, _rb, expr, dlist): return MultiIntegralNode(self._int_kind_from_token(tok), self.ensure_node(expr), dlist, sub=self.ensure_node(subexpr))
    def multi_int3_def_rule(self, tok, _us, _lb, lower, _rb, _cf, _lb2, upper, _rb2, expr, dlist):
        return MultiIntegralNode(self._int_kind_from_token(tok), self.ensure_node(expr), dlist, lower=self.ensure_node(lower), upper=self.ensure_node(upper))
    def contour_indef_rule(self, tok, expr, d): return MultiIntegralNode(self._int_kind_from_token(tok), self.ensure_node(expr), [self.ensure_node(d)])
    def contour_region_rule(self, tok, _us, _lb, subexpr, _rb, expr, d):
        return MultiIntegralNode(self._int_kind_from_token(tok), self.ensure_node(expr), [self.ensure_node(d)], sub=self.ensure_node(subexpr))
    def contour_def_rule(self, tok, _us, _lb, lower, _rb, _cf, _lb2, upper, _rb2, expr, d):
        return MultiIntegralNode(self._int_kind_from_token(tok), self.ensure_node(expr), [self.ensure_node(d)], lower=self.ensure_node(lower), upper=self.ensure_node(upper))


parser = Lark(grammar, parser='lalr', transformer=ASTTransformer(), propagate_positions=False, debug=False)

def parse_latex_ast(latex_string):
    return parser.parse(latex_string)

def print_ast(node, indent=0):
    prefix = "  " * indent
    if node is None:
        print(f"{prefix}None"); return

    node_name = type(node).__name__
    details = ""
    children = []

    if isinstance(node, ProgramNode):
        details = f"({len(node.statements)} statements)"
        children = node.statements
    elif isinstance(node, NumberNode):
        details = f"({node.value})"
    elif isinstance(node, InfinityNode):
        details = ""
    elif isinstance(node, SymbolNode):
        details = f"({node.name})"
    elif isinstance(node, SubscriptNode):
        details = f"(Base: {node.base}, Sub: {node.subscript})"
    elif isinstance(node, BinOpNode):
        details = f"(op='{node.op}')"
        children = [("Left", node.left), ("Right", node.right)]
    elif isinstance(node, FractionNode):
        children = [("Num", node.numerator), ("Den", node.denominator)]
    elif isinstance(node, SqrtNode):
        children.append(("Radicand", node.radicand))
        if node.index:
            children.append(("Index", node.index))
    elif isinstance(node, FunctionCallNode):
        details = f"({node.func_name})"
        children.append(("Arg", node.argument if node.argument else node.arguments))
        if node.power:
            children.append(("Power", node.power))
        if node.base:
            children.append(("Base", node.base))
    elif isinstance(node, IntegralNode):
        children = [("Integrand", node.integrand), ("Var", node.var)]
        if node.lower is not None:
            children.append(("Lower", node.lower))
        if node.upper is not None:
            children.append(("Upper", node.upper))
    elif isinstance(node, SumNode):
        children = [("Summand", node.summand), ("Var", node.var), ("Lower", node.lower), ("Upper", node.upper)]
    elif isinstance(node, ProductNode):
        children = [("Body", node.body), ("Var", node.var), ("Lower", node.lower), ("Upper", node.upper)]
    elif isinstance(node, DerivativeNode):
        details = f"({len(node.var_orders)} vars, kind={node.kind})"
        children = [("Expr", node.expr), ("Vars", node.var_orders)]
    elif isinstance(node, GradientNode):
        details = f"({len(node.vars_list)} vars)"
        children = [("Expr", node.expr), ("Vars", node.vars_list)]
    elif isinstance(node, LaplacianNode):
        details = f"({len(node.vars_list)} vars, power={node.power})"
        children = [("Expr", node.expr), ("Vars", node.vars_list), ("Power", node.power)]
    elif isinstance(node, AbsNode):
        children = [("Expr", node.expr)]
    elif isinstance(node, NormNode):
        details = f"(order={node.order})" if node.order is not None else "(order=None)"
        children = [("Expr", node.expr)]
    elif isinstance(node, InnerProductNode):
        details = f"({len(node.args)} args)"
        children = [("Args", node.args)]
    elif isinstance(node, EmptySetNode):
        details = ""
    elif isinstance(node, SetNode):
        details = f"({len(node.elements)} elems)"
        children = [("Elems", node.elements)]
    elif isinstance(node, SetBuilderNode):
        details = f"(bounds={len(node.bounds)})"
        children = [("Bounds", [b[0] for b in node.bounds]), ("Domains", [b[1] for b in node.bounds]), ("Pred", node.predicate)]
    elif isinstance(node, PiecewiseNode):
        details = f"({len(node.pieces)} pieces)"
        children = [("Pieces", node.pieces)]
    elif isinstance(node, RelOpNode):
        details = f"(op='{node.op}')"
        children = [("Left", node.left), ("Right", node.right)]
    elif isinstance(node, SetOpNode):
        details = f"(op='{node.op}')"
        children = [("Left", node.left), ("Right", node.right)]
    elif isinstance(node, LogicOpNode):
        details = f"(op='{node.op}')"
        children = [("Left", node.left), ("Right", node.right)]
    elif isinstance(node, NotNode):
        children = [("Expr", node.expr)]
    elif isinstance(node, QuantifierNode):
        details = f"({node.kind}, bounds={len(node.bounds)})"
        children = [("Bounds", node.bounds), ("Body", node.body)]
    elif isinstance(node, AccentNode):
        details = f"({node.kind})"
        children = [("Base", node.base)]
    elif isinstance(node, ModNode):
        children = [("Left", node.left), ("Right", node.right)]
    elif isinstance(node, ArgExtremumNode):
        details = f"({node.kind}, sub={'yes' if node.sub else 'no'}, sup={'yes' if node.sup else 'no'})"
        children = [("Body", node.body)]
        if node.sub is not None: children.append(("Sub", node.sub))
        if node.sup is not None: children.append(("Sup", node.sup))
    elif isinstance(node, MultiIntegralNode):
        details = f"(kind={node.kind}, diffs={len(node.diffs)})"
        children = [("Integrand", node.integrand), ("Diffs", node.diffs)]
        if node.lower is not None or node.upper is not None:
            children.append(("Lower", node.lower))
            children.append(("Upper", node.upper))
        if node.sub is not None: children.append(("Sub", node.sub))
        if node.sup is not None: children.append(("Sup", node.sup))
    elif isinstance(node, MatrixNode):
        details = f"({len(node.rows)} rows, type={node.matrix_type})"
        children = [("Rows", node.rows)]
    else:
        details = f"({str(node)})"

    print(f"{prefix}{node_name}{details}")
    for child_info in children:
        if isinstance(child_info, tuple):
            label, child_node = child_info
            print(f"{prefix}  +{label}:")
            if isinstance(child_node, list):
                if not child_node:
                    print(f"{prefix}    (empty list)")
                for i, item in enumerate(child_node):
                    if (
                        isinstance(item, tuple)
                        and len(item) == 2
                        and isinstance(item[0], ASTNode)
                        and isinstance(item[1], ASTNode)
                    ):
                        print(f"{prefix}    - Case {i}:")
                        print(f"{prefix}      Expr:")
                        print_ast(item[0], indent + 4)
                        print(f"{prefix}      Cond:")
                        print_ast(item[1], indent + 4)
                    elif isinstance(item, list):
                        print(f"{prefix}    - Row {i}:")
                        for j, cell in enumerate(item):
                            print(f"{prefix}      [{j}]:")
                            print_ast(cell, indent + 4)
                    else:
                        print(f"{prefix}    - [{i}]:")
                        print_ast(item, indent + 3)
            else:
                print_ast(child_node, indent + 2)
        else:
            print_ast(child_info, indent + 1)


def main():
    ap = argparse.ArgumentParser(description="Parse LaTeX into an AST and print the structure (SymPy-free).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pre", type=str, help="LaTeX string to preprocess then parse")
    g.add_argument("--raw", type=str, help="LaTeX string to parse as-is")
    args = ap.parse_args()

    try:
        if args.pre is not None:
            if not HAVE_PREPROCESS:
                print("ERROR: preprocess_latex not available, but --pre was used.", file=sys.stderr)
                sys.exit(2)
            s = preprocess_latex(args.pre)
        else:
            s = args.raw

        ast = parse_latex_ast(s)
        print_ast(ast)
    except Exception:
        print("\n!!! PARSING ERROR !!!")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
import sys
import argparse
import re


from ast_builder import (
    parse_latex_ast,
    ASTNode,
    ProgramNode,
    NumberNode,
    SymbolNode,
    BinOpNode,
    FunctionCallNode,
    MatrixNode,
    DerivativeNode,
    FractionNode,
    SumNode,
    ProductNode,
    MultiIntegralNode,
    IntegralNode,
    AbsNode,
    SqrtNode,
    NormNode,
    InfinityNode,
    SubscriptNode,
    SetNode,
    RelOpNode,
    GradientNode,
    LaplacianNode,
    InnerProductNode,
    PiecewiseNode,
    AccentNode,
    ModNode,
    ArgExtremumNode,
    print_ast
)


try:
    from preprocess_latex import preprocess_latex
except ImportError:
    print("ERROR: preprocess_latex not available!", file=sys.stderr)
    sys.exit(2)


class ASTToFutharkTranspiler:

    def __init__(self):
        self.symbols = {}
        self.bound_variables = set()
        self.current_indices = set()
        self.symbol_list = []
        self.diff_h = 1e-6
        self.integ_steps_1d = 100000
        self.integ_steps_nd = 300
        self.derivative_vars = set()
        self.expr_types = {}


        self.func_map = {
            '\\sin': 'f64.sin',
            '\\cos': 'f64.cos',
            '\\tan': 'f64.tan',
            '\\ln': 'f64.log',
            '\\log': 'f64.log',
            '\\exp': 'f64.exp',
            '\\sqrt': 'f64.sqrt',
            '\\arcsin': 'f64.asin',
            '\\arccos': 'f64.acos',
            '\\arctan': 'f64.atan',
            '\\det': 'det',
            '\\abs': 'f64.abs',
            '\\sinh': 'f64.sinh',
            '\\cosh': 'f64.cosh',
            '\\tanh': 'f64.tanh',
            '\\min': 'f64.min',
            '\\max': 'f64.max'
        }

    def transpile(self, ast_node):

        self._collect_symbols(ast_node)

        if isinstance(ast_node, ProgramNode):

            results = []
            for stmt in ast_node.statements:
                results.append(self.visit(stmt))
            return '\n'.join(results)
        else:
            return self.visit(ast_node)


    def _collect_symbols(self, node):
        if node is None:
            return

        if isinstance(node, SymbolNode):
            name = node.name
            if name not in ['e', 'E', 'pi', 'π'] and name not in self.bound_variables:
                self.register_symbol(name)

        elif isinstance(node, SubscriptNode):
            base = node.base
            if isinstance(base, ASTNode):
                self._collect_symbols(base)
            else:
                self.register_symbol(base)
            if isinstance(node.subscript, ASTNode):
                self._collect_symbols(node.subscript)

        elif isinstance(node, (SumNode, ProductNode)):
            if hasattr(node.var, 'name'):
                self.bound_variables.add(node.var.name)
            self._collect_symbols(node.lower)
            self._collect_symbols(node.upper)
            body = node.body if hasattr(node, 'body') else node.summand
            self._collect_symbols(body)
            if hasattr(node.var, 'name'):
                self.bound_variables.discard(node.var.name)

        elif isinstance(node, DerivativeNode):


            self._collect_symbols(node.expr)


            for vo in node.var_orders:
                v_ast = vo[0] if isinstance(vo, tuple) else vo
                self._collect_symbols(v_ast)
                if isinstance(vo, tuple):
                    self._collect_symbols(vo[1])

        elif isinstance(node, IntegralNode):
            if hasattr(node.var, 'name'):
                self.bound_variables.add(node.var.name)
            self._collect_symbols(node.integrand)
            self._collect_symbols(node.lower)
            self._collect_symbols(node.upper)
            if hasattr(node.var, 'name'):
                self.bound_variables.discard(node.var.name)

        elif isinstance(node, MultiIntegralNode):
            temp = []
            for v in node.diffs:
                if hasattr(v, 'name'):
                    self.bound_variables.add(v.name); temp.append(v.name)
            self._collect_symbols(node.integrand)
            self._collect_symbols(node.lower)
            self._collect_symbols(node.upper)
            for vname in temp: self.bound_variables.discard(vname)

        elif isinstance(node, list):
            for item in node:
                self._collect_symbols(item)

        elif hasattr(node, '__dict__'):
            for attr in node.__dict__.values():
                if isinstance(attr, ASTNode):
                    self._collect_symbols(attr)
                elif isinstance(attr, list):

                    for it in attr:
                        self._collect_symbols(it)

    def register_symbol(self, name, type_='f64'):
        if name not in self.symbols and name not in self.bound_variables:
            self.symbols[name] = type_
            if name not in self.symbol_list:
                self.symbol_list.append(name)

    def visit(self, node):
        if node is None:
            return '0.0'

        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        return f'-- Unhandled: {type(node).__name__}'

    def visit_NumberNode(self, node):
        value = node.value
        if isinstance(value, int):
            return f'{value}.0'
        return str(value)

    def visit_InfinityNode(self, node):
        return 'f64.inf'

    def visit_SymbolNode(self, node):
        name = node.name


        if name in ['e', 'E']:
            return '(f64.exp 1.0)'
        if name in ['pi', 'π']:
            return 'f64.pi'


        if name in self.current_indices:
            return f'f64.i64 {name}'

        return name

    def visit_SubscriptNode(self, node):
        base_str = self.visit(node.base) if isinstance(node.base, ASTNode) else str(node.base)
        sub_str = self.visit(node.subscript) if isinstance(node.subscript, ASTNode) else str(node.subscript)
        return f'{base_str}_{sub_str}'

    def visit_BinOpNode(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)

        op_map = {
            '+': '+', '-': '-', '*': '*', '/': '/',
            '^': '**', '**': '**',
            '\\cdot': '*', '\\times': '*', '\\odot': '*',
        }
        op = op_map.get(node.op, node.op)

        if op == '**':
            if left in ('f64.e', '(f64.exp 1.0)'):
                return f'(f64.exp {right})'
            return f'({left} ** {right})'


        left_type = self.expr_types.get(left, 'scalar')
        right_type = self.expr_types.get(right, 'scalar')

        if left_type == 'scalar' and right_type == 'array':
            result = f'(map (\\elB_ -> {left} {op} elB_) {right})'
            self.expr_types[result] = 'array'
            return result
        elif left_type == 'array' and right_type == 'scalar':
            result = f'(map (\\elB_ -> elB_ {op} {right}) {left})'
            self.expr_types[result] = 'array'
            return result
        elif left_type == 'array' and right_type == 'array':
            result = f'(map2 (\\xB_ yB_ -> xB_ {op} yB_) {left} {right})'
            self.expr_types[result] = 'array'
            return result

        if op == '/':
            return f'({left} / {right})'

        return f'({left} {op} {right})'

    def visit_FractionNode(self, node):
        num = self.visit(node.numerator)
        den = self.visit(node.denominator)
        return f'({num} / {den})'

    def visit_SqrtNode(self, node):
        rad = self.visit(node.radicand)


        if not (rad.startswith('(') and rad.endswith(')')):
            rad = f'({rad})'


        is_square = node.index is None
        if not is_square:
            try:
                idx_val = float(self.visit(node.index))
                is_square = (idx_val == 2.0)
            except Exception:
                is_square = False

        if is_square:
            return f'(f64.sqrt {rad})'
        else:
            idx = self.visit(node.index)
            if not (idx.startswith('(') and idx.endswith(')')):
                idx = f'({idx})'
            return f'({rad} ** (1.0 / {idx}))'

    def visit_AbsNode(self, node):
        expr = self.visit(node.expr)

        if not (expr.startswith('(') and expr.endswith(')')):
            expr = f'({expr})'
        return f'f64.abs {expr}'

    def visit_FunctionCallNode(self, node):
        func_name = node.func_name


        futhark_func = self.func_map.get(func_name, func_name.lstrip('\\'))


        if node.argument:
            if isinstance(node.argument, list):
                args = [self.visit(a) for a in node.argument]
            else:
                arg = self.visit(node.argument)
                args = [arg]
        elif node.arguments:
            args = [self.visit(a) for a in node.arguments]
        else:
            args = []


        if node.power:
            power = self.visit(node.power)

            if hasattr(node.power, 'value') and node.power.value == -1:
                if func_name in ['\\sin', '\\cos', '\\tan']:
                    inv_map = {'\\sin': 'f64.asin', '\\cos': 'f64.acos', '\\tan': 'f64.atan'}
                    futhark_func = inv_map[func_name]
            else:

                base_result = f'{futhark_func} {args[0]}' if len(args) == 1 else f'{futhark_func}({", ".join(args)})'
                return f'({base_result} ** {power})'


        if func_name == '\\log' and node.base:
            base = self.visit(node.base)
            if len(args) > 0:
                return f'(f64.log {args[0]} / f64.log {base})'


        if func_name == '\\det' and args:
            return f'det {args[0]}'


        if func_name in ['\\min', '\\max'] and len(args) == 1:

            arg = args[0]
            reduction_func = 'f64.minimum' if func_name == '\\min' else 'f64.maximum'

            arg_type = self.expr_types.get(arg, 'unknown')
            if arg_type == 'array' or '(map' in arg or 'flatten' in arg or '[' in arg:

                if ' ' in arg and not arg.startswith('('):
                    arg = f'({arg})'
                return f'{reduction_func} {arg}'
            else:


                pass

        elif len(args) == 1:
            arg = args[0]

            if ' ' in arg and not arg.startswith('('):
                arg = f'({arg})'
            result = f'{futhark_func} {arg}'

            if futhark_func in ['f64.min', 'f64.max']:
                result = f'({result})'
            return result
        elif len(args) > 1:

            if futhark_func in ['f64.min', 'f64.max']:
                result = args[-1]
                for arg in reversed(args[:-1]):
                    result = f'{futhark_func} {arg} ({result})'
                return result
            return f'{futhark_func}({", ".join(args)})'
        else:
            return f'{futhark_func}'

    def visit_MatrixNode(self, node):
        rows = []
        for row in node.rows:
            elems = [self.visit(elem) for elem in row]
            rows.append(f'[{", ".join(elems)}]')
        result = f'[{", ".join(rows)}]'
        self.expr_types[result] = 'array'
        return result

    def visit_DerivativeNode(self, node):


        for vo in node.var_orders:
            v_ast = vo[0] if isinstance(vo, tuple) else vo
            vname = v_ast.name if hasattr(v_ast, 'name') else str(v_ast)
            self.register_symbol(vname)

        expr = self.visit(node.expr)


        orders = {}
        order_list = []
        for vo in node.var_orders:
            if isinstance(vo, tuple):
                v_ast, ord_ast = vo
                vname = v_ast.name if hasattr(v_ast, 'name') else str(v_ast)
                ordv = int(float(self.visit(ord_ast)))
            else:
                v_ast = vo
                vname = v_ast.name if hasattr(v_ast, 'name') else str(v_ast)
                ordv = 1
            if vname not in orders:
                orders[vname] = 0; order_list.append(vname)
            orders[vname] += ordv

        if not order_list:
            return '0.0'

        params = ' '.join(order_list)
        h = self.diff_h
        h_str = repr(float(h))

        code = []
        code.append(f"let h = {h_str} in")
        code.append(f"let f0 = \\{params} -> {expr} in")
        cur = "f0"
        for v in order_list:
            for r in range(orders[v]):
                call_plus  = ' '.join([f"({w} + h)" if w == v else w for w in order_list])
                call_minus = ' '.join([f"({w} - h)" if w == v else w for w in order_list])
                fn = f"f_{v}_{r}"
                code.append(f"let {fn} = \\{params} -> (({cur} {call_plus}) - ({cur} {call_minus})) / (2.0*h) in")
                cur = fn

        args_now = ' '.join(order_list)
        body = '\n  '.join(code)
        return f"(\n  {body}\n  {cur} {args_now}\n)"

    def visit_GradientNode(self, node):
        expr = self.visit(node.expr)


        names = []
        if hasattr(node, 'vars_list') and node.vars_list:
            names = [v.name if hasattr(v, 'name') else str(v) for v in node.vars_list]
        else:

            names = [n for n in self.symbol_list if n not in self.bound_variables]

        if not names:
            return '([]: []f64)'


        for v in names:
            self.register_symbol(v)

        params = ' '.join(names)
        args_now = ' '.join(names)
        h_str = repr(float(self.diff_h))


        lines = [f"let h = {h_str} in",
                f"let f0 = \\{params} -> {expr} in"]
        dnames = []
        for v in names:
            call_plus  = ' '.join([f"({w} + h)" if w == v else w for w in names])
            call_minus = ' '.join([f"({w} - h)" if w == v else w for w in names])
            dv = f"d_{v}"
            lines.append(
                f"let {dv} = \\{params} -> ((f0 {call_plus}) - (f0 {call_minus})) / (2.0*h) in"
            )
            dnames.append(dv)

        vec = ', '.join([f"{dv} {args_now}" for dv in dnames])
        body = '\n  '.join(lines)
        result = f"(\n  {body}\n  [{vec}]\n)"
        self.expr_types[result] = 'array'
        return result

    def visit_LaplacianNode(self, node):
        expr = self.visit(node.expr)


        names = []
        if hasattr(node, 'vars_list') and node.vars_list:
            names = [v.name if hasattr(v, 'name') else str(v) for v in node.vars_list]
        else:
            names = [n for n in self.symbol_list if n not in self.bound_variables]

        if not names:
            return '0.0'


        for v in names:
            self.register_symbol(v)


        p = 1
        if getattr(node, 'power', None) is not None:
            try:
                p = int(float(self.visit(node.power)))
            except Exception:
                p = 1
        p = max(1, p)

        params   = ' '.join(names)
        args_now = ' '.join(names)
        h_str    = repr(float(self.diff_h))

        lines = [f"let h = {h_str} in",
                f"let f0 = \\{params} -> {expr} in"]

        cur = "f0"
        for kk in range(p):

            second_terms = []
            call_0 = args_now
            for v in names:
                call_plus  = ' '.join([f"({w} + h)" if w == v else w for w in names])
                call_minus = ' '.join([f"({w} - h)" if w == v else w for w in names])
                term = f"(({cur} {call_plus}) - 2.0*({cur} {call_0}) + ({cur} {call_minus})) / (h*h)"
                second_terms.append(term)
            body = ' + '.join(second_terms) if second_terms else '0.0'
            fn = f"L_{kk}"
            lines.append(f"let {fn} = \\{params} -> {body} in")
            cur = fn

        body = '\n  '.join(lines)
        return f"(\n  {body}\n  {cur} {args_now}\n)"

    def visit_SumNode(self, node):
        var = node.var; var_name = var.name if hasattr(var, 'name') else str(var)
        self.current_indices.add(var_name)
        lower = self.visit(node.lower); upper = self.visit(node.upper)
        summand = self.visit(node.summand)
        self.current_indices.discard(var_name)
        return f"""(let lower_{var_name} = i64.f64 ({lower}) in
        let upper_{var_name} = i64.f64 ({upper}) in
        let range_size_{var_name} = upper_{var_name} - lower_{var_name} + 1i64 in
        let idxs_{var_name} = map (\\i -> i + lower_{var_name}) (iota range_size_{var_name}) in
        reduce (+) 0.0 (map (\\{var_name} -> {summand}) idxs_{var_name}))"""

    def visit_ProductNode(self, node):
        var = node.var; var_name = var.name if hasattr(var, 'name') else str(var)
        self.current_indices.add(var_name)
        lower = self.visit(node.lower); upper = self.visit(node.upper)
        body = self.visit(node.body)
        self.current_indices.discard(var_name)
        return f"""(let lower_{var_name} = i64.f64 ({lower}) in
        let upper_{var_name} = i64.f64 ({upper}) in
        let range_size_{var_name} = upper_{var_name} - lower_{var_name} + 1i64 in
        let idxs_{var_name} = map (\\i -> i + lower_{var_name}) (iota range_size_{var_name}) in
        reduce (*) 1.0 (map (\\{var_name} -> {body}) idxs_{var_name}))"""

    def visit_IntegralNode(self, node):
        integrand = self.visit(node.integrand)
        var = node.var; var_name = var.name if hasattr(var, 'name') else str(var)
        if node.lower and node.upper:
            lower = self.visit(node.lower); upper = self.visit(node.upper)

            n_steps = self.integ_steps_1d if self.integ_steps_1d % 2 == 0 else self.integ_steps_1d + 1
            return f"""-- Numerical integration (1D Simpson's rule)
        (let n_int_ = {n_steps}i64 in
        let h_int_ = ({upper} - {lower}) / f64.i64 n_int_ in
        let pts_int_ = map (\\iI_ -> {lower} + h_int_ * f64.i64 iI_) (iota (n_int_ + 1i64)) in  # Changed idx_int_ → iI_
        let vals_int_ = map (\\{var_name} -> {integrand}) pts_int_ in
        let w_int_ = map (\\iI_ ->   # Changed idx_int_ → iI_
            if iI_ == 0i64 || iI_ == n_int_ then 1.0  # Changed idx_int_ → iI_
            else if iI_ % 2i64 == 1i64 then 4.0
            else 2.0) (iota (n_int_ + 1i64)) in
        (h_int_ / 3.0) * reduce (+) 0.0 (map2 (*) w_int_ vals_int_))"""
        else:
            return f'-- Indefinite integral\n  0.0'

    def visit_MultiIntegralNode(self, node):
        integrand = self.visit(node.integrand)
        if not (node.lower and node.upper):
            return f'-- Indefinite {node.kind}\n  0.0'

        lower = self.visit(node.lower)
        upper = self.visit(node.upper)

        names = []
        for v in node.diffs:
            names.append(v.name if hasattr(v, 'name') else str(v))
        n = len(names)


        if n == 1:
            var_stub = SymbolNode(names[0])
            fake = IntegralNode(kind='integral', integrand=node.integrand, var=var_stub, lower=node.lower, upper=node.upper)
            return self.visit_IntegralNode(fake)


        steps = self.integ_steps_nd if self.integ_steps_nd % 2 == 0 else self.integ_steps_nd + 1
        header = [
            f"-- {node.kind} ({n}D Simpson's rule tensor grid)",
            f"(let n_mInt_ = {steps}i64 in",
            f" let h_mInt_ = ({upper} - {lower}) / f64.i64 n_mInt_ in",
            f" let w_mInt_ = map (\\iM_ -> ",
            f"     if iM_ == 0i64 || iM_ == n_mInt_ then 1.0",
            f"     else if iM_ % 2i64 == 1i64 then 4.0",
            f"     else 2.0) (iota (n_mInt_ + 1i64)) in",
            f" let grid_mInt_ = map (\\iM_ -> {lower} + h_mInt_ * f64.i64 iM_) (iota (n_mInt_ + 1i64)) in",
        ]
        for nm in names:
            header.append(f" let {nm}s_ = grid_mInt_ in")
        header_str = "\n".join(header)

        def build_loops(depth):
            idx = f"iD{depth}_"
            nm  = names[depth]


            binds = (
                f"    let {nm} = {nm}s_[{idx}]\n"
                f"    let w_d{depth}_ = w_mInt_[{idx}]\n"
            )

            if depth == n - 1:
                wprod = " * ".join([f"w_d{d}_" for d in range(n)]) or "1.0"
                return (
                    f" let acc_d{depth}_ =\n"
                    f"  loop acc_d{depth}_ = 0.0 for {idx} < n_mInt_ + 1i64 do\n"
                    f"{binds}"
                    f"   in acc_d{depth}_ + ({wprod}) * ({integrand})\n"
                )

            inner = build_loops(depth + 1)
            return (
                f" let acc_d{depth}_ =\n"
                f"  loop acc_d{depth}_ = 0.0 for {idx} < n_mInt_ + 1i64 do\n"
                f"{binds}"
                f"{inner}"
                f"   in acc_d{depth}_ + acc_d{depth+1}_\n"
            )

        loops = build_loops(0)
        return f"""{header_str}
        {loops} in ((h_mInt_ / 3.0) ** {float(n)}) * acc_d0_)"""

    def visit_NormNode(self, node):
        expr = self.visit(node.expr)

        if node.order is None:
            order = 2
        elif hasattr(node.order, 'value'):
            order = node.order.value
        else:
            order = self.visit(node.order)


        flat_expr = f'(flatten {expr})'

        if order == 1:
            result = f'(f64.sum (map f64.abs {flat_expr}))'
        elif order == 2:
            result = f'(f64.sqrt (f64.sum (map (\\elN_ -> elN_ * elN_) {flat_expr})))'
        else:
            result = f'((f64.sum (map (\\elN_ -> f64.abs elN_ ** {order}) {flat_expr})) ** (1.0 / {order}))'

        self.expr_types[result] = 'scalar'
        return result

    # Needs work
    def visit_InnerProductNode(self, node):
        if len(node.args) == 2:
            left = self.visit(node.args[0])
            right = self.visit(node.args[1])
            return f'(f64.sum (map2 (*) {left} {right}))'
        else:
            return '0.0  -- Multi-arg inner product not supported'

    def visit_AccentNode(self, node):
        base = self.visit(node.base)

        return f'{base}_{node.kind}'

    def visit_ModNode(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        return f'(i64.f64 {left} %% i64.f64 {right})'

    # Needs work
    def visit_ArgExtremumNode(self, node):
        comment = f'-- {node.kind}'
        if node.sub: comment += f' over {self.visit(node.sub)}'
        return f'{comment}\n 0 -- TODO: implement {node.kind}'

    def visit_PiecewiseNode(self, node):

        result = []
        for i, (expr, cond) in enumerate(node.pieces):
            expr_str = self.visit(expr)
            if cond is not None:
                cond_str = self.visit(cond)
                if i == 0:
                    result.append(f'if {cond_str} then {expr_str}')
                else:
                    result.append(f'else if {cond_str} then {expr_str}')
            else:
                result.append(f'else {expr_str}')
        return '(' + ' '.join(result) + ')'

    def visit_RelOpNode(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_map = {
            '=': '==',
            '\\neq': '!=',
            '\\leq': '<=',
            '\\geq': '>=',
            '<': '<',
            '>': '>',
            '\\in': 'in',
            '\\equiv': '==',
            '\\approx': '==',
            '\\sim': '=='
        }
        op = op_map.get(node.op, node.op)
        return f'{left} {op} {right}'


def generate_futhark_program(ast_node, function_name='compute'):
    transpiler = ASTToFutharkTranspiler()
    futhark_expr = transpiler.transpile(ast_node)

    param_names = sorted(
        name for name in transpiler.symbol_list
        if name not in transpiler.bound_variables
    )

    det_code = ""
    if re.search(r"\bdet\b", futhark_expr):
        det_code = r"""
-- Determinant using Gaussian elimination
let det [n] (a_in: [n][n]f64): f64 =
  let a0 = copy a_in
  let (a, sgn) =
    loop (a, sgn) = (a0, 1.0) for ii < n do
      -- pick pivot in column ii from rows ii..n-1
      let rest =
        map (\off ->
               let jj = ii + 1 + off
               in (jj, a[jj, ii]))
            (iota (n - ii - 1))
      let choose =
        \(best_idx, best_val) (cand_idx, cand_val) ->
          if f64.abs cand_val > f64.abs best_val
          then (cand_idx, cand_val) else (best_idx, best_val)
      let (piv_idx, piv_val) = reduce choose (ii, a[ii, ii]) rest
      in if f64.abs piv_val < 1e-10 then
           -- singular matrix
           (a, 0.0)
         else
           -- swap rows if needed
           let (a, sgn) =
             if piv_idx != ii then
               let row_ii = copy a[ii]
               let row_p  = copy a[piv_idx]
               let a = a with [ii]      = row_p
               let a = a with [piv_idx] = row_ii
               in (a, -sgn)
             else (a, sgn)
           -- eliminate below pivot using a copied pivot row
           let piv_row = copy a[ii]
           let piv     = piv_row[ii]
           let a =
             loop a = a for off < n - ii - 1 do
               let rr    = ii + 1 + off
               let fac   = a[rr, ii] / piv
               let new_r = map2 (\x pv -> x - fac * pv) a[rr] piv_row
               let a     = a with [rr] = new_r
               in a
           in (a, sgn)
  in sgn * reduce (*) 1.0 (map (\kk -> a[kk, kk]) (iota n))
"""

    if param_names:
        param_str = ' '.join(param_names)
        param_use = ' '.join(param_names)
        unpack = '\n  '.join([f'let {name} = params[{i}] in'
                              for i, name in enumerate(param_names)])
        program = f"""-- Auto-generated Futhark code from LaTeX AST
{det_code}
let {function_name} {param_str} =
  {futhark_expr}

entry main (params: []f64) =
  {unpack}
  {function_name} {param_use}
"""
    else:
        program = f"""-- Auto-generated Futhark code from LaTeX AST
{det_code}
let {function_name} =
  {futhark_expr}

entry main =
  {function_name}
"""
    return program


def main():
    parser = argparse.ArgumentParser(
        description="Generate Futhark code from LaTeX via AST",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "\\frac{1}{2} + x^2"
  %(prog)s "\\sum_{i=1}^n i^2"
  echo "\\det\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}" | %(prog)s --ast
"""
    )


    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--pre", type=str,
                            help="(placeholder/deprecated?)")
    input_group.add_argument("--raw", type=str,
                            help="(placeholder/deprecated?)")
    parser.add_argument("latex", nargs="?", help="LaTeX string (will be preprocessed)")


    parser.add_argument("--ast", action="store_true",
                       help="Also print the AST structure")
    parser.add_argument("--function-name", type=str, default="compute",
                       help="Name for the generated function (default: compute)")
    parser.add_argument("-o", "--output", type=str,
                       help="Output file (default: stdout)")

    args = parser.parse_args()

    try:

        if args.pre is not None:
            raw_input = args.pre
        elif args.raw is not None:
            raw_input = args.raw
        elif args.latex is not None:
            raw_input = args.latex
        else:
            raw_input = sys.stdin.read().strip()
            if not raw_input:
                parser.error("Provide LaTeX as a positional argument or via stdin.")


        latex_str = preprocess_latex(raw_input)
        if args.ast:
            print(f"Preprocessed: {latex_str}\n", file=sys.stderr)


        ast = parse_latex_ast(latex_str)


        if args.ast:
            print("=== AST Structure ===", file=sys.stderr)
            print_ast(ast)
            print("\n=== Futhark Code ===", file=sys.stderr)


        futhark_code = generate_futhark_program(ast, args.function_name)


        if args.output:
            with open(args.output, 'w') as f:
                f.write(futhark_code)
            print(f"Futhark code written to {args.output}", file=sys.stderr)
        else:
            print(futhark_code)

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        if args.ast:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

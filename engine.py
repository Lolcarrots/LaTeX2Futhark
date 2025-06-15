import re
import subprocess
import warnings
import xml.etree.ElementTree as ET
from typing import Any, Callable, Union
import sympy
from sympy import Add, Derivative, Determinant, Float, Function, ImmutableDenseMatrix, ImmutableMatrix, Integer, Integral, MatAdd, MatMul, Matrix, MatrixSymbol, Mul, MutableMatrix, Pow, Product, Sum, Symbol, Transpose, acos, asin, atan, cos, cosh, exp, log, sin, sinh, sqrt, sympify, symbols, tan, tanh
import warnings, sympy
warnings.filterwarnings('ignore', message='non-Expr objects in a Matrix is deprecated', module='sympy')

def latex_to_mathml(latex_code: str) -> str:
    result = subprocess.run(['latexmlmath', '--preload=amsmath', '--presentationmathml=-', '-'], input=latex_code, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'LaTeXML error: {result.stderr.strip()}')
    return result.stdout

def parse_mathml_matrix(mathml: str) -> sympy.Matrix:
    ns = '{http://www.w3.org/1998/Math/MathML}'
    root = ET.fromstring(mathml)
    mtable = root.find(f'.//{ns}mtable')
    if mtable is None:
        raise ValueError('No <mtable> found in MathML')
    raw_rows = []
    for mtr in mtable.findall(f'{ns}mtr'):
        row = []
        for mtd in mtr.findall(f'{ns}mtd'):
            row.extend(parse_mathml_cell(mtd))
        raw_rows.append(row)
    if len({len(r) for r in raw_rows}) == 1:
        return sympy.Matrix(raw_rows)
    import math
    cols = math.gcd(*[len(r) for r in raw_rows])
    if cols == 1:
        total = sum((len(r) for r in raw_rows))
        root_n = int(math.isqrt(total))
        if root_n * root_n == total:
            cols = root_n
        else:
            raise ValueError('cannot infer matrix dimensions')
    rebuilt = []
    for r in raw_rows:
        if len(r) % cols:
            raise ValueError('row length not divisible by cols')
        rebuilt.extend([r[i:i + cols] for i in range(0, len(r), cols)])
    return sympy.Matrix(rebuilt)

def parse_nabla_latex(expr: str):
    import re
    m = re.match('\\\\nabla\\s*(.+)', expr)
    if m:
        func_expr = m.group(1).strip()
        try:
            clean_func = func_expr.strip()
            func_sym = parse_any_latex(clean_func)
        except:
            func_sym = sympify(func_expr.strip())
        all_vars = sorted(func_sym.free_symbols, key=lambda s: str(s))
        if not all_vars:
            return sympy.Matrix([[0]])
        gradient_vars = []
        for var in all_vars:
            partial = sympy.diff(func_sym, var)
            if var in partial.free_symbols:
                gradient_vars.append(var)
        if not gradient_vars:
            gradient_vars = all_vars
        gradient_vars = sorted(gradient_vars, key=lambda s: str(s))
        partials = []
        for var in gradient_vars:
            partial = sympy.diff(func_sym, var)
            partials.append([partial])
        return sympy.Matrix(partials)
    return None

def parse_norm_latex(expr: str):
    import re
    m = re.match('\\|\\|(.+)\\|\\|(?:_(\\d+))?', expr)
    if m:
        inner_expr = m.group(1).strip()
        norm_type = m.group(2)
        vec_expr = None
        if '\\begin{pmatrix}' in inner_expr or '\\begin{bmatrix}' in inner_expr:
            try:
                from sympy.parsing.latex import parse_latex as sympy_parse_latex
                vec_expr = sympy_parse_latex(inner_expr)
                print(f'DEBUG: SymPy parsed matrix: {vec_expr}')
                print(f'DEBUG: Matrix type: {type(vec_expr)}')
                if hasattr(vec_expr, 'free_symbols'):
                    print(f'DEBUG: Free symbols: {vec_expr.free_symbols}')
            except Exception as e:
                print(f'DEBUG: SymPy parse failed: {e}')
                try:
                    ml = latex_to_mathml(inner_expr)
                    vec_expr = parse_mathml_matrix(ml)
                    print(f'DEBUG: LaTeXML parsed matrix: {vec_expr}')
                except Exception as e2:
                    print(f'DEBUG: LaTeXML parse also failed: {e2}')
        elif '\\nabla' in inner_expr:
            vec_expr = parse_nabla_latex(inner_expr)
        if vec_expr is None:
            try:
                from sympy.parsing.latex import parse_latex as sympy_parse_latex
                vec_expr = sympy_parse_latex(inner_expr)
            except:
                vec_expr = sympify(inner_expr.strip())
        p = 2 if norm_type is None else int(norm_type)
        if hasattr(vec_expr, 'shape'):
            shape = vec_expr.shape
            elements = []
            if len(shape) == 1:
                for i in range(shape[0]):
                    elements.append(vec_expr[i])
            elif len(shape) == 2:
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        elem = vec_expr[i, j]
                        elements.append(elem)
                        print(f'DEBUG: Matrix element [{i},{j}] = {elem}, type = {type(elem)}')
            print(f'DEBUG: All elements: {elements}')
            if p == 1:
                terms = [sympy.Abs(e) for e in elements]
                norm_expr = sympy.Add(*terms) if terms else sympy.Integer(0)
            elif p == 2:
                terms = [e ** 2 for e in elements]
                sum_of_squares = sympy.Add(*terms) if terms else sympy.Integer(0)
                norm_expr = sympy.sqrt(sum_of_squares)
            else:
                terms = [sympy.Abs(e) ** p for e in elements]
                sum_of_powers = sympy.Add(*terms) if terms else sympy.Integer(0)
                norm_expr = sum_of_powers ** sympy.Rational(1, p)
            print(f'DEBUG: Final norm expression: {norm_expr}')
            if hasattr(norm_expr, 'free_symbols'):
                print(f'DEBUG: Free symbols in norm: {norm_expr.free_symbols}')
            return norm_expr
        else:
            return sympy.Abs(vec_expr)
    return None

def parse_any_latex(expr: str):
    import re
    import xml.etree.ElementTree as ET
    if expr is None:
        print('parse_any_latex received None input')
        return sympy.Integer(0)
    original_expr = expr
    print(f'parse_any_latex called with: {expr}')
    if '\\nabla' in expr:
        nabla_result = parse_nabla_latex(expr)
        if nabla_result is not None:
            print(f'Parsed nabla expression: {nabla_result}')
            return nabla_result
    if '||' in expr and re.match('\\|\\|(.+)\\|\\|(?:_(\\d+))?$', expr.strip()):
        norm_result = parse_norm_latex(expr)
        if norm_result is not None:
            print(f'Parsed norm expression: {norm_result}')
            return norm_result
    has_derivatives = bool(re.search('\\\\frac\\{d[\\^]?', expr))
    if has_derivatives:
        print('Expression contains derivatives - will be handled during splitting or by LaTeX parser')
    func_patterns = ['([A-Z][A-Za-z]*)\\s*\\\\left\\s*\\((.*?\\\\begin\\{[pb]?matrix\\}.*?\\\\end\\{[pb]?matrix\\}.*?)\\s*\\\\right\\s*\\)', '([A-Z][A-Za-z]*)\\s*\\((.*?\\\\begin\\{[pb]?matrix\\}.*?\\\\end\\{[pb]?matrix\\}.*?)\\)', '([A-Z][A-Za-z]*)\\s*(\\\\begin\\{[pb]?matrix\\}.*?\\\\end\\{[pb]?matrix\\})']
    for pattern in func_patterns:
        func_match = re.match(pattern, expr.strip(), re.DOTALL)
        if func_match:
            func_name = func_match.group(1)
            matrix_content = func_match.group(2)
            print(f'Found custom function application: {func_name} to matrix')
            try:
                mathml = latex_to_mathml(matrix_content)
                matrix = parse_mathml_matrix(mathml)
                func = sympy.Function(func_name)
                result = func(matrix)
                print(f'Created SymPy expression: {result}')
                return result
            except Exception as e:
                print(f'Failed to parse custom function application: {e}')
                continue
    if '\\frac{' in expr and '^' in expr:
        expr = re.sub('\\\\frac\\{([^{}]*(?:\\{[^{}]*\\}[^{}]*)*)\\}\\{([^{}]*(?:\\{[^{}]*\\}[^{}]*)*)\\}\\s*\\^(\\{[^}]+\\}|\\w+)', '\\\\frac{\\1}{(\\2)^\\3}', expr)
    if expr.strip().startswith('\\det') and '\\begin{pmatrix}' in expr:
        try:
            print('Using improved matrix parsing to create SymPy expression...')
            det_match = re.match('\\\\det\\\\left\\((.*?)\\\\right\\)', expr.strip(), re.DOTALL)
            if det_match:
                matrix_content = det_match.group(1)
                print(f'DEBUG: Extracted matrix content: {matrix_content}')
                temp_transpiler = LaTeX2FutharkTranspiler()
                blocks = temp_transpiler.split_latex_matrix_algebra(matrix_content)
                print(f'DEBUG: Matrix blocks: {blocks}')
                fixed_blocks = []
                for i, (kind, content) in enumerate(blocks):
                    fixed_blocks.append((kind, content))
                    if i + 1 < len(blocks) and kind in ('matrix', 'matrixT') and (blocks[i + 1][0] in ('matrix', 'matrixT')):
                        fixed_blocks.append(('op', '@'))
                print(f'DEBUG: Fixed blocks: {fixed_blocks}')
                matrices = []
                operations = []
                for kind, content in fixed_blocks:
                    if kind in ('matrix', 'matrixT'):
                        mathml = latex_to_mathml(content)
                        matrix = parse_mathml_matrix(mathml)
                        if kind == 'matrixT':
                            matrix = sympy.Transpose(matrix)
                        matrices.append(matrix)
                    elif kind == 'op':
                        operations.append(content)
                if matrices:
                    result = matrices[0]
                    for i, matrix in enumerate(matrices[1:]):
                        if i < len(operations):
                            op = operations[i]
                            if op == '+':
                                result = result + matrix
                            elif op == '-':
                                result = result - matrix
                            elif op == '@':
                                result = result * matrix
                        else:
                            result = result * matrix
                    final_result = sympy.Determinant(result)
                    print(f'DEBUG: Created SymPy determinant expression: {final_result}')
                    return final_result
        except Exception as e:
            print(f'Improved matrix parsing failed: {e}')
            import traceback
            traceback.print_exc()
    try:
        from sympy.parsing.latex import parse_latex
        result = parse_latex(expr)
        if result is not None:
            return result
    except Exception as e:
        print(f'parse_latex failed: {e}')
    processed_expr = expr
    processed_expr = processed_expr.replace('\\arctan', 'atan')
    processed_expr = processed_expr.replace('\\arcsin', 'asin')
    processed_expr = processed_expr.replace('\\arccos', 'acos')
    processed_expr = processed_expr.replace('\\ln', 'log')
    processed_expr = processed_expr.replace('\\cdot', '*')
    processed_expr = re.sub('\\be\\^\\{([^}]+)\\}', 'exp(\\1)', processed_expr)
    processed_expr = re.sub('\\be\\^(\\S+)(?![a-zA-Z])', 'exp(\\1)', processed_expr)
    processed_expr = re.sub('\\\\(left|right|big|Big|bigg|Bigg)', '', processed_expr)
    processed_expr = re.sub('\\\\(quad|qquad)', ' ', processed_expr)
    processed_expr = processed_expr.replace('\\,', ' ').replace('\\;', ' ').replace('\\:', ' ').replace('\\!', '')

    def frac_replacer(match):
        num = match.group(1)
        den = match.group(2)
        return f'(({num})/({den}))'
    processed_expr = re.sub('\\\\frac\\{([^{}]*(?:\\{[^{}]*\\}[^{}]*)*)\\}\\{([^{}]*(?:\\{[^{}]*\\}[^{}]*)*)\\}', frac_replacer, processed_expr)
    processed_expr = re.sub('\\\\sqrt\\{([^}]+)\\}', 'sqrt(\\1)', processed_expr)

    def sum_replacer(match):
        bounds = match.group(1)
        expr_part = match.group(2) if len(match.groups()) > 1 else ''
        if '=' in bounds:
            var_part, rest = bounds.split('=', 1)
            if '}^{' in rest:
                start, end = rest.split('}^{', 1)
                end = end.rstrip('}')
                return f'Sum({expr_part}, ({var_part.strip()}, {start}, {end}))'
        return match.group(0)
    processed_expr = re.sub('\\\\sum_\\{([^}]+)\\}\\s*([^+\\-]*?)(?=[\\s+\\-]|$)', sum_replacer, processed_expr)
    processed_expr = re.sub('\\\\[a-zA-Z]+', '', processed_expr)
    try:
        from sympy import sin, cos, tan, log, exp, sqrt, atan, asin, acos, sinh, cosh, tanh, pi, E, I, Symbol, Sum
        namespace = {'sin': sin, 'cos': cos, 'tan': tan, 'log': log, 'exp': exp, 'sqrt': sqrt, 'atan': atan, 'asin': asin, 'acos': acos, 'arctan': atan, 'arcsin': asin, 'arccos': acos, 'sinh': sinh, 'cosh': cosh, 'tanh': tanh, 'pi': pi, 'e': E, 'E': E, 'I': I, 'Sum': Sum}
        for match in re.finditer('\\b([a-zA-Z_]\\w*)\\b', processed_expr):
            var_name = match.group(1)
            if var_name not in namespace and (not var_name.isdigit()):
                namespace[var_name] = Symbol(var_name)
        result = sympify(processed_expr, locals=namespace)
        print(f'PROCESS_SPLIT: sympify returned: {result}')
        print(f'PROCESS_SPLIT: result type: {type(result)}')
        if hasattr(result, 'func'):
            print(f'PROCESS_SPLIT: result.func: {result.func}')
        return result
    except Exception as e:
        print(f'sympify failed: {e}')
        if '(' in processed_expr and 'atan(' in processed_expr:
            try:
                clean_expr = processed_expr.replace('\\', '')
                result = sympify(clean_expr, locals=namespace)
                return result
            except Exception as e2:
                print(f'Mixed notation parsing also failed: {e2}')
    try:
        from sympy.parsing.latex import parse_latex
        return parse_latex(original_expr)
    except:
        raise ValueError(f'Cannot parse LaTeX expression: {original_expr}')

def process_split_part(part: str):
    import re
    import sympy
    from sympy import sympify
    print(f'Processing split part: {part}')
    has_derivatives = bool(re.search('\\\\frac\\{d[\\^]?', part))
    if has_derivatives:
        complex_structures = ['\\\\sum_{', '\\\\int_{', '\\\\det\\\\left', '\\\\begin\\{pmatrix\\}']
        has_complex_structures = any((re.search(pattern, part) for pattern in complex_structures))
        if not has_complex_structures:
            print(f'Processing isolated derivative part: {part}')
            try:
                processed = preprocess_derivatives_in_expression(part)
                print(f'Derivative preprocessing result: {processed}')
                if '\\frac{d' not in processed:
                    try:
                        namespace = {'sin': sympy.sin, 'cos': sympy.cos, 'tan': sympy.tan, 'log': sympy.log, 'exp': sympy.exp, 'sqrt': sympy.sqrt, 'atan': sympy.atan, 'asin': sympy.asin, 'acos': sympy.acos, 'pi': sympy.pi, 'e': sympy.E, 'E': sympy.E, 'I': sympy.I}
                        for match in re.finditer('\\b([a-zA-Z_]\\w*)\\b', processed):
                            var_name = match.group(1)
                            if var_name not in namespace and (not var_name.isdigit()):
                                namespace[var_name] = sympy.Symbol(var_name)
                        result = sympify(processed, locals=namespace)
                        print(f'Successfully parsed derivative part as SymPy: {result}')
                        return result
                    except Exception as e:
                        print(f'SymPy parsing of preprocessed derivative failed: {e}')
            except Exception as e:
                print(f'Derivative preprocessing failed: {e}')
        else:
            print(f'Part has derivatives mixed with complex structures - using LaTeX parser')
    try:
        return parse_any_latex(part)
    except Exception as e:
        print(f'CRITICAL: Failed to parse: {part}')
        if '\\det' in part:
            print('This is a determinant expression that should have been handled!')
        return sympy.Symbol(f'UNPARSED_{hash(part) % 1000}')

def parse_derivative_latex(expr):
    import re
    m = re.match('\\\\frac\\{d\\^(\\d+)\\}\\{d([a-zA-Z]+)\\^\\1\\}\\s*(.+)', expr)
    if m:
        order = int(m.group(1))
        var = m.group(2)
        func_expr = m.group(3)
        var_sym = sympy.Symbol(var)
        try:
            clean_func = func_expr.strip().replace('\\arctan', 'atan')
            func_sym = sympify(clean_func)
        except:
            func_sym = parse_any_latex(func_expr.strip())
        try:
            result = sympy.diff(func_sym, var_sym, order)
            return result
        except Exception:
            return sympy.Derivative(func_sym, var_sym, order)
    m = re.match('\\\\frac\\{d\\}\\{d([a-zA-Z]+)\\}\\s*(.+)', expr)
    if m:
        var = m.group(1)
        func_expr = m.group(2)
        var_sym = sympy.Symbol(var)
        try:
            clean_func = func_expr.strip().replace('\\arctan', 'atan')
            func_sym = sympify(clean_func)
        except:
            func_sym = parse_any_latex(func_expr.strip())
        try:
            result = sympy.diff(func_sym, var_sym, 1)
            return result
        except Exception:
            return sympy.Derivative(func_sym, var_sym, 1)
    pat_ord = re.compile('\\\\frac\\s*\\{\\s*d(?:\\^(\\d+))?\\s*\\}\\s*\\{\\s*d\\s*([a-zA-Z]+)(?:\\^(\\d+))?\\s*\\}\\s*(.+)')
    pat_partial = re.compile('\\\\frac\\s*\\{\\s*\\\\partial(?:\\^(\\d+))?\\s*\\}\\s*\\{\\s*\\\\partial\\s*([a-zA-Z]+)(?:\\^(\\d+))?\\s*\\}\\s*(.+)')
    m = pat_ord.match(expr)
    m_partial = pat_partial.match(expr)
    if m:
        n1, var, n2, fx = m.groups()
        n = n1 or n2 or '1'
        if (n1 and n2) and n1 != n2:
            raise ValueError('Mismatch in derivative order for d/dx')
        var_sym = symbols(var)
        fx_sym = parse_derivative_latex(fx)
        try:
            result = sympy.diff(fx_sym, var_sym, int(n))
            return result
        except Exception:
            return Derivative(fx_sym, var_sym, int(n))
    elif m_partial:
        n1, var, n2, fx = m_partial.groups()
        n = n1 or n2 or '1'
        if (n1 and n2) and n1 != n2:
            raise ValueError('Mismatch in derivative order for partial/partial x')
        var_sym = symbols(var)
        fx_sym = parse_derivative_latex(fx)
        try:
            result = sympy.diff(fx_sym, var_sym, int(n))
            return result
        except Exception:
            return Derivative(fx_sym, var_sym, int(n))
    else:
        return parse_any_latex(expr)

def split_latex_robust(expr: str):
    import re
    print(f'Splitting expression: {expr}')
    if '\\det' in expr and '\\begin{pmatrix}' in expr:
        print(f'BACKSLASH CHECK: Found determinant with matrix')
        print(f'  Double backslashes (\\\\): {expr.count(chr(92) + chr(92))}')
        print(f'  Single backslashes (\\): {expr.count(chr(92))}')
        matrix_sections = re.findall('\\\\begin\\{pmatrix\\}.*?\\\\end\\{pmatrix\\}', expr)
        for i, section in enumerate(matrix_sections):
            print(f'  Matrix {i}: {section}')
            if '\\\\' not in section and '&' in section:
                print(f'    WARNING: Matrix {i} missing double backslashes!')
    split_positions = []
    depth = 0
    brace_depth = 0
    i = 0
    n = len(expr)
    while i < n:
        c = expr[i]
        if expr[i:i + 6] == '\\left(':
            depth += 1
            i += 6
            continue
        elif expr[i:i + 7] == '\\right)':
            depth -= 1
            i += 7
            continue
        if c == '{':
            brace_depth += 1
        elif c == '}':
            brace_depth -= 1
        elif c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        if depth == 0 and brace_depth == 0 and (c in '+-') and (i > 0):
            prev_context = expr[max(0, i - 10):i]
            next_context = expr[i:min(len(expr), i + 10)]
            in_derivative = '\\frac{d' in prev_context and (not '}' in prev_context[prev_context.rfind('\\frac{d'):])
            at_start_of_expression = expr[i - 1] in '(+=' or (i > 5 and expr[i - 6:i] in ['\\left(', '\\right)'])
            if not in_derivative and (not at_start_of_expression):
                split_positions.append(i)
        i += 1
    if not split_positions:
        return [('sympy', expr)]
    parts = []
    prev = 0
    for pos in split_positions:
        part = expr[prev:pos].strip()
        if part:
            parts.append(('sympy', part))
        parts.append(('op', expr[pos]))
        prev = pos + 1
    final_part = expr[prev:].strip()
    if final_part:
        parts.append(('sympy', final_part))
    print(f'Split into {len(parts)} parts: {parts}')
    return parts

def preprocess_derivatives_in_expression(expr: str) -> str:
    import re
    print(f'Preprocessing derivatives in: {expr}')
    if not expr:
        return expr
    try:
        has_derivatives = bool(re.search('\\\\frac\\{d[\\^]?', expr))
        if not has_derivatives:
            return expr

        def fix_exponents(expr):
            expr = re.sub('([^}])\\^\\{([^}]+)\\}', '\\1**(\\2)', expr)
            expr = re.sub('([a-zA-Z0-9_)\\]])\\^([a-zA-Z0-9_]+)', '\\1**\\2', expr)
            expr = expr.replace('^', '**')
            return expr

        def fix_implicit_multiplication(expr):
            expr = re.sub('\\b([a-zA-Z_]\\w*)\\s+([a-zA-Z_]\\w*)\\b', '\\1 * \\2', expr)
            expr = re.sub('\\b(\\d+)\\s*([a-zA-Z_]\\w*)\\b', '\\1*\\2', expr)
            expr = re.sub('\\)\\s*\\(', ')*(', expr)
            expr = re.sub('([a-zA-Z_]\\w*\\([^)]+\\))([a-zA-Z_]\\w*\\([^)]+\\))', '\\1*\\2', expr)
            return expr

        def find_derivative_end(text, start_pos):
            pos = start_pos
            brace_depth = 0
            paren_depth = 0
            while pos < len(text) and text[pos].isspace():
                pos += 1
            while pos < len(text):
                c = text[pos]
                remaining = text[pos:]
                function_skipped = False
                for func_name in ['\\arctan', '\\arcsin', '\\arccos', '\\sin', '\\cos', '\\tan', '\\log', '\\ln', '\\exp', '\\sqrt', '\\frac']:
                    if remaining.startswith(func_name):
                        pos += len(func_name)
                        function_skipped = True
                        break
                if function_skipped:
                    continue
                if c == '{':
                    brace_depth += 1
                elif c == '}':
                    if brace_depth > 0:
                        brace_depth -= 1
                    else:
                        return pos
                elif c == '(':
                    paren_depth += 1
                elif c == ')':
                    if paren_depth > 0:
                        paren_depth -= 1
                elif brace_depth == 0 and paren_depth == 0:
                    if c in '+-':
                        if pos > start_pos:
                            prev_char_pos = pos - 1
                            while prev_char_pos >= start_pos and text[prev_char_pos].isspace():
                                prev_char_pos -= 1
                            if prev_char_pos >= start_pos and text[prev_char_pos] in '^_':
                                pass
                            else:
                                return pos
                pos += 1
            return len(text)

        def find_matching_brace(text, start_pos):
            if start_pos >= len(text) or text[start_pos] != '{':
                return -1
            brace_count = 1
            pos = start_pos + 1
            while pos < len(text) and brace_count > 0:
                if text[pos] == '{':
                    brace_count += 1
                elif text[pos] == '}':
                    brace_count -= 1
                pos += 1
            return pos - 1 if brace_count == 0 else -1
        processed = expr
        derivatives_found = []
        nth_pattern = '\\\\frac\\{d\\^(\\d+)\\}\\{d([a-zA-Z]+)\\^(\\d+)\\}'
        max_iterations = 10
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            match = re.search(nth_pattern, processed)
            if not match:
                break
            try:
                order1, var, order2 = match.groups()
                if order1 != order2:
                    print(f'Order mismatch: {order1} != {order2}')
                    break
                order = int(order1)
                derivative_start = match.end()
                derivative_end = find_derivative_end(processed, derivative_start)
                func_expr = processed[derivative_start:derivative_end].strip()
                print(f"Found nth-order derivative: order={order}, var={var}, func='{func_expr}'")
                if func_expr:
                    computed = compute_single_derivative(func_expr, var, order, fix_exponents, fix_implicit_multiplication)
                    if computed:
                        full_match = processed[match.start():derivative_end]
                        processed = processed.replace(full_match, computed, 1)
                        derivatives_found.append((full_match, computed))
                        print(f'Replaced derivative, result: {processed}')
                    else:
                        print('Failed to compute derivative, stopping')
                        break
                else:
                    print('Empty function expression, stopping')
                    break
            except Exception as e:
                print(f'Error processing nth-order derivative: {e}')
                break
        first_pattern = '\\\\frac\\{d\\}\\{d([a-zA-Z]+)\\}'
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            match = re.search(first_pattern, processed)
            if not match:
                break
            try:
                var = match.group(1)
                derivative_start = match.end()
                derivative_end = find_derivative_end(processed, derivative_start)
                func_expr = processed[derivative_start:derivative_end].strip()
                print(f"Found first-order derivative: var={var}, func='{func_expr}'")
                if func_expr:
                    computed = compute_single_derivative(func_expr, var, 1, fix_exponents, fix_implicit_multiplication)
                    if computed:
                        full_match = processed[match.start():derivative_end]
                        processed = processed.replace(full_match, computed, 1)
                        derivatives_found.append((full_match, computed))
                        print(f'Replaced derivative, result: {processed}')
                    else:
                        print('Failed to compute derivative, stopping')
                        break
                else:
                    print('Empty function expression, stopping')
                    break
            except Exception as e:
                print(f'Error processing first-order derivative: {e}')
                break
        if derivatives_found:
            print('Doing careful cleanup after derivative processing')
            processed = processed.replace('\\arctan', 'atan')
            processed = processed.replace('\\sin', 'sin')
            processed = processed.replace('\\cos', 'cos')
            processed = processed.replace('\\tan', 'tan')
            processed = processed.replace('\\ln', 'log')
            processed = processed.replace('\\cdot', ' * ')
            processed = re.sub('\\^\\{([^}]+)\\}', '**(\\1)', processed)
            processed = re.sub('\\^([a-zA-Z0-9]+)', '**\\1', processed)
            processed = re.sub('([a-zA-Z_]\\w*\\([^)]*\\))\\s*([a-zA-Z_]\\w*\\([^)]*\\))', '\\1 * \\2', processed)
            sqrt_pos = 0
            while True:
                sqrt_match = processed.find('\\sqrt', sqrt_pos)
                if sqrt_match == -1:
                    break
                brace_start = sqrt_match + 5
                while brace_start < len(processed) and processed[brace_start].isspace():
                    brace_start += 1
                if brace_start < len(processed) and processed[brace_start] == '{':
                    brace_end = find_matching_brace(processed, brace_start)
                    if brace_end != -1:
                        content = processed[brace_start + 1:brace_end]
                        replacement = f'sqrt({content})'
                        processed = processed[:sqrt_match] + replacement + processed[brace_end + 1:]
                        sqrt_pos = sqrt_match + len(replacement)
                    else:
                        sqrt_pos = sqrt_match + 5
                else:
                    sqrt_pos = sqrt_match + 5

            def frac_replacer(match):
                num = match.group(1)
                den = match.group(2)
                return f'(({num})/({den}))'
            processed = re.sub('\\\\frac\\{([^{}]*(?:\\{[^{}]*\\}[^{}]*)*)\\}\\{([^{}]*(?:\\{[^{}]*\\}[^{}]*)*)\\}', frac_replacer, processed)
            processed = fix_implicit_multiplication(processed)
            open_parens = processed.count('(')
            close_parens = processed.count(')')
            if open_parens > close_parens:
                processed += ')' * (open_parens - close_parens)
        print(f'Final preprocessed result: {processed}')
        return processed
    except Exception as e:
        print(f'Error in preprocess_derivatives_in_expression: {e}')
        return expr

def compute_single_derivative(func_expr, var, order, fix_exponents, fix_implicit_multiplication):
    try:
        print(f"Computing {order}-order derivative of '{func_expr}' w.r.t. {var}")
        clean_func = func_expr.strip()
        open_parens = clean_func.count('(')
        close_parens = clean_func.count(')')
        if open_parens > close_parens:
            clean_func += ')' * (open_parens - close_parens)
        clean_func = clean_func.replace('\\arctan', 'atan').replace('\\ln', 'log').replace('\\sin', 'sin').replace('\\cos', 'cos').replace('\\tan', 'tan')

        def frac_replacer(match):
            num = match.group(1)
            den = match.group(2)
            return f'(({num})/({den}))'
        clean_func = re.sub('\\\\frac\\{([^{}]*(?:\\{[^{}]*\\}[^{}]*)*)\\}\\{([^{}]*(?:\\{[^{}]*\\}[^{}]*)*)\\}', frac_replacer, clean_func)
        clean_func = fix_exponents(clean_func)
        clean_func = fix_implicit_multiplication(clean_func)
        print(f'Cleaned function: {clean_func}')
        var_sym = sympy.Symbol(var)
        try:
            namespace = {'atan': sympy.atan, 'sin': sympy.sin, 'cos': sympy.cos, 'tan': sympy.tan, 'log': sympy.log, 'exp': sympy.exp, 'sqrt': sympy.sqrt, 'asin': sympy.asin, 'acos': sympy.acos, 'x': sympy.Symbol('x'), 'y': sympy.Symbol('y'), 'z': sympy.Symbol('z'), 't': sympy.Symbol('t'), 'n': sympy.Symbol('n'), var: var_sym}
            func_sym = sympify(clean_func, locals=namespace)
            print(f'Parsed function: {func_sym}')
        except Exception as e:
            print(f"Failed to parse function '{clean_func}': {e}")
            return None
        try:
            result = sympy.diff(func_sym, var_sym, order)
            print(f'Derivative result: {result}')
            result_str = str(result)
            if any((op in result_str for op in ['+', '-', '*', '/'])):
                result_str = f'({result_str})'
            return result_str
        except Exception as e:
            print(f'Failed to compute derivative: {e}')
            return None
    except Exception as e:
        print(f'Overall derivative computation failed: {e}')
        return None

def strip_ns(tag: str) -> str:
    return tag.split('}')[-1]

def parse_mathml_cell(mtd: ET.Element) -> list[sympy.Expr]:
    raw = ''.join(mtd.itertext())
    pieces = [clean_mathml_text(t) for t in raw.split() if clean_mathml_text(t)]
    exprs: list[sympy.Expr] = []
    for tok in pieces or ['0']:
        try:
            exprs.append(sympy.sympify(tok))
        except Exception:
            exprs.append(sympy.Symbol(tok))
    return exprs

def clean_mathml_text(text: str) -> str:
    if not text:
        return ''
    return ''.join((c for c in text if c not in ['\u2062', '\u2062', '\u2061', '\u200b'])).strip()

def mathml_to_sympy(elem: ET.Element) -> sympy.Expr:
    tag = strip_ns(elem.tag)
    if tag in ('math', 'mrow'):
        children = [c for c in elem if isinstance(c.tag, str)]
        for i, child in enumerate(children):
            if strip_ns(child.tag) == 'munderover':
                op_elem = child[0]
                op_text = clean_mathml_text(op_elem.text or '')
                if op_text in ('∑', 'Σ', '∏', 'Π'):
                    lower_elem = child[1]
                    upper_elem = child[2]
                    lower = mathml_to_sympy(lower_elem)
                    upper = mathml_to_sympy(upper_elem)
                    remaining_children = children[i + 1:]
                    if remaining_children:
                        summand_elem = ET.Element('mrow')
                        for rc in remaining_children:
                            summand_elem.append(rc)
                        summand = mathml_to_sympy(summand_elem)
                    else:
                        raise ValueError('Sum/Product without expression')
                    var = None
                    lower_val = None
                    if isinstance(lower, sympy.Symbol):
                        var = lower
                        if i + 1 < len(children) and strip_ns(children[i + 1].tag) == 'mo' and (clean_mathml_text(children[i + 1].text) == '='):
                            if i + 2 < len(children):
                                lower_val = mathml_to_sympy(children[i + 2])
                                remaining_children = children[i + 3:]
                        else:
                            lower_val = sympy.Integer(1)
                    elif hasattr(lower, 'free_symbols') and len(lower.free_symbols) == 1:
                        var = list(lower.free_symbols)[0]
                        lower_val = lower
                    else:
                        var = sympy.Symbol('n')
                        lower_val = lower
                    if op_text in ('∑', 'Σ'):
                        return sympy.Sum(summand, (var, lower_val, upper))
                    else:
                        return sympy.Product(summand, (var, lower_val, upper))
    if tag == 'mn':
        txt = clean_mathml_text(elem.text)
        if '.' in txt:
            return sympy.Float(txt)
        else:
            return sympy.Integer(txt)
    elif tag in ('mi', 'ci'):
        txt = clean_mathml_text(elem.text)
        if txt == 'e':
            return sympy.E
        elif txt in ('π', 'pi'):
            return sympy.pi
        elif txt == 'i':
            return sympy.I
        else:
            return sympy.Symbol(txt)
    elif tag == 'mfrac':
        num = mathml_to_sympy(elem[0])
        den = mathml_to_sympy(elem[1])
        return num / den
    elif tag == 'msup':
        base = mathml_to_sympy(elem[0])
        exp = mathml_to_sympy(elem[1])
        return sympy.Pow(base, exp)
    elif tag == 'msqrt':
        return sympy.sqrt(mathml_to_sympy(elem[0]))
    elif tag == 'mtable':
        ns = '{http://www.w3.org/1998/Math/MathML}'
        rows = []
        for mtr in elem.findall(f'.//{ns}mtr'):
            row = []
            for mtd in mtr.findall(f'{ns}mtd'):
                if len(mtd) > 0:
                    cell_value = mathml_to_sympy(mtd[0])
                else:
                    cell_text = clean_mathml_text(mtd.text or '0')
                    cell_value = sympy.sympify(cell_text)
                row.append(cell_value)
            rows.append(row)
        return sympy.Matrix(rows)
    elif tag == 'mo':
        return clean_mathml_text(elem.text)
    elif tag in ('mrow', 'math'):
        return parse_mathml_expression(elem)
    text = clean_mathml_text(''.join(elem.itertext()))
    try:
        return sympy.sympify(text)
    except:
        return sympy.Symbol(text)

def parse_mathml_expression(elem):
    children = [c for c in elem if isinstance(c.tag, str)]
    if not children:
        return sympy.Integer(0)
    if len(children) == 1:
        return mathml_to_sympy(children[0])
    tokens = []
    for child in children:
        tag = strip_ns(child.tag)
        if tag == 'mo':
            op = clean_mathml_text(child.text)
            tokens.append(('op', op))
        else:
            expr = mathml_to_sympy(child)
            tokens.append(('expr', expr))
    if not tokens:
        return sympy.Integer(0)
    result = None
    i = 0
    while i < len(tokens) and result is None:
        if tokens[i][0] == 'expr':
            result = tokens[i][1]
            i += 1
            break
        i += 1
    if result is None:
        return sympy.Integer(0)
    while i < len(tokens) - 1:
        if tokens[i][0] == 'op' and i + 1 < len(tokens) and (tokens[i + 1][0] == 'expr'):
            op = tokens[i][1]
            rhs = tokens[i + 1][1]
            if op == '+':
                result = result + rhs
            elif op == '-':
                result = result - rhs
            elif op in ('*', '·', '×', '⋅'):
                result = result * rhs
            elif op == '/':
                result = result / rhs
            elif op in ('^', '**'):
                result = result ** rhs
            i += 2
        else:
            i += 1
    return result

class LaTeX2FutharkTranspiler:
    EXCLUDED_SYMBOLS = {'e', 'd', 'dx', 'diff', 'operatorname'}

    def __init__(self):
        self.handlers = {}
        self.register_default_handlers()
        self.bound_variables = set()
        self.RESERVED: set[str] = {'e', 'E', 'pi', 'Pi', 'i', 'I', 'dx', 'dy', 'dz', 'left', 'right', 'frac', 'begin', 'end', 'cdot', 'pmatrix'}
        self.sympy_futhark_funcs = {sin: 'f64.sin', cos: 'f64.cos', tan: 'f64.tan', sympy.cot: '\\x -> 1.0 / f64.tan x', sympy.sec: '\\x -> 1.0 / f64.cos x', sympy.csc: '\\x -> 1.0 / f64.sin x', asin: 'f64.asin', acos: 'f64.acos', atan: 'f64.atan', sympy.atan2: 'f64.atan2', sinh: 'f64.sinh', cosh: 'f64.cosh', tanh: 'f64.tanh', exp: 'f64.exp', log: 'f64.log', sympy.ln: 'f64.log', sqrt: 'f64.sqrt', sympy.Abs: 'f64.abs', sympy.sign: 'f64.sgn', sympy.ceiling: 'f64.ceil', sympy.floor: 'f64.floor', sympy.pi: 'f64.pi', sympy.E: 'f64.e', sympy.Min: 'f64.min', sympy.Max: 'f64.max'}
        self.clean_operators = {'+', '-', '*', '/'}
        self.symbols = {}
        self.symbol_list = []
        self.current_indices = set()
        self.minimize_parentheses = True
        self.sympy_futhark_funcs[sympy.E] = 'f64.e'
        self.sympy_futhark_funcs[sympy.Function('SVD')] = 'svd_default'
        self.sympy_futhark_funcs[sympy.Function('svd')] = 'svd_default'

    def register_handler(self, expr_type: type, handler: Callable):
        self.handlers[expr_type] = handler

    def register_default_handlers(self):
        self.register_handler(sympy.Symbol, self.handle_symbol)
        self.register_handler(sympy.Integer, self.handle_number)
        self.register_handler(sympy.Float, self.handle_number)
        self.register_handler(sympy.Rational, self.handle_rational)
        self.register_handler(sympy.Add, self.handle_add)
        self.register_handler(sympy.Mul, self.handle_mul)
        self.register_handler(sympy.Pow, self.handle_pow)
        self.register_handler(sympy.Function, self.handle_function)
        self.register_handler(Sum, self.handle_sum)
        self.register_handler(Integral, self.handle_integral)
        self.register_handler(sympy.Piecewise, self.handle_piecewise)
        self.register_handler(sympy.Derivative, self.handle_derivative)
        self.register_handler(Product, self.handle_product)
        self.register_handler(sympy.Matrix, self.handle_matrix)
        self.register_handler(sympy.ImmutableMatrix, self.handle_matrix)
        self.register_handler(sympy.ImmutableDenseMatrix, self.handle_matrix)
        self.register_handler(sympy.MutableMatrix, self.handle_matrix)
        self.register_handler(sympy.MatrixSymbol, self.handle_matrix)
        self.register_handler(sympy.MatMul, self.handle_matrix_multiplication)
        self.register_handler(sympy.MatAdd, self.handle_matrix_addition)
        self.register_handler(sympy.Determinant, self.handle_determinant)
        self.register_handler(sympy.Transpose, self.handle_transpose)
        from sympy.core.numbers import Exp1
        self.register_handler(Exp1, lambda _: 'f64.e')
        self.register_handler(sympy.pi.__class__, lambda _: 'f64.pi')

    def register_symbol(self, s, type_='f64'):
        name = str(s)
        if not name:
            return
        if name in self.EXCLUDED_SYMBOLS:
            return
        if name not in self.symbols:
            self.symbols[name] = type_
            if name not in self.symbol_list:
                self.symbol_list.append(name)
        else:
            current_type_str = self.symbols[name]
            proposed_type_str = type_
            type_rank_map = {'f64': 0, '[]f64': 1, '[][]f64': 2}
            current_rank = type_rank_map.get(current_type_str, -1)
            proposed_rank = type_rank_map.get(proposed_type_str, -1)
            if proposed_rank > current_rank:
                self.symbols[name] = proposed_type_str
            if name not in self.symbol_list:
                self.symbol_list.append(name)

    def get_safe_variable_name(self, base_name, avoid_names=None):
        if avoid_names is None:
            avoid_names = set(self.symbols.keys()) | self.bound_variables
        if base_name not in avoid_names:
            return base_name
        safe_name = f'{base_name}_'
        if safe_name not in avoid_names:
            return safe_name
        i = 1
        while f'{base_name}_{i}' in avoid_names:
            i += 1
        return f'{base_name}_{i}'

    def sympy_matrix_to_futhark(self, matrix) -> str:
        rows = [f'[{', '.join(map(str, row))}]' for row in matrix.tolist()]
        return f'[{', '.join(rows)}]'

    def split_latex_matrix_algebra(self, tex: str):
        blocks = []
        i, N = (0, len(tex))
        while i < N:
            m0 = tex.find('\\begin{pmatrix}', i)
            if m0 == -1:
                break
            pre = tex[i:m0].rstrip()
            if blocks and pre:
                op = pre[-1]
                if op in '+-':
                    blocks.append(('op', op))
                else:
                    blocks.append(('op', '@'))
            level, j = (0, m0)
            while j < N:
                if tex.startswith('\\begin{pmatrix}', j):
                    level += 1
                    j += len('\\begin{pmatrix}')
                elif tex.startswith('\\end{pmatrix}', j):
                    level -= 1
                    j += len('\\end{pmatrix}')
                    if level == 0:
                        break
                else:
                    j += 1
            mat_src = tex[m0:j]
            k = j
            while k < N and tex[k].isspace():
                k += 1
            if tex.startswith('\\right)', k):
                k += len('\\right)')
                while k < N and tex[k].isspace():
                    k += 1
            transposed = False
            if tex.startswith('^T', k):
                transposed = True
                k += 2
                while k < N and tex[k].isspace():
                    k += 1
            kind = 'matrixT' if transposed else 'matrix'
            blocks.append((kind, mat_src))
            i = k
        return blocks

    def promote_frac_derivs(self, expr):

        def _replace(node):
            if not isinstance(node, Mul):
                return node
            num, den = node.as_numer_denom()
            num_facs = Mul.make_args(num)
            den_facs = Mul.make_args(den)
            if len(num_facs) != 2 or len(den_facs) != 1:
                return node
            d_pow_val = None
            func_expr = None
            for fac in num_facs:
                if isinstance(fac, Pow) and isinstance(fac.base, Symbol) and (fac.base.name == 'd') and isinstance(fac.exp, Integer) and (fac.exp > 0):
                    d_pow_val = int(fac.exp)
                else:
                    func_expr = fac
            if d_pow_val is None or func_expr is None:
                return node
            dv_fac = den_facs[0]
            if not (isinstance(dv_fac, Pow) and isinstance(dv_fac.base, Symbol) and dv_fac.base.name.startswith('d') and (len(dv_fac.base.name) > 1) and isinstance(dv_fac.exp, Integer) and (int(dv_fac.exp) == d_pow_val)):
                return node
            var_name = dv_fac.base.name[1:]
            return Derivative(func_expr, Symbol(var_name), d_pow_val)
        return expr.replace(lambda n: isinstance(n, Mul), _replace)

    def _omit(self, expr) -> str:
        return ''

    def handle_symbol(self, expr):
        try:
            name = str(expr)
        except Exception:
            return 'unknown_symbol'
        if name == 'e':
            return 'f64.e'
        if name == 'pi':
            return 'f64.pi'
        if name in self.current_indices:
            return f'f64.i64 {name}'
        return name

    def handle_number(self, expr: Union[sympy.Integer, sympy.Float]) -> str:
        if isinstance(expr, sympy.Integer):
            return f'{expr}.0'
        return str(expr)

    def handle_rational(self, expr: sympy.Rational) -> str:
        return f'({expr.p}.0 / {expr.q}.0)'

    def handle_add(self, expr: sympy.Add) -> str:
        terms = [self.sympy_to_futhark(arg) for arg in expr.args]
        return f'({' + '.join(terms)})'

    def handle_mul(self, expr: sympy.Mul) -> str:
        numerator_parts = []
        denominator_parts = []
        for arg in expr.args:
            if isinstance(arg, sympy.Pow) and arg.args[1] == -1:
                denominator_parts.append(arg.args[0])
            else:
                numerator_parts.append(arg)
        if denominator_parts:
            if not numerator_parts:
                numerator_parts.append(sympy.Integer(1))
            if len(numerator_parts) == 1:
                numerator = numerator_parts[0]
            else:
                numerator = sympy.Mul(*numerator_parts)
            if len(denominator_parts) == 1:
                denominator = denominator_parts[0]
            else:
                denominator = sympy.Mul(*denominator_parts)
            num_str = self.sympy_to_futhark(numerator)
            den_str = self.sympy_to_futhark(denominator)
            return f'({num_str} / {den_str})'
        factors = [self.sympy_to_futhark(a) for a in expr.args]
        result = f'({' * '.join(factors)})'
        print(f'MUL DEBUG: {expr} -> {result}')
        return result

    def handle_pow(self, expr):
        base, exponent = expr.args
        base_s = self.sympy_to_futhark(base)
        exp_s = self.sympy_to_futhark(exponent)
        is_integer_exp = False
        exp_var_name = None
        
        if isinstance(exponent, sympy.Symbol):
            exp_var_name = str(exponent)
            if exp_var_name in self.symbols and self.symbols[exp_var_name] == "i64":
                is_integer_exp = True
            elif exp_var_name in self.current_indices:
                is_integer_exp = True
        elif isinstance(exponent, sympy.Integer):
            is_integer_exp = True
        
        if is_integer_exp and hasattr(base, 'is_negative') and base.is_negative:
            if isinstance(base, (sympy.Integer, sympy.Float, sympy.Rational)):
                base_val = float(base)
                abs_base = abs(base_val)
                
                if isinstance(exponent, sympy.Integer):
                    exp_val = int(exponent)
                    result = abs_base ** exp_val
                    if exp_val % 2 == 1:
                        result = -result
                    return f"{result}"
                else:
                    if base_val == -1.0:
                        return f"(if {exp_var_name} % 2i64 == 0i64 then 1.0 else -1.0)"
                    else:
                        return f"(let abs_result = {abs_base} ** f64.i64 {exp_var_name} in " \
                                f"if {exp_var_name} % 2i64 == 0i64 then abs_result else -abs_result)"
            else:
                return f"""(let base = {base_s} in
                        if base >= 0.0 then base ** f64.i64 {exp_var_name}
                        else let abs_result = (-base) ** f64.i64 {exp_var_name} in
                                if {exp_var_name} % 2i64 == 0i64 then abs_result else -abs_result)"""
        return f"({base_s} ** {exp_s})"

    def _fixup_derivative_literals(self, expr):
        if isinstance(expr, sympy.Symbol):
            name = str(expr)
            if name == 'derivative':
                return sympy.Integer(0)
            return expr
        if hasattr(expr, 'args') and hasattr(expr, 'func'):
            try:
                fixed_args = []
                for arg in expr.args:
                    try:
                        fixed_arg = self._fixup_derivative_literals(arg)
                        fixed_args.append(fixed_arg)
                    except Exception:
                        fixed_args.append(arg)
                if isinstance(expr, sympy.Integer):
                    return expr
                elif isinstance(expr, sympy.Add):
                    return sympy.Add(*fixed_args)
                elif isinstance(expr, sympy.Mul):
                    return sympy.Mul(*fixed_args)
                elif isinstance(expr, sympy.Pow):
                    if len(fixed_args) == 2:
                        return sympy.Pow(fixed_args[0], fixed_args[1])
                    return expr
                else:
                    if any((fixed_args[i] is not expr.args[i] for i in range(len(fixed_args)))):
                        return expr.func(*fixed_args)
                    return expr
            except Exception as e:
                return expr
        return expr

    def latex_to_sympy(self, latex_expr: str):
        import re
        safe = self.preprocess_derivatives(latex_expr)
        if '\\nabla' in safe:
            nabla_result = parse_nabla_latex(safe)
            if nabla_result is not None:
                self._collect_symbols_preorder(nabla_result)
                return nabla_result
        if '||' in safe and re.match('\\|\\|(.+)\\|\\|(?:_(\\d+))?$', safe.strip()):
            norm_result = parse_norm_latex(safe)
            if norm_result is not None:
                self._collect_symbols_preorder(norm_result)
                return norm_result
        try:
            expr = parse_derivative_latex(safe)
        except Exception as err:
            raise ValueError(f'parse_latex failed: {err}\nLaTeX after preprocessing was:\n{safe}') from None
        expr = self.promote_frac_derivs(expr)
        self._collect_symbols_preorder(expr)
        return expr

    def handle_division(self, numerator, denominator):
        numer_str = self.sympy_to_futhark(numerator)
        denom_str = self.sympy_to_futhark(denominator)
        return f'({numer_str} / {denom_str})'

    def handle_function(self, expr: sympy.Function) -> str:
        print(f'HANDLE_FUNC: processing {expr}')
        print(f'HANDLE_FUNC: expr.func = {expr.func}')
        print(f'HANDLE_FUNC: type(expr.func) = {type(expr.func)}')
        print(f'HANDLE_FUNC: str(expr.func) = {str(expr.func)}')
        print(f'HANDLE_FUNC: expr.args = {expr.args}')
        func_obj = expr.func
        if func_obj in self.sympy_futhark_funcs:
            func_name = self.sympy_futhark_funcs[func_obj]
            print(f'HANDLE_FUNC: found in direct lookup: {func_name}')
        else:
            print(f'HANDLE_FUNC: not found in direct lookup, trying string matching')
            func_str = str(func_obj).lower()
            if '.' in func_str:
                func_str = func_str.split('.')[-1]
            manual_map = {'sin': 'f64.sin', 'cos': 'f64.cos', 'tan': 'f64.tan', 'log': 'f64.log', 'ln': 'f64.log', 'exp': 'f64.exp', 'sqrt': 'f64.sqrt', 'atan': 'f64.atan', 'asin': 'f64.asin', 'acos': 'f64.acos', 'arctan': 'f64.atan', 'arcsin': 'f64.asin', 'arccos': 'f64.acos', 'abs': 'f64.abs', 'sign': 'f64.sgn', 'ceiling': 'f64.ceil', 'floor': 'f64.floor', 'min': 'f64.min', 'max': 'f64.max'}
            func_name = manual_map.get(func_str, f'{func_str}')
        args = [self.sympy_to_futhark(a) for a in expr.args]
        func_str_lower = str(func_obj).lower()
        if (func_obj in (sympy.Min, sympy.Max) or 'min' in func_str_lower or 'max' in func_str_lower) and len(args) >= 2:
            if func_obj == sympy.Min or 'min' in func_str_lower:
                op = 'f64.min'
            else:
                op = 'f64.max'
            result = args[-1]
            for arg in reversed(args[:-1]):
                result = f'{op} {arg} ({result})'
            print(f'HANDLE_FUNC: returning (nested min/max): {result}')
            return result
        func_str = str(func_obj).lower()
        if 'log' in func_str and len(args) > 1:
            main_arg, base_arg = (args[0], args[1])
            if base_arg in ('f64.e', 'e', '2.718'):
                return f'f64.log {main_arg}'
            return f'(f64.log {main_arg} / f64.log {base_arg})'
        if len(args) == 1:
            result = f'{func_name} {args[0]}'
            print(f'HANDLE_FUNC: returning (single arg): {result}')
            return result
        result = f'{func_name}({', '.join(args)})'
        print(f'HANDLE_FUNC: returning (multi arg): {result}')
        return result

    def handle_sum(self, expr: Sum) -> str:
        try:
            summand = expr.args[0]
            if len(expr.args) >= 2:
                limits = expr.args[1:]
                indices = []
                for limit in limits:
                    if hasattr(limit, '__iter__') and len(limit) >= 3:
                        index, lower, upper = (limit[0], limit[1], limit[2])
                        index_str = str(index)
                        self.register_symbol(index, 'i64')
                        self.current_indices.add(index_str)
                        self.bound_variables.add(index_str)
                        indices.append((index_str, lower, upper))
                if indices:
                    summand_str = self.sympy_to_futhark(summand)
                    if hasattr(summand, 'as_coeff_Mul'):
                        coeff, rest = summand.as_coeff_Mul()
                        for idx, _, _ in indices:
                            if str(coeff).find(f'(-1)**{idx}') != -1:
                                summand_str = summand_str.replace(f'((-1.0) ** f64.i64 {idx})', f'(if {idx} % 2i64 == 0i64 then 1.0 else -1.0)')
                    futhark_code = summand_str
                    for index_str, lower, upper in reversed(indices):
                        lower_str = self.sympy_to_futhark(lower)
                        upper_str = self.sympy_to_futhark(upper)
                        futhark_code = f'(let lower_{index_str} = i64.f64 {lower_str} in let upper_{index_str} = i64.f64 {upper_str} in let range_size_{index_str} = upper_{index_str} - lower_{index_str} + 1i64 in let sum_indices_{index_str} = map (\\i -> i + lower_{index_str}) (iota range_size_{index_str}) in f64.sum (map (\\{index_str} -> {futhark_code}) sum_indices_{index_str}))'
                    for index_str, _, _ in indices:
                        if index_str in self.current_indices:
                            self.current_indices.remove(index_str)
                    return futhark_code
        except Exception as e:
            print(f'Exception in handle_sum: {e}')
            import traceback
            traceback.print_exc()
        return self._omit(expr)

    def handle_integral(self, expr: Integral) -> str:
        try:
            integrand = expr.args[0]
            var_limits = []
            for i in range(1, len(expr.args)):
                var_limit = expr.args[i]
                if isinstance(var_limit, tuple) or (hasattr(var_limit, "__iter__") and len(var_limit) >= 1):
                    if len(var_limit) == 1:
                        var = var_limit[0]
                        var_limits.append((var, None, None))
                    elif len(var_limit) >= 3:
                        var, lower, upper = var_limit[0], var_limit[1], var_limit[2]
                        var_limits.append((var, lower, upper))
            if len(var_limits) == 1 and var_limits[0][1] is not None:
                var, lower, upper = var_limits[0]
                var_str = str(var)
                self.register_symbol(var_str)
                self.current_indices.add(var_str)
                self.bound_variables.add(var_str)
                safe_var = self.get_safe_variable_name(var_str)
                safe_i = self.get_safe_variable_name("i", {safe_var})
                safe_weight = self.get_safe_variable_name("w", {safe_var, safe_i})
                lower_str = self.sympy_to_futhark(lower)
                upper_str = self.sympy_to_futhark(upper)
                safe_var_symbol = sympy.Symbol(safe_var)
                replacements = {}
                for sym in integrand.free_symbols:
                    if str(sym) == var_str:
                        replacements[sym] = safe_var_symbol
                
                integrand_substituted = integrand.xreplace(replacements) if replacements else integrand
                integrand_str = self.sympy_to_futhark(integrand_substituted)

                if var_str in self.current_indices:
                    self.current_indices.remove(var_str)

                return f"""
    -- Numerical integration using trapezoidal rule
    let steps = 1000  -- Number of integration steps
    let step_size = ({upper_str} - {lower_str}) / f64.i64 steps
    let points = map (\\{safe_i} -> {lower_str} + step_size * f64.i64 {safe_i}) (iota (steps + 1))
    let values = map (\\{safe_var} -> {integrand_str}) points
    let weights = map (\\{safe_i} -> if {safe_i} == 0 || {safe_i} == steps then 0.5 else 1.0) (iota (steps + 1))
    let terms = map2 (*) weights values
    in step_size * f64.sum terms
    """
            else:
                return self._omit(expr)
        
        except Exception as e:
            print(f"Exception in handle_integral: {e}")
            import traceback
            traceback.print_exc()
            return self._omit(expr)

    def handle_product(self, expr: sympy.Product) -> str:
        try:
            if len(expr.args) >= 2 and isinstance(expr.args[1], tuple) and len(expr.args[1]) >= 3:
                term = expr.args[0]
                index, lower, upper = expr.args[1][0], expr.args[1][1], expr.args[1][2]
                index_str = str(index)

                self.register_symbol(index, "i64")
                self.current_indices.add(index_str)
                self.bound_variables.add(index_str)

                safe_index = self.get_safe_variable_name(index_str)
                safe_iter = self.get_safe_variable_name("i", {safe_index})
                
                lower_str = self.sympy_to_futhark(lower)
                upper_str = self.sympy_to_futhark(upper)
                
                term_str = self.sympy_to_futhark(term)
                
                if index_str in self.current_indices:
                    self.current_indices.remove(index_str)
                
                return f"""(f64.reduce (*) 1.0 (map (\\{safe_index} -> 
    {term_str}
    ) (map (\\{safe_iter} -> {safe_iter} + i64.f64 {lower_str}) (iota (i64.f64 ({upper_str} - {lower_str} + 1.0))))))"""
        
        except Exception as e:
            print(f"Exception in handle_product: {e}")
            import traceback
            traceback.print_exc()
        
        return self._omit(expr)

    def preprocess_derivatives(self, s: str) -> str:
        s = s.replace('\\left', '').replace('\\right', '')
        s = re.sub('\\\\frac\\{d\\^(\\d+)\\s*([\\s\\S]+?)\\}\\{d([A-Za-z]+)\\^(\\d+)\\}', lambda m: '\\frac{{d^{}}}{{d{}^{}}} {}'.format(m.group(1), m.group(3), m.group(4), m.group(2)), s)
        s = re.sub('\\\\frac\\{\\\\partial\\^(\\d+)\\s*([\\s\\S]+?)\\}\\{\\\\partial\\s*([A-Za-z]+)\\^(\\d+)\\}', lambda m: '\\frac{{\\partial^{}}}{{\\partial {}^{}}} {}'.format(m.group(1), m.group(3), m.group(4), m.group(2)), s)
        return s

    def handle_derivative(self, expr: sympy.Derivative) -> str:
        try:
            analytic = expr.doit()
        except Exception:
            func_to_diff = expr.expr if hasattr(expr, 'expr') else expr.args[0]
            diff_vars_args = []
            if hasattr(expr, 'variables'):
                for v_spec in expr.variables:
                    if isinstance(v_spec, tuple):
                        diff_vars_args.extend(v_spec)
                    else:
                        diff_vars_args.append(v_spec)
            else:
                for arg_spec in expr.args[1:]:
                    if isinstance(arg_spec, tuple) and len(arg_spec) == 2:
                        diff_vars_args.extend(arg_spec)
                    else:
                        diff_vars_args.append(arg_spec)
            analytic = sympy.diff(func_to_diff, *diff_vars_args)
        return self.sympy_to_futhark(analytic)

    def handle_matrix(self, expr) -> str:
        try:
            print(f'Matrix type: {type(expr)}')
            print(f'Matrix repr: {repr(expr)}')
            if hasattr(expr, 'shape') and hasattr(expr, '__getitem__'):
                rows, cols = expr.shape
                if cols == 1:
                    vector_entries = []
                    for i in range(rows):
                        try:
                            entry = expr[i, 0]
                            entry_str = self.sympy_to_futhark(entry)
                            vector_entries.append(entry_str)
                        except Exception as e:
                            print(f'Error accessing vector element [{i}]: {e}')
                            vector_entries.append('0.0')
                    return f'[{', '.join(vector_entries)}]'
                matrix_entries = []
                for i in range(rows):
                    row_entries = []
                    for j in range(cols):
                        try:
                            try:
                                entry = expr[i, j]
                            except:
                                try:
                                    entry = expr[i][j]
                                except:
                                    entry = None
                            if entry is not None:
                                entry_str = self.sympy_to_futhark(entry)
                                row_entries.append(entry_str)
                            else:
                                row_entries.append('0.0')
                        except Exception as e:
                            print(f'Error accessing matrix element [{i},{j}]: {e}')
                            row_entries.append('0.0')
                    matrix_entries.append(f'[{', '.join(row_entries)}]')
                return f'[{', '.join(matrix_entries)}]'
            elif hasattr(expr, 'name'):
                return str(expr.name)
            elif hasattr(expr, '__len__') and (not hasattr(expr, 'shape')):
                vector_entries = []
                for i in range(len(expr)):
                    try:
                        entry = expr[i]
                        entry_str = self.sympy_to_futhark(entry)
                        vector_entries.append(entry_str)
                    except Exception as e:
                        print(f'Error accessing vector element [{i}]: {e}')
                        vector_entries.append('0.0')
                return f'[{', '.join(vector_entries)}]'
            else:
                return self._omit(expr)
        except Exception as e:
            print(f'Exception in handle_matrix: {e}')
            import traceback
            traceback.print_exc()
            return self._omit(expr)

    def handle_matrix_multiplication(self, expr) -> str:
        try:
            matrices = [self.sympy_to_futhark(arg) for arg in expr.args]
            shapes = []
            for arg in expr.args:
                if hasattr(arg, 'shape'):
                    shapes.append(arg.shape)
                else:
                    shapes.append(None)
            
            row_var = self.get_safe_variable_name("row")
            col_var = self.get_safe_variable_name("col")
            vec_elem_var = self.get_safe_variable_name("elem")
            
            if len(shapes) == 2 and shapes[0] is not None and shapes[1] is not None:
                if len(shapes[0]) == 2 and len(shapes[1]) == 2 and shapes[1][1] == 1:
                    return f"""
    -- Matrix-vector multiplication
    let mat = {matrices[0]}
    let vec = map (\\{vec_elem_var} -> {vec_elem_var}[0]) {matrices[1]}  -- Extract column vector elements to 1D array
    let result = map (\\{row_var} -> 
                    reduce (+) 0.0 (map2 (*) {row_var} vec)
                    ) mat
    in result
    """

            result = matrices[0]
            for i in range(1, len(matrices)):
                next_matrix = matrices[i]
                result = f"""
    -- Matrix multiplication
    let a = {result}
    let b = {next_matrix}
    let result = map (\\{row_var} ->
                    map (\\{col_var} ->
                        reduce (+) 0.0 (map2 (*) {row_var} {col_var})
                        ) (transpose b)
                    ) a
    in result
    """
            return result
        except Exception as e:
            print(f"Exception in handle_matrix_multiplication: {e}")
            return self._omit(expr)

    def handle_matrix_addition(self, expr: sympy.MatAdd) -> str:
        try:
            matrices = []
            operations = []

            for arg in expr.args:
                if isinstance(arg, sympy.Mul) and len(arg.args) >= 1 and arg.args[0] == -1:
                    matrices.append(self.sympy_to_futhark(arg.args[1] if len(arg.args) > 1 else sympy.S.One))
                    operations.append("-")
                else:
                    matrices.append(self.sympy_to_futhark(arg))
                    operations.append("+")
            
            operations[0] = "+"
            
            safe_row1 = self.get_safe_variable_name("row1")
            safe_row2 = self.get_safe_variable_name("row2", {safe_row1})
            
            if len(matrices) == 1:
                return matrices[0]
            
            result = matrices[0]
            for i in range(1, len(matrices)):
                op = operations[i]
                next_matrix = matrices[i]
                result = f"""
    -- Matrix addition/subtraction
    let result = 
    map2 (\\{safe_row1} {safe_row2} ->
        map2 {op} {safe_row1} {safe_row2}
    ) {result} {next_matrix}
    in result
    """
            return result
        except Exception as e:
            print(f"Exception in handle_matrix_addition: {e}")
            import traceback
            traceback.print_exc()
            return self._omit(expr)

    def handle_determinant(self, expr: sympy.Determinant) -> str:
        try:
            matrix = self.sympy_to_futhark(expr.args[0])
            return f'det({matrix})'
        except Exception as e:
            print(f'Exception in handle_determinant: {e}')
            import traceback
            traceback.print_exc()
            return self._omit(expr)

    def handle_transpose(self, expr: sympy.Transpose) -> str:
        try:
            matrix = self.sympy_to_futhark(expr.args[0])
            return f'transpose {matrix}'
        except Exception as e:
            print(f'Exception in handle_transpose: {e}')
            import traceback
            traceback.print_exc()
            return self._omit(expr)

    def handle_piecewise(self, expr: sympy.Piecewise) -> str:
        futhark_code = []
        for i, (e, c) in enumerate(expr.args):
            if i == 0:
                cond = self.sympy_to_futhark(c)
                if cond == 'True':
                    return self.sympy_to_futhark(e)
                value = self.sympy_to_futhark(e)
                futhark_code.append(f'if {cond} then {value}')
            elif c == True:
                value = self.sympy_to_futhark(e)
                futhark_code.append(f'else {value}')
            else:
                cond = self.sympy_to_futhark(c)
                value = self.sympy_to_futhark(e)
                futhark_code.append(f'else if {cond} then {value}')
        return ' '.join(futhark_code)

    def _collect_symbols_preorder(self, expr):
        try:
            for sym in expr.free_symbols:
                if hasattr(sym, 'name') and sym.name not in self.RESERVED:
                    self.register_symbol(sym)
        except Exception as e:
            print(f'Warning: symbol collection error: {e}')

    def _fix_derivative_multiplication(self, expr):
        if not isinstance(expr, sympy.Mul):
            return expr
        try:
            num, den = expr.as_numer_denom()
            num_factors = sympy.Mul.make_args(num)
            den_factors = sympy.Mul.make_args(den)
            d_power = None
            d_order = None
            func_expr = None
            var_name = None
            remaining_num_factors = []
            for factor in num_factors:
                if isinstance(factor, sympy.Pow) and isinstance(factor.base, sympy.Symbol) and (factor.base.name == 'd') and isinstance(factor.exp, (sympy.Integer, int)):
                    d_power = factor
                    d_order = int(factor.exp)
                else:
                    remaining_num_factors.append(factor)
            for factor in den_factors:
                if isinstance(factor, sympy.Pow) and isinstance(factor.base, sympy.Symbol) and factor.base.name.startswith('d') and (len(factor.base.name) > 1) and isinstance(factor.exp, (sympy.Integer, int)):
                    var_name = factor.base.name[1:]
                    den_order = int(factor.exp)
                    if d_order and d_order == den_order:
                        if len(remaining_num_factors) == 1:
                            func_expr = remaining_num_factors[0]
                        elif len(remaining_num_factors) > 1:
                            func_expr = sympy.Mul(*remaining_num_factors)
                        else:
                            func_expr = sympy.Integer(1)
                        var_sym = sympy.Symbol(var_name)
                        derivative = sympy.Derivative(func_expr, var_sym, d_order)
                        computed = derivative.doit()
                        print(f'Fixed derivative multiplication: {expr} -> {computed}')
                        return computed
        except Exception as e:
            print(f'Error fixing derivative multiplication: {e}')
        return expr

    def sympy_to_futhark(self, expr: Any) -> str:
        print('SYM2F: input expr:', repr(expr))
        print(f'SYM2F: expr type: {type(expr)}')
        if hasattr(expr, 'func'):
            print(f'SYM2F: expr.func: {expr.func}')
        if hasattr(expr, 'args'):
            print(f'SYM2F: expr.args: {expr.args}')
        'Convert SymPy expression to Futhark code using the handler registry'
        if expr is None:
            return '-- None expression'
        try:
            if isinstance(expr, sympy.Mul):
                expr = self._fix_derivative_multiplication(expr)
            elif hasattr(expr, 'args'):

                def fix_derivs_recursive(e):
                    if isinstance(e, sympy.Mul):
                        return self._fix_derivative_multiplication(e)
                    elif hasattr(e, 'args') and e.args:
                        new_args = [fix_derivs_recursive(arg) for arg in e.args]
                        if hasattr(e, 'func'):
                            try:
                                return e.func(*new_args)
                            except:
                                return e
                    return e
                expr = fix_derivs_recursive(expr)
        except Exception as e:
            print(f'Error fixing derivative multiplications: {e}')

        def doit_derivs_only(e):
            if isinstance(e, sympy.Derivative):
                try:
                    return e.doit()
                except Exception:
                    return e
            if not hasattr(e, 'args') or not e.args:
                return e
            return e.func(*[doit_derivs_only(arg) for arg in e.args])
        expr = doit_derivs_only(expr)
        try:
            expr = self._fixup_derivative_literals(expr)
        except Exception as e:
            print(f'Warning: could not fix derivative literals: {e}')
        handler = None
        expr_type = type(expr)
        print(f'SYM2F: looking for handler for type {expr_type}')
        if isinstance(expr, sympy.Symbol):
            result = self.handle_symbol(expr)
            print(f'SYM2F: symbol handler returned: {result}')
            return result
        if expr_type in self.handlers:
            handler = self.handlers[expr_type]
            print(f'SYM2F: found exact handler: {handler}')
        else:
            for base_type, base_handler in self.handlers.items():
                if isinstance(expr, base_type):
                    handler = base_handler
                    print(f'SYM2F: found parent class handler: {handler} for type {base_type}')
                    break
        if handler:
            try:
                result = handler(expr)
                print(f'SYM2F: handler returned: {result}')
                return result
            except Exception as e:
                print(f'SYM2F: Handler error for {expr_type}: {e}')
                import traceback
                traceback.print_exc()
                fallback = str(expr)
                print(f'SYM2F: using fallback: {fallback}')
                return fallback
        else:
            print(f'SYM2F: no handler found, using _omit')
            return self._omit(expr)

    def transpile(self, latex_expr: str) -> str:
        L = latex_expr.strip()
        if L.startswith('\\det') and '\\begin{pmatrix}' in L:
            try:
                matrix_count = len(re.findall('\\\\begin\\{pmatrix\\}', L))
                if matrix_count > 1:
                    expr = parse_any_latex(L)
                    self._collect_symbols_preorder(expr)
                    return self.sympy_to_futhark(expr)
                else:
                    result = self.transpile_determinant(L)
                    if result and (not result.startswith('--')):
                        return result
            except Exception as e:
                print(f'Determinant transpilation failed: {e}')
                return f'-- det error: {str(e)[:50]}...'
        if not any((op in L for op in ['+', '-'])) or len(L) < 50:
            try:
                expr = parse_any_latex(L)
                self._collect_symbols_preorder(expr)
                return self.sympy_to_futhark(expr)
            except Exception as e:
                print(f'Direct parsing failed: {e}')
        try:
            blocks = split_latex_robust(L)
            pieces = []
            for kind, payload in blocks:
                if kind == 'op':
                    pieces.append(payload)
                elif kind == 'sympy':
                    try:
                        expr = process_split_part(payload)
                        self._collect_symbols_preorder(expr)
                        result = self.sympy_to_futhark(expr)
                        pieces.append(result)
                    except Exception as e:
                        print(f"[parse error for part '{payload[:50]}...']: {e}")
                        pieces.append(f'-- unparsed: {payload[:50]}...')
            return ' '.join(pieces)
        except Exception as e:
            print(f'Transpilation failed: {e}')
            return f'-- transpilation error: {str(e)[:50]}...'

    def transpile_determinant(self, latex_expr):
        print('Trying determinant special handler')
        if not ('\\det' in latex_expr and '\\begin{pmatrix}' in latex_expr):
            return None
        try:
            start_idx = latex_expr.find('\\begin{pmatrix}')
            end_idx = latex_expr.find('\\end{pmatrix}') + len('\\end{pmatrix}')
            if start_idx == -1 or end_idx == -1:
                return None
            matrix_latex = latex_expr[start_idx:end_idx]
            print(f'Extracted matrix: {matrix_latex}')
            mathml = latex_to_mathml(matrix_latex)
            matrix = parse_mathml_matrix(mathml)
            matrix_code = self.sympy_matrix_to_futhark(matrix)
            result = f'det({matrix_code})'
            print(f'Determinant result: {result}')
            return result
        except Exception as e:
            print(f'Direct determinant handler error: {e}')
            import traceback
            traceback.print_exc()
            return None

    def infer_type(self, expr):
        if expr is None:
            return 'f64'
        if isinstance(expr, sympy.Symbol):
            return self.symbols.get(str(expr), 'f64')
        if isinstance(expr, (sympy.Integer, sympy.Float, sympy.Rational)):
            return 'f64'
        if hasattr(expr, 'shape'):
            shape = expr.shape
            if len(shape) == 0:
                return 'f64'
            elif len(shape) == 1 or (len(shape) == 2 and shape[1] == 1):
                return '[]f64'
            elif len(shape) == 2:
                return '[][]f64'
        if isinstance(expr, (sympy.Add, sympy.Mul, sympy.Pow)):
            arg_types = [self.infer_type(arg) for arg in expr.args]
            type_rank = {'f64': 0, '[]f64': 1, '[][]f64': 2}
            ranks = [type_rank.get(t, 0) for t in arg_types]
            max_rank = max(ranks)
            rank_type = {0: 'f64', 1: '[]f64', 2: '[][]f64'}
            return rank_type[max_rank]
        if isinstance(expr, sympy.Function):
            arg_types = [self.infer_type(arg) for arg in expr.args]
            if str(expr.func).lower() in ['sum', 'product', 'det', 'determinant']:
                return 'f64'
            type_rank = {'f64': 0, '[]f64': 1, '[][]f64': 2}
            ranks = [type_rank.get(t, 0) for t in arg_types]
            max_rank = max(ranks)
            rank_type = {0: 'f64', 1: '[]f64', 2: '[][]f64'}
            return rank_type[max_rank]
        return 'f64'

def generate_futhark_program(expr: str, function_name: str='compute') -> str:
    tp = LaTeX2FutharkTranspiler()
    print(f'Program generation: transpiling {expr[:100]}…')
    futhark_expr = tp.transpile(expr)
    print(f'Program generation result: {futhark_expr[:200]}…')
    symbols, params, param_names = (tp.symbol_list, [], [])
    for name in symbols:
        if name not in tp.bound_variables:
            typ = tp.symbols.get(name, 'f64')
            params.append(f'({name}: {typ})')
            param_names.append(name)
    unpack = '\n'.join((f'  let {n} = params[{i}] in' for i, n in enumerate(param_names)))
    det_code = """
-- Determinant of a matrix using Gaussian elimination
let det [n] (mat: [n][n]f64): f64 =
  let mat_copy = copy mat
  let (m, sign) =
    loop (m, sign) = (mat_copy, 1.0) for i < n do
      let (pivot_idx, max_val) =
        loop (pivot_idx, max_val) = (i, f64.abs m[i,i]) for j in i+1..<n do
          let abs_val = f64.abs m[j,i]
          in if abs_val > max_val then (j, abs_val) else (pivot_idx, max_val)
      in if max_val == 0.0 then
           (m, 0.0)
         else
           let (m, sign) =
             if pivot_idx == i then
               (m, sign)
             else
               let tmp       = copy m[i]
               let pivot_row = copy m[pivot_idx]
               let m[i]      = pivot_row
               let m[pivot_idx] = tmp
               in (m, -sign)
           let m =
             loop m = m for j in i+1..<n do
               let factor = m[j,i] / m[i,i]
               let m[j]   = map2 (\mji mii -> mji - factor * mii) m[j] m[i]
               in m
           in (m, sign)
  in sign * reduce (*) 1.0 (map (\k -> m[k,k]) (iota n))
    """
        
    svd_code = r"""
-- SVD Jacobi with Bubble Sort (Slow)
def dot [n_dim] (x: [n_dim]f64) (y: [n_dim]f64) : f64 =
  reduce (+) 0.0 (map2 (*) x y)

def vecnorm [n_dim] (x: [n_dim]f64) : f64 =
  f64.sqrt (dot x x)

def tabulate_2d [r][c] 't (f: i64 -> i64 -> t) : [r][c]t =
  map (\i -> map (\j -> f i j) (iota c)) (iota r)

def perform_jacobi_sweeps [m_dim][n_dim]
    (a_initial: [m_dim][n_dim]f64) (v_initial: [n_dim][n_dim]f64)
    (max_sweeps_local: i64) (tol_local: f64)
  : ([m_dim][n_dim]f64, [n_dim][n_dim]f64) =
  loop (a_cur_sweep: [m_dim][n_dim]f64, v_cur_sweep: [n_dim][n_dim]f64) = (a_initial, v_initial)
       for _sweep < max_sweeps_local do
    let (a_after_p_loops, v_after_p_loops, _rotated_flag_unused) =
      loop (a_p_iter: [m_dim][n_dim]f64, v_p_iter: [n_dim][n_dim]f64, rotated_in_p_iter: bool) =
           (a_cur_sweep, v_cur_sweep, false)
           for p < n_dim - 1 do
        loop (a_q_iter: [m_dim][n_dim]f64, v_q_iter: [n_dim][n_dim]f64, rotated_in_q_iter: bool) =
             (a_p_iter, v_p_iter, rotated_in_p_iter)
             for q < n_dim do
          if q <= p then (a_q_iter, v_q_iter, rotated_in_q_iter)
          else
            let col_p = a_q_iter[:, p]
            let col_q = a_q_iter[:, q]
            let app   = dot col_p col_p
            let aqq   = dot col_q col_q
            let apq   = dot col_p col_q
            in
            if apq == 0.0 || f64.abs apq < tol_local * f64.sqrt (app * aqq) then
                 (a_q_iter, v_q_iter, rotated_in_q_iter)
            else
                 let tau = (aqq - app) / (2.0 * apq)
                 let t   = if tau >= 0.0
                           then 1.0 / (tau + f64.sqrt (1.0 + tau * tau))
                           else -1.0 / (f64.abs tau + f64.sqrt (1.0 + tau * tau))
                 let c   = 1.0 / f64.sqrt (1.0 + t * t)
                 let s   = t * c
                 let a_rot: [m_dim][n_dim]f64 =
                   tabulate_2d
                     (\(i: i64) (j: i64) : f64 ->
                       if j == p then c * a_q_iter[i,p] - s * a_q_iter[i,q]
                       else if j == q then s * a_q_iter[i,p] + c * a_q_iter[i,q]
                       else a_q_iter[i,j])
                 let v_rot: [n_dim][n_dim]f64 =
                   tabulate_2d
                     (\(i: i64) (j: i64) : f64 ->
                       if j == p then c * v_q_iter[i,p] - s * v_q_iter[i,q]
                       else if j == q then s * v_q_iter[i,p] + c * v_q_iter[i,q]
                       else v_q_iter[i,j])
                 in (a_rot, v_rot, true)
    in (a_after_p_loops, v_after_p_loops)

def svd
    [m][n] (a_input: [m][n]f64)
    (max_iter: i64) (eps: f64)
  :              ([i64.min m n]f64,
                  [m][i64.min m n]f64,
                  [n][i64.min m n]f64) =
  let k    = i64.min m n
  let a0   = copy a_input
  let v0_identity: [n][n]f64 = tabulate_2d (\(r:i64) (c_idx:i64):f64 -> if r == c_idx then 1.0 else 0.0)
  let (a_final_cols, v_final_full) = perform_jacobi_sweeps a0 v0_identity max_iter eps
  let pairs : [n](f64, i64) =
       map (\(j: i64) -> (vecnorm a_final_cols[:, j], j)) (iota n)

  -- Bubble Sort (descending)
  let sorted_pairs : [n](f64, i64) =
       loop (arr_outer : [n](f64,i64)) = pairs for i < n - 1 do
         loop (arr_inner_param : [n](f64,i64)) = arr_outer for j < n - 1 - i do
           let arr_to_modify = copy arr_inner_param
           let (s1_val, _idx1) = arr_to_modify[j]
           let (s2_val, _idx2) = arr_to_modify[j+1]
           in if s1_val < s2_val
              then arr_to_modify with [j]   = arr_to_modify[j+1]
                                 with [j+1] = arr_to_modify[j]
              else arr_to_modify
  
  let sigmas_vec  : [k]f64 = map (.0) (take k sorted_pairs)
  let original_indices : [k]i64 = map (.1) (take k sorted_pairs)
  let u_econ : [m][k]f64 =
    tabulate_2d
      (\(row_idx:i64) (col_k_idx:i64):f64 ->
         let original_col_idx = original_indices[col_k_idx]
         let s_val = sigmas_vec[col_k_idx]
         in if s_val > eps then a_final_cols[row_idx, original_col_idx] / s_val else 0.0)
  let v_econ : [n][k]f64 =
    tabulate_2d
      (\(row_idx:i64) (col_k_idx:i64):f64 ->
         v_final_full[row_idx, original_indices[col_k_idx]])
  in (sigmas_vec, u_econ, v_econ)

def svd_default [m][n] (a:[m][n]f64) =
  svd a 1000 1e-12
"""
    extra_helpers = ""
    if "det(" in futhark_expr:
        extra_helpers += det_code
    if re.search(r"\bsvd_default\b", futhark_expr):
        extra_helpers += svd_code

    try:
        parsed_expr = tp.latex_to_sympy(expr)
        return_type = tp.infer_type(parsed_expr)
    except:
        if futhark_expr.strip().startswith("[["):
            return_type = "[][]f64"
        elif futhark_expr.strip().startswith("["):
            return_type = "[]f64"
        else:
            return_type = "f64"
    
    if futhark_expr.strip().startswith("svd_default"):
        return_type = "auto"
    if return_type == "auto":
        program = f"""-- Auto-generated from LaTeX: {expr}
{extra_helpers}
let {function_name} {' '.join(params)} =
  {futhark_expr}

entry main(params: []f64) =
{unpack}
  {function_name} {' '.join(param_names)}
"""
    else:
        program = f"""-- Auto-generated from LaTeX: {expr}
{extra_helpers}
let {function_name} {' '.join(params)}: {return_type} =
  {futhark_expr}

entry main(params: []f64): {return_type} =
{unpack}
  {function_name} {' '.join(param_names)}
"""
    
    return program
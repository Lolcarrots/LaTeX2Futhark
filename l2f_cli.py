import sys
import argparse
from engine import LaTeX2FutharkTranspiler, generate_futhark_program

def transpile_expression(latex_expr, show_debug=False, generate_program=False):
    print('INPUT LaTeX:')
    print('=' * 60)
    print(latex_expr)
    print('\n' + '=' * 60)
    tp = LaTeX2FutharkTranspiler()
    try:
        result = tp.transpile(latex_expr)
        if result.startswith('--'):
            print('TRANSPILATION FAILED:')
            print(result)
            return False
        else:
            print('FUTHARK OUTPUT:')
            print('=' * 60)
            if len(result) > 200:
                print('Long result - showing formatted version:')
                print(format_long_result(result))
            else:
                print(result)
            print(f'\nLength: {len(result)} characters')
            if tp.symbols:
                print(f'\nVariables found: {list(tp.symbols.keys())}')
                print('Variable types:')
                for var, var_type in tp.symbols.items():
                    print(f'  {var}: {var_type}')
            if generate_program:
                print('\n' + '=' * 60)
                print('COMPLETE FUTHARK PROGRAM:')
                print('=' * 60)
                try:
                    program = generate_futhark_program(latex_expr)
                    print(program)
                except Exception as e:
                    print(f'Program generation failed: {e}')
            return True
    except Exception as e:
        print('ERROR:')
        print(f'  {e}')
        if show_debug:
            import traceback
            traceback.print_exc()
        return False

def format_long_result(result):
    formatted = result
    formatted = formatted.replace(' + (let ', ' +\n    (let ')
    formatted = formatted.replace(')) + (((', '))\n + (((')
    formatted = formatted.replace(' + det(', '\n + det(')
    formatted = formatted.replace(' + ((', '\n + ((')
    lines = []
    for line in formatted.split('\n'):
        if 'let ' in line and len(line) > 80:
            parts = line.split(' let ')
            if len(parts) > 1:
                lines.append(parts[0])
                for part in parts[1:]:
                    lines.append('    let ' + part)
        else:
            lines.append(line)
    return '\n'.join(lines)

def get_example_expressions():
    """Return a dictionary of example expressions"""
    return {
        "simple": r"\frac{x^2 + 1}{y - 2}",
        "derivative": r"\frac{d}{dx} x^3 + \sin(x)",
        "sum": r"\sum_{n=1}^{10} \frac{1}{n^2}",
        "matrix": r"\det\begin{pmatrix} x & y \\ 1 & z \end{pmatrix}",
        "complex": r"\frac{e^{-x}}{1 + \ln(y^2)} + \sqrt{x^2 + y^2}",
        "integral": r"\int_0^1 x^2 dx",
        "giant": r"""\frac{x^4 + 2 x^3 y - 3 x^2 y^2 + 4 x y^3 - y^4}{1 + x^2 + y^2 + z^2}^2 + \sum_{n=1}^{M} \frac{(-1)^n}{n^2} \cdot \frac{\sin(n x)\cos(n y)}{1 + \ln(n + z^2)} + \sqrt{\frac{d^2}{dx^2} \arctan(x y)^2 + \frac{d}{dy}\frac{x^2}{1 + y^2}^2} + \frac{1}{1 + e^{-x y z}}^3 + \det\left(\begin{pmatrix}x & y\\1 & z\end{pmatrix}\begin{pmatrix}2 & x\\y & 3\end{pmatrix}^T - \begin{pmatrix}1 & 0\\0 & x\end{pmatrix}\right)""",
        "bigger": r"""\frac{x^4 + 2 x^3 y - 3 x^2 y^2 + 4 x y^3 - y^4}{1 + x^2 + y^2 + z^2}^2 + \sum_{n=1}^{M} \frac{(-1)^n}{n^2} \cdot \frac{\sin(n x)\cos(n y)}{1 + \ln(n + z^2)} + \sqrt{\frac{d^2}{dx^2} \arctan(x y)^2 + \frac{d}{dy}\frac{x^2}{1 + y^2}^2} + \frac{1}{1 + e^{-x y z}}^3 + \det\left(\begin{pmatrix}x & y\\1 & z\end{pmatrix}\begin{pmatrix}2 & x\\y & 3\end{pmatrix}^T - \begin{pmatrix}1 & 0\\0 & x\end{pmatrix}\right) + \left(\int_0^1 t^2 dt\right) + \left(\sinh(x) + \cosh(y) + \tanh(z)\right) + \left(\arcsin(x) + \arccos(y)\right) + \left(\min(x,y,z) + \max(a,b,c)\right) + \left(|x-y|\right) + \det\left(\begin{pmatrix}a & b\\c & w\end{pmatrix} + \begin{pmatrix}1 & 2\\3 & 4\end{pmatrix}\right)"""
    }

def interactive_mode():
    print('LaTeX to Futhark Interactive Mode')
    print('=' * 40)
    print("Enter LaTeX expressions (or 'quit' to exit)")
    print('Special commands:')
    print("  'examples' - show example expressions")
    print("  'giant' - test the giant expression")
    print("  'program <expr>' - generate full Futhark program")
    print("  'quit' - exit")
    print()
    while True:
        try:
            user_input = input('LaTeX> ').strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'examples':
                show_examples()
            elif user_input.lower() == 'giant':
                transpile_expression(get_example_expressions()['giant'])
            elif user_input.startswith('program '):
                latex_expr = user_input[8:].strip()
                if latex_expr:
                    transpile_expression(latex_expr, generate_program=True)
                else:
                    print("Please provide a LaTeX expression after 'program'")
            elif user_input:
                transpile_expression(user_input)
            else:
                print('Please enter a LaTeX expression')
        except KeyboardInterrupt:
            print('\nGoodbye!')
            break
        except EOFError:
            print('\nGoodbye!')
            break
        print()

def show_examples():
    examples = get_example_expressions()
    print('Available example expressions:')
    print('-' * 40)
    for name, expr in examples.items():
        if name == 'giant':
            print(f'{name:10}: {expr[:60]}...')
        else:
            print(f'{name:10}: {expr}')
    print()
    print('Use: LaTeX> <expression>  or  LaTeX> examples')

def main():
    parser = argparse.ArgumentParser(description='LaTeX to Futhark transpiler test tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 l2f_cli.py '\frac{x^2}{y+1}'
  python3 l2f_cli.py --file input.txt
  python3 l2f_cli.py --interactive
  python3 l2f_cli.py --giant
  python3 l2f_cli.py --examples
  python3 l2f_cli.py --program 'x^2 + y^2'
        """
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('expression', nargs='?', help='LaTeX expression to transpile')
    input_group.add_argument('--file', '-f', help='Read LaTeX expression from file')
    input_group.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    input_group.add_argument('--giant', '-g', action='store_true', help='Test the giant expression')
    input_group.add_argument('--examples', '-e', action='store_true', help='Show example expressions')
    parser.add_argument('--debug', '-d', action='store_true', help='Show debug information')
    parser.add_argument('--program', '-p', action='store_true', help='Generate complete Futhark program')
    args = parser.parse_args()
    if args.examples:
        show_examples()
        return
    if args.interactive:
        interactive_mode()
        return
    latex_expr = None
    if args.expression:
        latex_expr = args.expression
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                latex_expr = f.read().strip()
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found")
            return 1
        except Exception as e:
            print(f'Error reading file: {e}')
            return 1
    elif args.giant:
        latex_expr = get_example_expressions()['giant']
    if latex_expr:
        success = transpile_expression(latex_expr, show_debug=args.debug, generate_program=args.program)
        return 0 if success else 1
    else:
        print('No LaTeX expression provided')
        return 1
if __name__ == '__main__':
    sys.exit(main())
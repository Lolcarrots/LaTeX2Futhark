# LaTeX2Futhark
This is a very early-stage transpiler for generating Futhark code based on LaTeX math inputs. There are currently some important limitations in the existing implementation (i.e., products and complex expressions with nesting), and the parsing currently requires the use of specific LaTeX syntax. The generated code often has suboptimal performance.

# Dependencies
sympy

antlr4-python3-runtime

LaTeXML + latexmlmath

# Dependency Installation (Ubuntu/Debian)
```bash
pip3 install sympy antlr4-python3-runtime
```
```bash
sudo apt install latexml
```

# Usage
Place one-liner inputs in quotation marks:
```bash
python3 l2f_cli.py "\frac{x}{2}"
```
Read a LaTeX math input from a file:
```bash
python3 l2f_cli.py --file expr.tex
```
Generate a full Futhark program stub:
```bash
python3 l2f_cli.py -p "\frac{x}{2}"
```

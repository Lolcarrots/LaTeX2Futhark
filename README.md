# LaTeX2Futhark
This is a transpiler for generating Futhark code based on LaTeX math inputs. There are currently some important limitations in the existing implementation (missing operators like argmin, argmax and others), and the implicit multiplication parsing (while often good) is imperfect (using \cdot gets around this). The generated code often has suboptimal performance (no transformations). Both integrals and multi-dimensionsal integrals use Simpson's rule, so there might be some tiny discrepancies between the current version of the transpiler and other tools/calculators when integrals are involved. Sums and products greedily absorb following terms when they are not enclosed in parentheses.

Lark is now the only dependency!

The previous version had trouble with complex/nested expressions-- the parser is now *much* better!

```text
\Bigg ( \det{\begin{pmatrix} \frac{\partial^{2}}{\partial j^{2}} e^{j u} & \frac{\partial^{2}}{\partial j \partial u} e^{j q} \\ \frac{\partial^{2}}{\partial u \partial j} e^{u j} & \frac{\partial^{2}}{\partial u^{2}} e^{u q} \end{pmatrix}} \Bigg ) \cdot \Bigg ( \frac{|b|^{\sum_{m=1}^{M} \bigg (\frac{\bigg (\prod_{r=1}^{x^{2}} \sum_{i=1}^{x^3} ( i + r ) \Bigg) \cdot t}{\bigg (\sum_{i=1}^{|k|} (y^{5})\bigg)^{\frac{\pi}{\ln(k)}} \cdot (3q)} \bigg)} \cdot \bigg ( \iiint_{\epsilon}^{L} \ln(x \cdot y \cdot z) \ln(1+(x \cdot y)) \, dx \, dy \, dz \bigg)}{\pi \cdot z \cdot \bigg (\prod_{r=1}^{p} \frac{1+\cos(r x)}{1+r} \bigg)^{\frac{d^{2}}{d x^{2}} \sum_{m=1}^{M}(\arcsin(\frac{x}{q}))}} \Bigg) \cdot \Bigg (\nabla_{q,y} \sin(q y) \Bigg) \cdot \| \begin{pmatrix} x & q \\ 3 & y^{2} \end{pmatrix} \| \cdot |\min \Bigg( \| \begin{bmatrix} \max \bigg( \| \begin{bmatrix} x & q \\ u^2 & y^{2} \\ y & x \end{bmatrix} \|, \| \begin{bmatrix} x & q \\ 3 & y^{2} \\ y & |x^u| \end{bmatrix} \|, \| \begin{bmatrix} u & 4 \\ 3 & y^{2} \\ |y| & |x| \end{bmatrix} \| \bigg) & q \\ \max \bigg( \| \begin{bmatrix} x & q \\ 3 & y^{2} \\ y & x \end{bmatrix} \|, \| \begin{bmatrix} x & q \\ 3 & y^{2} \\ y & |q^{q}| \end{bmatrix} \|, \| \begin{bmatrix} x & q \\ 4 & y^{2} \\ y & |x^{y}| \end{bmatrix} \| \bigg) & y^{2} \end{bmatrix} \|, \| \begin{bmatrix} x & q & \bigg( \sum_{i=1}^{|k|} ( 22^{x} - \sqrt{|b|} ) \bigg) \\ 3 & y^{2} & \bigg( \sum_{i=1}^{|k|} ( 22^{x} - b ) \bigg) \end{bmatrix} \|\Bigg)|
```
(GitHub's rendering handles line wrapping)
```math
\Bigg ( \det{\begin{pmatrix} \frac{\partial^{2}}{\partial j^{2}} e^{j u} & \frac{\partial^{2}}{\partial j \partial u} e^{j q} \\ \frac{\partial^{2}}{\partial u \partial j} e^{u j} & \frac{\partial^{2}}{\partial u^{2}} e^{u q} \end{pmatrix}} \Bigg ) \cdot \Bigg ( \frac{|b|^{\sum_{m=1}^{M} \bigg (\frac{\bigg (\prod_{r=1}^{x^{2}} \sum_{i=1}^{x^3} ( i + r ) \Bigg) \cdot t}{\bigg (\sum_{i=1}^{|k|} (y^{5})\bigg)^{\frac{\pi}{\ln(k)}} \cdot (3q)} \bigg)} \cdot \bigg ( \iiint_{\epsilon}^{L} \ln(x \cdot y \cdot z) \ln(1+(x \cdot y)) \, dx \, dy \, dz \bigg)}{\pi \cdot z \cdot \bigg (\prod_{r=1}^{p} \frac{1+\cos(r x)}{1+r} \bigg)^{\frac{d^{2}}{d x^{2}} \sum_{m=1}^{M}(\arcsin(\frac{x}{q}))}} \Bigg) \cdot \Bigg (\nabla_{q,y} \sin(q y) \Bigg) \cdot \| \begin{pmatrix} x & q \\ 3 & y^{2} \end{pmatrix} \| \cdot |\min \Bigg( \| \begin{bmatrix} \max \bigg( \| \begin{bmatrix} x & q \\ u^2 & y^{2} \\ y & x \end{bmatrix} \|, \| \begin{bmatrix} x & q \\ 3 & y^{2} \\ y & |x^u| \end{bmatrix} \|, \| \begin{bmatrix} u & 4 \\ 3 & y^{2} \\ |y| & |x| \end{bmatrix} \| \bigg) & q \\ \max \bigg( \| \begin{bmatrix} x & q \\ 3 & y^{2} \\ y & x \end{bmatrix} \|, \| \begin{bmatrix} x & q \\ 3 & y^{2} \\ y & |q^{q}| \end{bmatrix} \|, \| \begin{bmatrix} x & q \\ 4 & y^{2} \\ y & |x^{y}| \end{bmatrix} \| \bigg) & y^{2} \end{bmatrix} \|, \| \begin{bmatrix} x & q & \bigg( \sum_{i=1}^{|k|} ( 22^{x} - \sqrt{|b|} ) \bigg) \\ 3 & y^{2} & \bigg( \sum_{i=1}^{|k|} ( 22^{x} - b ) \bigg) \end{bmatrix} \|\Bigg)|
```

Using [futhark-pycffi](https://github.com/pepijndevos/futhark-pycffi/):
```python
# L M b epsilon j k p q t u x y z (in that order)
params = np.array([3.0, 2.0, -2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 1.3, 1.0, 1.0, 1.2, 0.9])

result = hugefutharktest.main(params)
result_array = hugefutharktest.from_futhark(result)

print(result_array)
```
```text
[-25775.35925206 -42958.93208083]
```


# Dependencies
lark

# Dependency Installation (Ubuntu/Debian)
```bash
pip3 install lark
```

# Usage
Place one-liner inputs in quotation marks:
```bash
python3 l2f_cli.py "\frac{x}{2}"
```
See only the AST (--raw for no preprocessing):
```bash
python3 ast_builder.py --pre "\frac{x}{2}"
```
See the preprocessed LaTeX, the AST, and the Futhark code:
```bash
python3 l2f_cli.py "\frac{x}{2}" --ast
```

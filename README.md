# luna-grad
A scalar-valued autograd engine and a neural net library on top of it with PyTorch-like API (Cpu computations only)

# Examples
```python
from lunagrad.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
```

-----

```python
from lunagrad import nn
n = nn.Neuron(2)
x = [Value(1.0), Value(-2.0)]
y = n(x)
```
![image](https://github.com/user-attachments/assets/6a8c1659-f3e6-4fa9-81cf-5d1e638e440b)

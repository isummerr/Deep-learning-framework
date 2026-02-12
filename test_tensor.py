from tensor import Tensor
a = Tensor(2.0, requires_grad=True)
b = Tensor(3.0, requires_grad=True)
c = a + b
c.backward()
print(f"Addition test: {c.data == 5.0}")
print(f"Gradient test: {a.grad == 1.0}")
import math
from tensor import Tensor

class CrossEntropyLoss:
    def forward(self, logits_tensor, target_indices):
        logits = logits_tensor.data
        batch_size = len(logits)
        probs = []
        total_loss = 0.0

        for i in range(batch_size):
            # Softmax with numerical stability
            max_l = max(logits[i])
            exps = [math.exp(l - max_l) for l in logits[i]]
            sum_exps = sum(exps)
            p = [e / sum_exps for e in exps]
            probs.append(p)

            target_idx = target_indices[i]
            total_loss += -math.log(max(p[target_idx], 1e-15))

        avg_loss = total_loss / batch_size
        out = Tensor(avg_loss, _children=(logits_tensor,), _op='CrossEntropy')

        def _backward():
            grad_logits = []
            for i in range(batch_size):
                row_grad = [p_val for p_val in probs[i]]
                row_grad[target_indices[i]] -= 1.0
                row_grad = [rg / batch_size for rg in row_grad]
                grad_logits.append(row_grad)
            
            if logits_tensor.grad is None:
                logits_tensor.grad = grad_logits
            else:
                for r in range(len(grad_logits)):
                    for c in range(len(grad_logits[0])):
                        logits_tensor.grad[r][c] += grad_logits[r][c]

        out._backward = _backward
        return out, probs

def get_accuracy(probs, target_indices):
    """Helper to calculate accuracy percentage."""
    correct = 0
    for i, p in enumerate(probs):
        # Find index of max probability
        pred = p.index(max(p))
        if pred == target_indices[i]:
            correct += 1
    return (correct / len(target_indices)) * 100

class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

    def step(self):
        for p in self.parameters:
            if p.grad is not None:
                self._update_recursive(p.data, p.grad)

    def _update_recursive(self, data, grad):
        for i in range(len(data)):
            if isinstance(data[i], list):
                self._update_recursive(data[i], grad[i])
            else:
                data[i] -= self.lr * grad[i]
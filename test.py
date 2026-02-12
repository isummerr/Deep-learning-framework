import pickle
import math
from data_loader import load_dataset, get_batches
from layers import Conv2d, ReLU, MaxPool2d, Flatten, Linear
from functional import get_accuracy

# SETTINGS
DATASET_NAME = "data_1" 
WEIGHTS_PATH = "model_weights.pkl" # Check your train.py to see which name you used

# 1. Load Weights
with open(WEIGHTS_PATH, "rb") as f:
    saved = pickle.load(f)
class_map = saved['class_map']

# 2. Load Test Data
images, labels, _, _ = load_dataset(DATASET_NAME, mode='test')

# 3. Setup Model
conv = Conv2d(3, 8, 3)
relu = ReLU(); pool = MaxPool2d(kernel_size=2); flat = Flatten()
fc = Linear(8 * 15 * 15, len(class_map))

# 4. Inject Weights
conv.weight.data = saved['conv_w']; conv.bias.data = saved['conv_b']
fc.weight.data = saved['fc_w']; fc.bias.data = saved['fc_b']

# 5. Evaluate
print("\nEvaluating on Test Set...")
total_acc, batches = 0, 0
for b_imgs, b_lbls in get_batches(images, labels, batch_size=4):
    x = conv.forward(b_imgs)
    logits = fc.forward(flat.forward(pool.forward(relu.forward(x.data))))
    
    probs = []
    for row in logits.data:
        m = max(row)
        exps = [math.exp(l - m) for l in row]
        s = sum(exps)
        probs.append([e/s for e in exps])
        
    total_acc += get_accuracy(probs, b_lbls)
    batches += 1

print(f"Final Test Accuracy: {total_acc/batches:.2f}%")
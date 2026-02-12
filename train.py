import time
import pickle
from data_loader import load_dataset, get_batches
from layers import Conv2d, ReLU, MaxPool2d, Flatten, Linear
from functional import CrossEntropyLoss, SGD, get_accuracy

# ==========================================
# 1. SETTINGS & HYPERPARAMETERS
# ==========================================
DATASET_NAME = "data_1"  # Your folder name in 'Assignment 1 Datasets'
SAMPLE_LIMIT = 2000      # Keeps training within 3 hours on MacBook Air
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 0.01

# ==========================================
# 2. LOAD DATA (Mandatory Metric: Loading Time)
# ==========================================
print(f"Initializing data loading for {DATASET_NAME}...")
images, labels, loading_time, class_map = load_dataset(DATASET_NAME, mode='train', sample_limit=SAMPLE_LIMIT)

print("-" * 30)
print(f"METRIC - Dataset Loading Time: {loading_time:.2f}s")
print(f"Total Samples Loaded: {len(images)}")
print(f"Classes Found: {class_map}")
print("-" * 30)

# ==========================================
# 3. DEFINE MODEL ARCHITECTURE
# ==========================================
# Architecture: Input (32x32x3) -> Conv3x3 -> ReLU -> Pool2x2 -> Flatten -> Linear
conv = Conv2d(in_channels=3, out_channels=8, kernel_size=3)
relu = ReLU()
pool = MaxPool2d(kernel_size=2)
flat = Flatten()
# After Conv(32->30) and Pool(30->15), feature map is 15x15
fc = Linear(in_features=8 * 15 * 15, out_features=len(class_map))

# Define Optimizer and Loss
params = conv.parameters() + fc.parameters()
optimizer = SGD(params, lr=LEARNING_RATE)
loss_fn = CrossEntropyLoss()

# ==========================================
# 4. MANDATORY METRIC - MACs / FLOPs
# ==========================================
# We run a single dummy forward pass to trigger the MACs counters in our layers
dummy_x = conv.forward([images[0]])
dummy_p = pool.forward(relu.forward(dummy_x.data))
dummy_l = fc.forward(flat.forward(dummy_p))

total_macs = conv.last_macs + fc.last_macs
print(f"METRIC - MACs per forward pass: {total_macs}")
print(f"METRIC - FLOPs per forward pass: {total_macs * 2}")
print("-" * 30)

# ==========================================
# 5. TRAINING LOOP
# ==========================================
print("\nStarting Training...")
start_train_time = time.time()

for epoch in range(EPOCHS):
    epoch_loss = 0
    epoch_acc = 0
    batches = 0
    
    for batch_imgs, batch_labels in get_batches(images, labels, BATCH_SIZE):
        optimizer.zero_grad()
        
        # --- Forward Pass ---
        # Note: relu.forward takes .data because it's a stateless activation
        x = conv.forward(batch_imgs)
        x = relu.forward(x.data)
        x = pool.forward(x)
        x = flat.forward(x)
        logits = fc.forward(x)
        
        # --- Loss & Accuracy ---
        loss_tensor, probs = loss_fn.forward(logits, batch_labels)
        acc = get_accuracy(probs, batch_labels)
        
        # --- Backward Pass ---
        loss_tensor.backward()
        
        # --- Update Weights ---
        optimizer.step()
        
        epoch_loss += loss_tensor.data
        epoch_acc += acc
        batches += 1
        
    avg_loss = epoch_loss / batches
    avg_acc = epoch_acc / batches
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}%")

total_train_time = time.time() - start_train_time
print(f"\nTraining Complete in {total_train_time/60:.2f} minutes.")

# ==========================================
# 6. SAVE WEIGHTS
# ==========================================
weights_file = f"{DATASET_NAME}_weights.pkl"
with open(weights_file, "wb") as f:
    pickle.dump({
        'conv_w': conv.weight.data, 
        'conv_b': conv.bias.data, 
        'fc_w': fc.weight.data, 
        'fc_b': fc.bias.data, 
        'class_map': class_map
    }, f)

print(f"Success! Weights saved to {weights_file}")
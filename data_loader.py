import cv2
import os      
import time
import random  

def load_dataset(dataset_name, mode='train', sample_limit=None):
    base_path = '/Volumes/codebase/Archive/deep learning/Assignment 1 Datasets'
    
    # Try the standard path first
    dataset_path = os.path.join(base_path, dataset_name, mode)
    
    # Fallback: if 'train' folder doesn't exist, look directly in the dataset folder
    if not os.path.exists(dataset_path):
        print(f"Note: '{mode}' folder not found, checking {dataset_name} directly...")
        dataset_path = os.path.join(base_path, dataset_name)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Could not find data in {dataset_name}")

    imgs, lbls = [], []
    # Get folders, excluding files like .zip or .DS_Store
    folders = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f)) and not f.startswith('.')])
    
    if not folders:
        raise ValueError(f"No class folders found in {dataset_path}. Are the images unzipped into folders?")

    cmap = {n: i for i, n in enumerate(folders)}
    start = time.time()

    for folder in folders:
        f_path = os.path.join(dataset_path, folder)
        files = [f for f in os.listdir(f_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if sample_limit:
            random.shuffle(files)
            files = files[:max(1, sample_limit // len(folders))]
            
        for f in files:
            img = cv2.imread(os.path.join(f_path, f))
            if img is None: continue
            
            img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (32, 32)).astype('float32')/255.0
            lst = img.tolist()
            imgs.append([[[lst[h][w][c] for w in range(32)] for h in range(32)] for c in range(3)])
            lbls.append(cmap[folder])
            
    return imgs, lbls, time.time()-start, cmap

def get_batches(imgs, lbls, b_sz):
    """
    Groups the loaded images and labels into small batches 
    to be processed by the model.
    """
    data = list(zip(imgs, lbls))
    random.shuffle(data) # Important for training stability
    for i in range(0, len(data), b_sz):
        batch = data[i:i+b_sz]
        # Separate the images and labels back into two lists
        yield [x[0] for x in b], [x[1] for x in b]
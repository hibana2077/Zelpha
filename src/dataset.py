import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from datasets import load_dataset
import numpy as np
from PIL import Image

class ScaleTransform:
    """
    Applies scaling to the image relative to a target size.
    s > 1: Zoom in (Resize larger then center crop)
    s < 1: Zoom out (Resize smaller then pad)
    """
    def __init__(self, scale, target_size):
        self.scale = scale
        self.target_size = target_size

    def __call__(self, img):
        if self.scale == 1.0:
            return F.resize(img, self.target_size)
        
        w, h = img.size
        new_w = int(w * self.scale)
        new_h = int(h * self.scale)
        
        # Resize
        img = F.resize(img, (new_h, new_w))
        
        if self.scale > 1.0:
            # Zoom in: Crop center
            img = F.center_crop(img, self.target_size)
        else:
            # Zoom out: Pad
            pad_w = (self.target_size[1] - new_w) // 2
            pad_h = (self.target_size[0] - new_h) // 2
            # Handle odd padding
            pad_w_2 = self.target_size[1] - new_w - pad_w
            pad_h_2 = self.target_size[0] - new_h - pad_h
            
            padding = (pad_w, pad_h, pad_w_2, pad_h_2)
            img = F.pad(img, padding, fill=0, padding_mode='constant')
            
        return img

class UCMercedDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, scale=1.0, image_size=256):
        self.dataset = hf_dataset
        self.base_transform = transform
        target_size = (image_size, image_size)
        self.scale_transform = ScaleTransform(scale, target_size=target_size) if scale != 1.0 else None
        self.default_resize = transforms.Resize(target_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']

        # Apply scaling if needed (Test time)
        if self.scale_transform:
            image = self.scale_transform(image)
        else:
            image = self.default_resize(image)

        # Apply standard transforms (ToTensor, Normalize)
        if self.base_transform:
            image = self.base_transform(image)

        return image, label

def get_dataloaders(batch_size=32, num_workers=4, test_scales=[0.7, 0.85, 1.0, 1.15, 1.3], image_size=256, dataset_name: str = "UC_Merced"):
    # Resolve dataset ID
    name_key = (dataset_name or "UC_Merced").lower()
    dataset_map = {
        "uc_merced": "blanchon/UC_Merced",
        "ucmerced": "blanchon/UC_Merced",
        "uc": "blanchon/UC_Merced",
        "uc merced": "blanchon/UC_Merced",
        "eurosat": "blanchon/EuroSAT_RGB",
        "eurosat_rgb": "blanchon/EuroSAT_RGB",
    }
    hf_id = dataset_map.get(name_key, dataset_name)

    # Load dataset
    print(f"Loading dataset: {dataset_name} ({hf_id})...")
    try:
        base_ds = load_dataset(hf_id, split="train")
    except Exception:
        # Fallback: load as DatasetDict and pick a split
        dsdict = load_dataset(hf_id)
        sel_split = "train" if "train" in dsdict else list(dsdict.keys())[0]
        base_ds = dsdict[sel_split]
        print(f"Using split: {sel_split}")

    # Split 60/20/20
    # First split train_val / test
    train_val_test = base_ds.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
    test_set = train_val_test['test']
    train_val = train_val_test['train']
    
    # Split train / val (0.25 of 0.8 is 0.2)
    train_val_split = train_val.train_test_split(test_size=0.25, seed=42, stratify_by_column="label")
    train_set = train_val_split['train']
    val_set = train_val_split['test']

    print(f"Split sizes: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

    # Determine number of classes
    num_classes = None
    try:
        label_feat = base_ds.features.get("label", None)
        if label_feat is not None and hasattr(label_feat, "names") and label_feat.names:
            num_classes = len(label_feat.names)
        elif label_feat is not None and hasattr(label_feat, "num_classes") and label_feat.num_classes:
            num_classes = int(label_feat.num_classes)
        else:
            # Fallback to unique labels from a subset for speed
            num_classes = len(set(base_ds[:2048]["label"]))
    except Exception:
        num_classes = len(set(base_ds[:2048]["label"]))

    # Transforms
    # Mean and Std for ImageNet are commonly used, or calculate for UCMerced
    # Using ImageNet stats for simplicity
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        normalize
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # Datasets
    train_ds = UCMercedDataset(train_set, transform=train_transform, image_size=image_size)
    val_ds = UCMercedDataset(val_set, transform=eval_transform, image_size=image_size)
    
    # Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Test Loaders for each scale
    test_loaders = {}
    for s in test_scales:
        test_ds = UCMercedDataset(test_set, transform=eval_transform, scale=s, image_size=image_size)
        test_loaders[s] = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loaders, num_classes

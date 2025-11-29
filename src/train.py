import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import numpy as np
import os
from ptflops import get_model_complexity_info
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans

from dataset import get_dataloaders
from model import ZelphaModel
from loss import ZelphaLoss

def calculate_metrics(logits, labels, num_classes=None):
    """
    Calculates Top-1, Top-5 Accuracy and F1 Score.
    """
    with torch.no_grad():
        # Top-1
        preds = logits.argmax(dim=1)
        acc1 = (preds == labels).sum().item() / labels.size(0)
        
        # Top-5
        # If num_classes < 5, top5 is always 1.0 if we just take top min(5, C)
        k = min(5, logits.size(1))
        _, topk_preds = logits.topk(k, dim=1)
        acc5 = 0
        for i in range(labels.size(0)):
            if labels[i] in topk_preds[i]:
                acc5 += 1
        acc5 /= labels.size(0)
        
        # F1 Score (Macro)
        # Pass explicit label set to avoid sklearn classification/regression ambiguity warnings
        if num_classes is not None:
            label_set = np.arange(num_classes)
            f1 = f1_score(
                labels.cpu().numpy(),
                preds.cpu().numpy(),
                average='macro',
                zero_division=0,
                labels=label_set,
            )
        else:
            f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)
        
    return acc1, acc5, f1

def calculate_margin(logits, dist_sq, labels, use_prototype=True):
    """
    Calculates margin statistics.
    For prototype: m(x) = min_{c!=y} (|z-mu_c| - |z-mu_y|)
    Note: dist_sq is |z-mu|^2.
    So |z-mu| = sqrt(dist_sq).
    """
    if not use_prototype or dist_sq is None:
        # Fallback for linear head: m(x) = logit_y - max_{c!=y} logit_c
        # This is standard margin definition for linear classifiers
        # logits [B, C]
        batch_size = logits.size(0)
        target_logits = logits[torch.arange(batch_size), labels] # [B]
        
        # Mask out target logits to find max of others
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(batch_size), labels] = False
        other_logits = logits[mask].view(batch_size, -1)
        max_other_logits, _ = other_logits.max(dim=1)
        
        margins = target_logits - max_other_logits
        return margins
    
    # Prototype Margin
    # dist_sq: [B, C]
    dists = torch.sqrt(torch.clamp(dist_sq, min=1e-8))
    batch_size = dists.size(0)
    
    target_dists = dists[torch.arange(batch_size), labels] # [B]
    
    mask = torch.ones_like(dists, dtype=torch.bool)
    mask[torch.arange(batch_size), labels] = False
    other_dists = dists[mask].view(batch_size, -1)
    min_other_dists, _ = other_dists.min(dim=1)
    
    # m(x) = min_other - target (positive is good)
    margins = min_other_dists - target_dists
    return margins

def select_device(requested: str):
    """Resolve best available device based on requested string.
    Supports 'auto', 'cuda', 'mps', and 'cpu'. Falls back gracefully.
    """
    req = (requested or "auto").lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if req == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if req == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device("cpu")

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, use_prototype=True):
    model.train()
    running_loss = 0.0
    
    # Metrics accumulators
    acc1_sum = 0.0
    acc5_sum = 0.0
    f1_sum = 0.0
    num_batches = 0
    
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logits, z, dist_sq = model(images)
        
        # Handle loss calculation based on head type
        if use_prototype:
            mu = model.classifier.mu
            loss, metrics = criterion(logits, dist_sq, labels, mu)
        else:
            # Linear head standard CE
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
            
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate metrics
        a1, a5, f1 = calculate_metrics(logits, labels, num_classes=model.num_classes)
        acc1_sum += a1
        acc5_sum += a5
        f1_sum += f1
        num_batches += 1
        
        if i % 10 == 0:
            print(f"Epoch {epoch} [{i}/{len(loader)}] Loss: {loss.item():.4f} Acc@1: {a1:.4f}")
        
    return {
        'loss': running_loss / num_batches,
        'acc1': acc1_sum / num_batches,
        'acc5': acc5_sum / num_batches,
        'f1': f1_sum / num_batches
    }

def initialize_prototypes_kmeans(model, loader, device, num_prototypes=1):
    """
    Initializes prototypes using K-Means on the features of each class.
    """
    print("Initializing prototypes with K-Means...")
    model.eval()
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            z = model.backbone(images)
            all_feats.append(z.cpu())
            all_labels.append(labels.cpu())
            
    all_feats = torch.cat(all_feats, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    num_classes = model.num_classes
    feature_dim = model.feature_dim
    
    new_mu = torch.zeros(num_classes, num_prototypes, feature_dim)
    
    for c in range(num_classes):
        mask = (all_labels == c)
        class_feats = all_feats[mask]
        
        if len(class_feats) < num_prototypes:
            # Fallback if not enough samples: just repeat mean
            if len(class_feats) > 0:
                mean = np.mean(class_feats, axis=0)
                new_mu[c] = torch.tensor(mean).unsqueeze(0).repeat(num_prototypes, 1)
            else:
                # No samples? Random init
                new_mu[c] = torch.randn(num_prototypes, feature_dim)
        else:
            kmeans = KMeans(n_clusters=num_prototypes, n_init=10, random_state=42)
            kmeans.fit(class_feats)
            centers = kmeans.cluster_centers_
            new_mu[c] = torch.tensor(centers)
            
    model.classifier.mu.data = new_mu.to(device)
    print("Prototypes initialized.")

def evaluate(model, test_loaders, device, use_prototype=True):
    model.eval()
    results = {}
    
    # Store predictions for Robust Accuracy
    all_preds = {}
    ground_truth = None
    
    # Margin stats
    all_margins = []
    
    print("\nEvaluating on Test Scales...")
    
    for scale, loader in test_loaders.items():
        scale_preds = []
        scale_gt = []
        
        # Metrics for this scale
        acc1_sum = 0
        acc5_sum = 0
        f1_sum = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                logits, _, dist_sq = model(images)
                
                # Metrics
                a1, a5, f1 = calculate_metrics(logits, labels, num_classes=model.num_classes)
                acc1_sum += a1
                acc5_sum += a5
                f1_sum += f1
                num_batches += 1
                
                # Predictions for Robust Acc
                preds = logits.argmax(dim=1)
                scale_preds.extend(preds.cpu().numpy())
                scale_gt.extend(labels.cpu().numpy())
                
                # Margins
                margins = calculate_margin(logits, dist_sq, labels, use_prototype)
                all_margins.extend(margins.cpu().numpy())
        
        results[f'scale_{scale}_acc1'] = acc1_sum / num_batches
        results[f'scale_{scale}_acc5'] = acc5_sum / num_batches
        results[f'scale_{scale}_f1'] = f1_sum / num_batches
        
        all_preds[scale] = scale_preds
        if ground_truth is None:
            ground_truth = scale_gt
            
    # Calculate Robust Accuracy
    num_samples = len(ground_truth)
    robust_count = 0
    
    for i in range(num_samples):
        is_robust = True
        gt = ground_truth[i]
        for scale in all_preds:
            if all_preds[scale][i] != gt:
                is_robust = False
                break
        if is_robust:
            robust_count += 1
            
    results['robust_acc'] = robust_count / num_samples
    
    # Margin Analysis
    all_margins = np.array(all_margins)
    results['margin_mean'] = np.mean(all_margins)
    results['margin_std'] = np.std(all_margins)
    results['margin_min'] = np.min(all_margins)
    results['margin_max'] = np.max(all_margins)
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--linear_epochs', type=int, default=10, help='Epochs for linear classifier training')
    parser.add_argument('--finetune_epochs', type=int, default=10, help='Epochs for prototype fine-tuning')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--finetune_lr', type=float, default=0.0001, help='Learning rate for fine-tuning')
    parser.add_argument('--proto_lr_scale', type=float, default=0.1, help='Scale factor for prototype learning rate')
    parser.add_argument('--device', type=str, default='auto', help="Device: 'auto'|'cuda'|'mps'|'cpu'")
    
    # Model & Ablation Args
    parser.add_argument('--model_name', type=str, default='zelpha', help='Model name: zelpha or timm model name (e.g. resnet18)')
    parser.add_argument('--no_lipschitz', action='store_true', help='Disable spectral normalization')
    parser.add_argument('--no_scale_pooling', action='store_true', help='Disable multi-scale pooling (use only scale 1.0)')
    
    # Prototype Args
    parser.add_argument('--num_prototypes', type=int, default=3, help='Number of prototypes per class')
    parser.add_argument('--beta', type=float, default=0.1, help='Weight for prototype loss (intra/inter)')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for inter-class loss')
    parser.add_argument('--image_size', type=int, default=256, help='Input image size (square).')
    parser.add_argument('--save_dir', type=str, default='', help='(Unused for disk) Kept for compatibility; checkpoints kept in memory.')
    
    args = parser.parse_args()
    
    device = select_device(args.device)
    print(f"Using device: {device.type}")
    print(f"Configuration: Model={args.model_name}, Lipschitz={not args.no_lipschitz}, ScalePool={not args.no_scale_pooling}")
    print(f"Prototype Config: K={args.num_prototypes}, Beta={args.beta}, Margin={args.margin}")
    
    # Data
    train_loader, val_loader, test_loaders = get_dataloaders(batch_size=args.batch_size, image_size=args.image_size)
    
    # Print dataset info
    print(f"\nDataset Info:")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Get number of classes from a batch
    sample_batch = next(iter(train_loader))
    num_classes_detected = len(torch.unique(sample_batch[1]))
    print(f"Number of classes (detected from batch): {num_classes_detected}")
    
    # Model - Start with Linear Head
    model = ZelphaModel(
        num_classes=21, 
        model_name=args.model_name,
        use_spectral_norm=not args.no_lipschitz,
        use_prototype=False, # Start with Linear
        use_scale_pooling=not args.no_scale_pooling,
        num_prototypes=args.num_prototypes
    ).to(device)
    
    # FLOPs & Params
    try:
        macs, params = get_model_complexity_info(model, (3, args.image_size, args.image_size), as_strings=True, print_per_layer_stat=False, verbose=False)
        print(f'{args.model_name:<30}  {macs:<8} GMACs   {params:<8} params')
    except Exception as e:
        print(f"Could not calculate FLOPs: {e}")

    # Track best checkpoints in memory (no disk writes)
    best_linear_acc = -1.0
    best_proto_acc = -1.0
    best_linear_state = None
    best_proto_state = None

    # --- Phase 1: Linear Training ---
    print("\n=== Phase 1: Linear Classifier Training ===")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.linear_epochs)
    
    for epoch in range(args.linear_epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, None, device, epoch, 
            use_prototype=False
        )
        
        # Validation
        model.eval()
        val_acc1_sum = 0
        val_batches = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits, _, _ = model(images)
                a1, _, _ = calculate_metrics(logits, labels, num_classes=model.num_classes)
                val_acc1_sum += a1
                val_batches += 1
        val_acc = val_acc1_sum / val_batches
        
        # Keep best linear model by val_acc in memory
        if val_acc > best_linear_acc:
            best_linear_acc = val_acc
            best_linear_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch}: Loss={train_metrics['loss']:.4f}, Train Acc={train_metrics['acc1']:.4f}, Val Acc={val_acc:.4f} (best {best_linear_acc:.4f})")
        scheduler.step()

    # --- Phase 2: Prototype Initialization ---
    print("\n=== Phase 2: Prototype Initialization ===")
    # Switch to Prototype Head
    # Load best Phase 1 (linear) weights before switching head
    if best_linear_state is not None:
        model.load_state_dict(best_linear_state, strict=False)
    model.set_classifier(type='prototype', num_prototypes=args.num_prototypes)
    model.to(device)
    
    # Initialize Prototypes
    initialize_prototypes_kmeans(model, train_loader, device, num_prototypes=args.num_prototypes)
    
    # --- Phase 3: Prototype Fine-tuning ---
    print("\n=== Phase 3: Prototype Fine-tuning ===")
    
    # Loss
    criterion = ZelphaLoss(lambda_intra=args.beta, lambda_inter=args.beta, margin=args.margin)
    
    # Optimizer with different LR for prototypes
    # Separate parameters
    backbone_params = list(model.backbone.parameters())
    classifier_params = list(model.classifier.parameters())
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.finetune_lr},
        {'params': classifier_params, 'lr': args.finetune_lr * args.proto_lr_scale} # Slower update for prototypes
    ])
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)
    
    for epoch in range(args.finetune_epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, 
            use_prototype=True
        )
        
        # Validation
        model.eval()
        val_acc1_sum = 0
        val_batches = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits, _, dist_sq = model(images)
                a1, _, _ = calculate_metrics(logits, labels, num_classes=model.num_classes)
                val_acc1_sum += a1
                val_batches += 1
        val_acc = val_acc1_sum / val_batches
        
        # Keep best prototype model by val_acc in memory
        if val_acc > best_proto_acc:
            best_proto_acc = val_acc
            best_proto_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch}: Loss={train_metrics['loss']:.4f}, Train Acc={train_metrics['acc1']:.4f}, Val Acc={val_acc:.4f} (best {best_proto_acc:.4f})")
        scheduler.step()

    # Load best prototype checkpoint before final evaluation (if kept in memory)
    if best_proto_state is not None:
        model.load_state_dict(best_proto_state, strict=False)
        model.to(device)

    # Final Evaluation
    results = evaluate(model, test_loaders, device, use_prototype=True)
    print("\nFinal Results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()

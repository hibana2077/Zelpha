import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import numpy as np
import os
from ptflops import get_model_complexity_info
from sklearn.metrics import f1_score

from dataset import get_dataloaders
from model import ZelphaModel
from loss import ZelphaLoss

def calculate_metrics(logits, labels):
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

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, warmup=False, use_prototype=True):
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
        else:
            mu = None # Not used for linear head loss usually, or handled differently

        if warmup and use_prototype:
            # Only CE loss during warmup for prototype model
            loss, metrics = criterion(logits, dist_sq, labels, mu)
            loss = metrics['loss_ce'] 
        elif use_prototype:
            loss, metrics = criterion(logits, dist_sq, labels, mu)
        else:
            # Linear head standard CE
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
            
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate metrics
        a1, a5, f1 = calculate_metrics(logits, labels)
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

def update_prototypes(model, loader, device):
    """
    Updates prototypes to be the mean of features for each class.
    Used during warmup.
    """
    if not hasattr(model.classifier, 'mu'):
        return

    model.eval()
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            z = model.backbone(images)
            all_feats.append(z)
            all_labels.append(labels)
            
    all_feats = torch.cat(all_feats, dim=0)
    all_labels = torch.cat(all_labels, dim=0).to(device)
    
    new_mu = model.classifier.mu.data.clone()
    for c in range(model.classifier.num_classes):
        mask = (all_labels == c)
        if mask.sum() > 0:
            class_mean = all_feats[mask].mean(dim=0)
            new_mu[c] = class_mean
            
    model.classifier.mu.data = new_mu
    print("Prototypes updated to class means.")

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
                a1, a5, f1 = calculate_metrics(logits, labels)
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
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model & Ablation Args
    parser.add_argument('--model_name', type=str, default='zelpha', help='Model name: zelpha or timm model name (e.g. resnet18)')
    parser.add_argument('--no_lipschitz', action='store_true', help='Disable spectral normalization')
    parser.add_argument('--no_prototype', action='store_true', help='Use linear head instead of prototype')
    parser.add_argument('--no_scale_pooling', action='store_true', help='Disable multi-scale pooling (use only scale 1.0)')
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    print(f"Configuration: Model={args.model_name}, Lipschitz={not args.no_lipschitz}, Proto={not args.no_prototype}, ScalePool={not args.no_scale_pooling}")
    
    # Data
    train_loader, val_loader, test_loaders = get_dataloaders(batch_size=args.batch_size)
    
    # Model
    model = ZelphaModel(
        num_classes=21, 
        model_name=args.model_name,
        use_spectral_norm=not args.no_lipschitz,
        use_prototype=not args.no_prototype,
        use_scale_pooling=not args.no_scale_pooling
    ).to(args.device)
    
    # FLOPs & Params
    try:
        macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=False, verbose=False)
        print(f'{args.model_name:<30}  {macs:<8} GMACs   {params:<8} params')
    except Exception as e:
        print(f"Could not calculate FLOPs: {e}")

    # Loss & Optimizer
    criterion = ZelphaLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training Loop
    for epoch in range(args.epochs):
        is_warmup = epoch < args.warmup_epochs
        
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, args.device, epoch, 
            warmup=is_warmup, use_prototype=not args.no_prototype
        )
        
        if is_warmup and not args.no_prototype:
            update_prototypes(model, train_loader, args.device)
            
        # Validation
        model.eval()
        val_acc1_sum = 0
        val_batches = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(args.device), labels.to(args.device)
                logits, _, _ = model(images)
                a1, _, _ = calculate_metrics(logits, labels)
                val_acc1_sum += a1
                val_batches += 1
        val_acc = val_acc1_sum / val_batches
        
        print(f"Epoch {epoch}: Loss={train_metrics['loss']:.4f}, Train Acc={train_metrics['acc1']:.4f}, Val Acc={val_acc:.4f}")
        
        scheduler.step()

    # Final Evaluation
    results = evaluate(model, test_loaders, args.device, use_prototype=not args.no_prototype)
    print("\nFinal Results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()

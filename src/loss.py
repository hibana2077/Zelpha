import torch
import torch.nn as nn
import torch.nn.functional as F

class ZelphaLoss(nn.Module):
    def __init__(self, lambda_intra=0.1, lambda_inter=0.1, margin=1.0):
        super().__init__()
        self.lambda_intra = lambda_intra
        self.lambda_inter = lambda_inter
        self.margin = margin
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, dist_sq, targets, model_prototypes):
        """
        logits: [B, C]
        dist_sq: [B, C] (squared distances to each prototype, minimized over K)
        targets: [B]
        model_prototypes: [C, K, D] or [C, D]
        """
        # 1. Cross Entropy
        loss_ce = self.ce_loss(logits, targets)

        # 2. Intra-class (Pull)
        # dist_sq[i, targets[i]] is the distance to the correct prototype (nearest one)
        # We can gather it.
        batch_size = logits.size(0)
        # Create index tensor
        target_indices = targets.view(-1, 1) # [B, 1]
        intra_dists = torch.gather(dist_sq, 1, target_indices) # [B, 1]
        loss_intra = intra_dists.mean()

        # 3. Inter-class (Push)
        # Pairwise distance between prototypes
        # Handle multi-prototype case
        if model_prototypes.dim() == 3:
            C, K, D = model_prototypes.shape
            all_protos = model_prototypes.view(-1, D) # [CK, D]
            
            # Pairwise distance
            mu_sq = all_protos.pow(2).sum(dim=1, keepdim=True) # [CK, 1]
            proto_dist_sq = mu_sq + mu_sq.t() - 2 * torch.mm(all_protos, all_protos.t()) # [CK, CK]
            proto_dist = torch.sqrt(torch.clamp(proto_dist_sq, min=1e-8))
            
            # Mask: We want to ignore distances between prototypes of the SAME class.
            # Create label tensor for prototypes: [0, 0, ..., 1, 1, ...]
            proto_labels = torch.arange(C, device=model_prototypes.device).unsqueeze(1).expand(C, K).reshape(-1)
            
            # Mask where labels are equal
            label_mask = proto_labels.unsqueeze(0) == proto_labels.unsqueeze(1) # [CK, CK]
            
            proto_dist_masked = proto_dist.clone()
            proto_dist_masked[label_mask] = self.margin + 1.0
            
            # Count valid pairs
            num_valid_pairs = (C * K) * (C * K) - label_mask.sum()
            if num_valid_pairs > 0:
                loss_inter = torch.clamp(self.margin - proto_dist_masked, min=0).pow(2).sum() / num_valid_pairs
            else:
                loss_inter = torch.tensor(0.0, device=logits.device)

        else:
            # Old single prototype logic
            mu = model_prototypes
            mu_sq = mu.pow(2).sum(dim=1, keepdim=True) # [C, 1]
            proto_dist_sq = mu_sq + mu_sq.t() - 2 * torch.mm(mu, mu.t()) # [C, C]
            
            proto_dist = torch.sqrt(torch.clamp(proto_dist_sq, min=1e-8))
            
            mask = torch.eye(mu.size(0), device=mu.device).bool()
            proto_dist_masked = proto_dist.clone()
            proto_dist_masked[mask] = self.margin + 1.0 
            
            loss_inter = torch.clamp(self.margin - proto_dist_masked, min=0).pow(2).sum() / (mu.size(0) * (mu.size(0) - 1))

        total_loss = loss_ce + self.lambda_intra * loss_intra + self.lambda_inter * loss_inter
        
        return total_loss, {
            "loss_ce": loss_ce,
            "loss_intra": loss_intra,
            "loss_inter": loss_inter
        }

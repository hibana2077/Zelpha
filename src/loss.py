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
        dist_sq: [B, C] (squared distances to each prototype)
        targets: [B]
        model_prototypes: [C, D]
        """
        # 1. Cross Entropy
        loss_ce = self.ce_loss(logits, targets)

        # 2. Intra-class (Pull)
        # dist_sq[i, targets[i]] is the distance to the correct prototype
        # We can gather it.
        batch_size = logits.size(0)
        # Create index tensor
        target_indices = targets.view(-1, 1) # [B, 1]
        intra_dists = torch.gather(dist_sq, 1, target_indices) # [B, 1]
        loss_intra = intra_dists.mean()

        # 3. Inter-class (Push)
        # Pairwise distance between prototypes
        # |mu_c - mu_c'|^2
        # We can reuse the logic: |a-b|^2 = |a|^2 + |b|^2 - 2ab
        mu = model_prototypes
        mu_sq = mu.pow(2).sum(dim=1, keepdim=True) # [C, 1]
        proto_dist_sq = mu_sq + mu_sq.t() - 2 * torch.mm(mu, mu.t()) # [C, C]
        
        # We want sqrt distance for margin? 
        # imp.md says: max(0, m - |mu_c - mu_c'|)^2
        # So we need Euclidean distance, not squared.
        proto_dist = torch.sqrt(torch.clamp(proto_dist_sq, min=1e-8))
        
        # Mask diagonal (distance to self is 0, we don't want to push self away)
        mask = torch.eye(mu.size(0), device=mu.device).bool()
        # We only care about c != c'
        # But we can just set diagonal to infinity or > margin so loss is 0
        proto_dist_masked = proto_dist.clone()
        proto_dist_masked[mask] = self.margin + 1.0 
        
        loss_inter = torch.clamp(self.margin - proto_dist_masked, min=0).pow(2).sum() / (mu.size(0) * (mu.size(0) - 1))

        total_loss = loss_ce + self.lambda_intra * loss_intra + self.lambda_inter * loss_inter
        
        return total_loss, {
            "loss_ce": loss_ce,
            "loss_intra": loss_intra,
            "loss_inter": loss_inter
        }

import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (SimCLR loss).
    """
    def __init__(self, temperature: float = 0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Computes NT-Xent loss for a batch of pairs (z_i, z_j).
        Args:
            z_i: Representations of the first view (batch_size, dim)
            z_j: Representations of the second view (batch_size, dim)
        """
        batch_size = z_i.shape[0]
        
        # Helper to compute cosine similarity
        # Normalize representations to unit hypersphere
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate for similarity matrix calculation
        # Represents [z_i, z_j]
        representations = torch.cat([z_i, z_j], dim=0) # (2*N, dim)
        
        # Similarity matrix
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        # Mask out self-contrast cases
        mask = torch.eye(2 * batch_size, device=z_i.device).bool()
        
        # Labels: positive for i is j (offset by batch_size) and vice versa
        # We want to maximize similarity between i and j.
        # But for standard implementation, it's often easier to do manually or use ready implementation logic.
        
        # Let's derive the logits.
        # sim(i, j) / temperature
        logits = similarity_matrix / self.temperature
        
        # For each element i in 0..N-1, positive is i+N
        # For each element i in N..2N-1, positive is i-N
        
        # We remove the diagonal (self-similarity)
        # This implementation can be slightly tricky to get efficient.
        
        # Efficient SimCLR implementation:
        # 1. Positives are top-right and bottom-left quadrants' diagonals of the similarity matrix of (z_i, z_j) 
        #    if we didn't cat them efficiently.
        #    With cat-ed representations:
        #    Row i (0..N-1): Positive is column i+N.
        #    Row i+N (N..2N-1): Positive is column i.
            
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z_i.device),
            torch.arange(0, batch_size, device=z_i.device)
        ], dim=0)
        
        # Mask out self-similarity (diagonal) from logits for proper softmax?
        # Standard CrossEntropyLoss will take care of the denominator if we provide all logits.
        # We just need to ensure the self-similarity is very low or removed so it's not picked.
        # Typically we perform log_softmax on (logits - large_mask).
        
        # Let's keep it simple: use the mask to filter logits passed to CrossEntropy
        # But CrossEntropy expects a fixed number of classes. Here the "classes" are other samples.
        # If we remove diagonal, we have 2N-1 classes.
        
        logits = logits[~mask].view(2 * batch_size, -1)
        
        # Adjust labels because we removed one element (the diagonal)
        # If label was < index (on diagonal), it stays same.
        # If label was > index, it decreases by 1.
        
        # Actually, simpler way:
        # exp(sim(i,j)/t) / sum_{k!=i} exp(sim(i,k)/t)
        
        # Positive pairs are (i, i+N) and (i+N, i)
        # We can extract positive logits.
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        
        # Denominator: sum over all other samples
        # sum_{k!=i} exp(sim(i,k)/t)
        # We can sum rows of exp(logits), subtract exp(self_sim/t)
        
        denominator = torch.sum(torch.exp(similarity_matrix / self.temperature), dim=1) - \
                      torch.exp(torch.diag(similarity_matrix) / self.temperature)
        
        loss = -torch.log(nominator / denominator).mean()
        
        return loss

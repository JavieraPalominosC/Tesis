import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=256,
                 commitment_cost=0.25, decay=0.99, epsilon=1e-5,
                 reset_threshold=1.0, reset_every=10):
        super().__init__()
        self.num_embeddings   = num_embeddings
        self.embedding_dim    = embedding_dim
        self.commitment_cost  = commitment_cost
        self.decay            = decay
        self.epsilon          = epsilon
        self.reset_threshold  = reset_threshold  # umbral bajo el cual se resetea
        self.reset_every      = reset_every       # cada cuántos steps se revisa
        self._step            = 0

        embedding = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer('embedding', embedding)
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_dw', embedding.clone())

    def forward(self, z_e):
        B, D, H, W = z_e.shape
        z_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D)

        distances = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.embedding.t()
            + self.embedding.pow(2).sum(1)
        )
        indices = distances.argmin(1)
        one_hot = F.one_hot(indices, self.num_embeddings).float()

        if self.training:
            with torch.no_grad():
                cluster_size = one_hot.sum(0)
                dw = one_hot.t() @ z_flat

                self.ema_cluster_size.mul_(self.decay).add_(
                    cluster_size * (1 - self.decay))
                self.ema_dw.mul_(self.decay).add_(
                    dw * (1 - self.decay))

                n = self.ema_cluster_size.sum()
                smoothed = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon) * n
                )
                self.embedding.data.copy_(
                    self.ema_dw / smoothed.unsqueeze(1))

                # Codebook reset: reinicializar embeddings poco usados
                self._step += 1
                if self._step % self.reset_every == 0:
                    dead = self.ema_cluster_size < self.reset_threshold
                    n_dead = dead.sum().item()
                    if n_dead > 0:
                        # Tomar vectores aleatorios del batch actual
                        perm = torch.randperm(z_flat.size(0), device=z_flat.device)
                        n_replace = min(n_dead, z_flat.size(0))
                        new_vecs = z_flat[perm[:n_replace]].detach().float()
                        dead_idx = dead.nonzero(as_tuple=False).squeeze(1)[:n_replace]
                        self.embedding.data[dead_idx] = new_vecs
                        self.ema_cluster_size[dead_idx] = self.reset_threshold
                        self.ema_dw.data[dead_idx] = new_vecs * self.reset_threshold

        z_q_flat = self.embedding[indices]
        z_q = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = self.commitment_cost * commitment_loss
        codebook_loss = torch.tensor(0.0, device=z_e.device)

        avg_probs  = one_hot.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        z_q = z_e + (z_q - z_e).detach()
        indices_2d = indices.view(B, H, W)

        return z_q, vq_loss, indices_2d, perplexity, codebook_loss, commitment_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import math


class TransformerBlock(nn.Module):
    """Bloque Transformer bidireccional (estilo BERT)."""
    def __init__(self, hidden_dim, n_heads, ff_mult=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads,
                                          dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x


class MaskGIT(L.LightningModule):
    """
    Prior model tipo MaskGIT para aprender p(s) sobre tokens del VQ-VAE.

    Durante training: enmascara tokens aleatoriamente y los predice.
    Durante inference: desenmascara iterativamente para generar/completar secuencias.
    Para anomaly detection: calcula -log p(s) sobre la secuencia completa.

    Args:
        num_tokens:    tamaño del codebook (512)
        seq_len:       largo de la secuencia de tokens (32*32=1024)
        hidden_dim:    dimensión del Transformer
        n_layers:      número de capas Transformer
        n_heads:       número de heads de atención
        ff_mult:       multiplicador del feedforward
        dropout:       dropout
        learning_rate: learning rate
    """
    def __init__(self,
                 num_tokens=512,
                 seq_len=1024,
                 hidden_dim=512,
                 n_layers=8,
                 n_heads=8,
                 ff_mult=4,
                 dropout=0.1,
                 learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.num_tokens = num_tokens
        self.seq_len    = seq_len
        self.mask_token = num_tokens  # token especial para máscara
        self.learning_rate = learning_rate

        # Embeddings
        self.tok_emb = nn.Embedding(num_tokens + 1, hidden_dim)  # +1 para mask token
        self.pos_emb = nn.Embedding(seq_len, hidden_dim)

        # Transformer bidireccional
        self.blocks = nn.Sequential(*[
            TransformerBlock(hidden_dim, n_heads, ff_mult, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_tokens)

    def forward(self, x):
        """
        x: (B, seq_len) — secuencia de índices (puede tener mask_token)
        retorna logits: (B, seq_len, num_tokens)
        """
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        emb = self.tok_emb(x) + self.pos_emb(pos)
        out = self.blocks(emb)
        out = self.norm(out)
        return self.head(out)

    def _mask_tokens(self, tokens):
        """
        Enmascara una fracción aleatoria de tokens.
        La fracción se samplea de una distribución coseno (como en MaskGIT).
        """
        B, T = tokens.shape
        # Samplear ratio de enmascaramiento de distribución coseno
        r = torch.rand(B, device=tokens.device)
        r = (1 - torch.cos(r * math.pi)) / 2  # distribución coseno
        n_mask = (r * T).long().clamp(min=1)

        masked = tokens.clone()
        mask = torch.zeros(B, T, dtype=torch.bool, device=tokens.device)
        for i in range(B):
            perm = torch.randperm(T, device=tokens.device)
            mask[i, perm[:n_mask[i]]] = True
            masked[i, mask[i]] = self.mask_token

        return masked, mask

    def _step(self, batch, stage):
        tokens = batch  # (B, seq_len)
        masked, mask = self._mask_tokens(tokens)

        logits = self(masked)  # (B, seq_len, num_tokens)

        # Loss solo sobre tokens enmascarados
        loss = F.cross_entropy(
            logits[mask],       # (n_masked, num_tokens)
            tokens[mask],       # (n_masked,)
        )

        self.log(f"{stage}/loss", loss, prog_bar=True,
                 on_step=True, on_epoch=False, sync_dist=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @torch.no_grad()
    def anomaly_score(self, tokens):
        """
        Calcula el anomaly score = -log p(s) para cada secuencia.
        Usa masked prediction: para cada token, calcula la log-prob
        condicionada en todos los demás tokens.

        tokens: (B, seq_len)
        retorna: (B,) scores — mayor score = más anómalo
        """
        self.eval()
        B, T = tokens.shape
        total_nll = torch.zeros(B, device=tokens.device)

        # Para cada posición, enmascarar ese token y predecir
        # (aproximación eficiente: enmascarar todos a la vez)
        for t in range(T):
            masked = tokens.clone()
            masked[:, t] = self.mask_token
            logits = self(masked)  # (B, T, num_tokens)
            log_probs = F.log_softmax(logits[:, t, :], dim=-1)  # (B, num_tokens)
            token_log_prob = log_probs[torch.arange(B), tokens[:, t]]  # (B,)
            total_nll -= token_log_prob

        return total_nll / T  # normalizar por largo

    @torch.no_grad()
    def generate_counterfactual(self, tokens, anomaly_positions, n_steps=10,
                                temperatura=True, modo='muestreo', k=0):
        """
        Genera un contrafactual enmascarando las posiciones anómalas
        y remuestreando iterativamente.

        tokens:            (B, seq_len) secuencia original
        anomaly_positions: (B, seq_len) bool mask — True donde hay anomalía
        n_steps:           número de pasos de remuestreo iterativo
        modo:              'muestreo' multinomial sobre todo el vocabulario, con
                           recocido si temperatura=True; 'greedy' toma la moda;
                           'topk' muestrea entre los k más probables a T=1.
        k:                 tamaño del top-k cuando modo='topk'
        retorna:           (B, seq_len) secuencia contrafactual

        Para un contrafactual la moda es lo que se busca: no queremos una
        muestra plausible de lo normal sino la completación más plausible. Como
        el revelado es progresivo y lo revelado queda fijo, un token sacado de
        la cola en los primeros pasos contamina el contexto de todo el resto.
        """
        self.eval()
        B, T = tokens.shape

        # Inicializar con tokens originales, enmascarar posiciones anómalas
        cf_tokens = tokens.clone()
        cf_tokens[anomaly_positions] = self.mask_token

        # Remuestreo iterativo estilo MaskGIT
        for step in range(n_steps):
            logits = self(cf_tokens)  # (B, T, num_tokens)

            # Samplear tokens para posiciones enmascaradas
            still_masked = (cf_tokens == self.mask_token)
            if not still_masked.any():
                break

            # Recocido de temperatura: baja de 1.0 hasta 1/n_steps, de modo que
            # las primeras revelaciones exploran y las ultimas son casi
            # deterministas. Antes se calculaba esta linea y nunca se aplicaba.
            if temperatura and modo == 'muestreo':
                t = max(1.0 - step / n_steps, 1e-2)
                probs = F.softmax(logits / t, dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)

            for b in range(B):
                mask_pos = still_masked[b].nonzero(as_tuple=True)[0]
                if len(mask_pos) == 0:
                    continue
                p = probs[b, mask_pos, :]  # (n_mask, num_tokens)
                fila = torch.arange(len(mask_pos), device=p.device)
                if modo == 'greedy':
                    confidence, sampled = p.max(dim=-1)
                elif modo == 'topk':
                    v, idx = p.topk(min(k, p.shape[-1]), dim=-1)
                    j = torch.multinomial(v / v.sum(-1, keepdim=True), 1).squeeze(1)
                    sampled = idx[fila, j]
                    confidence = p[fila, sampled]
                else:
                    sampled = torch.multinomial(p, num_samples=1).squeeze(1)
                    confidence = p[fila, sampled]
                n_reveal = max(1, len(mask_pos) // (n_steps - step))
                top_idx = confidence.topk(n_reveal).indices
                cf_tokens[b, mask_pos[top_idx]] = sampled[top_idx]

        return cf_tokens

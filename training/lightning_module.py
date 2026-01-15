import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from models.clip_model import CLIPDualEncoder

class CLIPLightningModule(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.model = CLIPDualEncoder()
        self.lr = lr

    def contrastive_loss(self, img_emb, txt_emb):
        logits = img_emb @ txt_emb.T / self.model.temperature
        labels = torch.arange(len(img_emb), device=self.device)

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)

        return (loss_i + loss_t) / 2

    def training_step(self, batch, batch_idx):
        images, texts = batch
        images = images.to(self.device)
        texts = {k: v.to(self.device) for k, v in texts.items()}

        img_emb, txt_emb = self.model(images, texts)
        loss = self.contrastive_loss(img_emb, txt_emb)

        # Log for Lightning
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        # Explicit print for reports
        avg_loss = self.trainer.callback_metrics["train_loss"].item()
        print(f"[Epoch {self.current_epoch}] Train Loss: {avg_loss:.4f}")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


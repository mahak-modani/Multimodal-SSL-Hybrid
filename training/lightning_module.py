import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from models.clip_model import CLIPDualEncoder
from utils.masking import random_patch_mask


class CLIPLightningModule(pl.LightningModule):
    def __init__(self, phase=1, lr=1e-4, recon_weight=0.5):
        """
        phase = 1 -> contrastive-only (Phase 1)
        phase = 2 -> contrastive + reconstruction (Phase 2)
        """
        super().__init__()

        self.phase = phase
        self.lr = lr
        self.recon_weight = recon_weight

        self.model = CLIPDualEncoder()

        self.save_hyperparameters()

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

        if self.phase == 1:
            img_emb, txt_emb = self.model(images, texts)
            loss = self.contrastive_loss(img_emb, txt_emb)

            self.log("loss_contrastive", loss, on_epoch=True, prog_bar=True)
            self.log("loss_total", loss, on_epoch=True, prog_bar=True)

            return loss

        else:
            masked_images, _ = random_patch_mask(
                images,
                mask_ratio=0.5
            )

            img_emb, txt_emb, recon = self.model(
                images,
                texts,
                masked_images
            )

            loss_contrastive = self.contrastive_loss(img_emb, txt_emb)
            loss_recon = F.mse_loss(recon, images)

            loss = loss_contrastive + self.recon_weight * loss_recon

            # Logging (important for paper)
            self.log("loss_contrastive", loss_contrastive, on_epoch=True)
            self.log("loss_reconstruction", loss_recon, on_epoch=True)
            self.log("loss_total", loss, on_epoch=True, prog_bar=True)

            return loss

    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics

        if "loss_total" in metrics:
            total = metrics["loss_total"].item()
            print(f"[Epoch {self.current_epoch}] Train Loss: {total:.4f}")

            if self.phase == 2:
                c = metrics["loss_contrastive"].item()
                r = metrics["loss_reconstruction"].item()
                print(
                    f"    Contrastive: {c:.4f} | Reconstruction: {r:.4f}"
                )

    # Optimizer
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    # Save Model
    def on_train_end(self):
        if self.phase == 1:
            path = "phase1_clip_model.pth"
        else:
            path = "phase2_clip_model.pth"

        torch.save(self.model.state_dict(), path)
        print(f"Saved trained model to {path}")

import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from models.clip_model import CLIPDualEncoder
from utils.masking import random_patch_mask


class CLIPLightningModule(pl.LightningModule):
    def __init__(self, phase=1, lr=1e-4, recon_weight=0.5):
        """
        phase = 1 -> contrastive-only baseline
        phase = 2 -> contrastive + reconstruction
        """
        super().__init__()

        self.phase = phase
        self.lr = lr
        self.recon_weight = recon_weight

        self.model = CLIPDualEncoder()
        self.save_hyperparameters()

    # Contrastive Loss (InfoNCE)
    def contrastive_loss(self, img_emb, txt_emb):
        logits = img_emb @ txt_emb.T / self.model.temperature
        labels = torch.arange(len(img_emb), device=self.device)

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)

        return (loss_i + loss_t) / 2

    # Training Step
    def training_step(self, batch, batch_idx):
        images, texts = batch

        images = images.to(self.device)
        texts = {k: v.to(self.device) for k, v in texts.items()}

        # ---------- Phase 1 ----------
        if self.phase == 1:
            img_emb, txt_emb = self.model(images, texts)

            loss_contrastive = self.contrastive_loss(img_emb, txt_emb)
            loss_total = loss_contrastive

            self.log("loss_contrastive", loss_contrastive, on_epoch=True)
            self.log("loss_total", loss_total, on_epoch=True)

            return loss_total

        # ---------- Phase 2 ----------
        masked_images, _ = random_patch_mask(images, mask_ratio=0.5)

        img_emb, txt_emb, recon = self.model(
            images,
            texts,
            masked_images
        )

        loss_contrastive = self.contrastive_loss(img_emb, txt_emb)
        target = F.interpolate(
          images,
          size=recon.shape[-2:],  # match 7x7 or 8x8
          mode="bilinear",
          align_corners=False
        )

        loss_reconstruction = F.mse_loss(recon, target)

        loss_total = loss_contrastive + self.recon_weight * loss_reconstruction

        self.log("loss_contrastive", loss_contrastive, on_epoch=True)
        self.log("loss_reconstruction", loss_reconstruction, on_epoch=True)
        self.log("loss_total", loss_total, on_epoch=True)

        return loss_total

    # Epoch-End Console Logging
    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics

        if self.phase == 1:
            total = metrics["loss_total"].item()
            print(f"[Epoch {self.current_epoch}] loss_total = {total:.4f}")

        else:
            c = metrics["loss_contrastive"].item()
            r = metrics["loss_reconstruction"].item()
            t = metrics["loss_total"].item()

            print(
                f"[Epoch {self.current_epoch}] "
                f"loss_contrastive = {c:.4f} | "
                f"loss_reconstruction = {r:.4f} | "
                f"loss_total = {t:.4f}"
            )

    # Optimizer
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    # Save Model
    def on_train_end(self):
        path = (
            "phase1_clip_model.pth"
            if self.phase == 1
            else "phase2_clip_model.pth"
        )

        torch.save(self.model.state_dict(), path)
        print(f"Saved trained model to {path}")

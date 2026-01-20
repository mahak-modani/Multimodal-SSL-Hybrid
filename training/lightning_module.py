import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from models.clip_model import CLIPDualEncoder
from utils.masking import random_patch_mask


class CLIPLightningModule(pl.LightningModule):
    def __init__(self, mode="contrastive", lr=1e-4, recon_weight=0.5):
        """
        mode ∈ {"contrastive", "reconstruction", "hybrid"}
        """
        super().__init__()

        self.mode = mode
        self.lr = lr
        self.recon_weight = recon_weight

        self.model = CLIPDualEncoder()
        self.save_hyperparameters()

    # ---------- Contrastive loss ----------
    def contrastive_loss(self, img_emb, txt_emb):
        logits = img_emb @ txt_emb.T / self.model.temperature
        labels = torch.arange(len(img_emb), device=self.device)

        return 0.5 * (
            F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)
        )

    # ---------- Training step ----------
    def training_step(self, batch, batch_idx):
        images, texts = batch
        images = images.to(self.device)
        texts = {k: v.to(self.device) for k, v in texts.items()}

        # ---- Contrastive only ----
        if self.mode == "contrastive":
            img_emb, txt_emb = self.model(images, texts)
            loss = self.contrastive_loss(img_emb, txt_emb)
            self.log("loss_total", loss, on_epoch=True)
            return loss

        # ---- Reconstruction only ----
        if self.mode == "reconstruction":
            masked_images, _ = random_patch_mask(images)
            recon = self.model(images, None, masked_images)

            target = F.interpolate(
                images,
                size=recon.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            loss = F.mse_loss(recon, target)
            self.log("loss_reconstruction", loss, on_epoch=True)
            return loss

        # ---- Hybrid ----
        masked_images, _ = random_patch_mask(images)
        img_emb, txt_emb, recon = self.model(images, texts, masked_images)

        loss_c = self.contrastive_loss(img_emb, txt_emb)

        target = F.interpolate(
            images,
            size=recon.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        loss_r = F.mse_loss(recon, target)
        loss = loss_c + self.recon_weight * loss_r

        self.log("loss_contrastive", loss_c, on_epoch=True)
        self.log("loss_reconstruction", loss_r, on_epoch=True)
        self.log("loss_total", loss, on_epoch=True)

        return loss

    def on_train_epoch_end(self):
        m = self.trainer.callback_metrics
        if self.mode == "contrastive":
            print(f"[Epoch {self.current_epoch}] loss_total = {m['loss_total']:.4f}")
        elif self.mode == "reconstruction":
            print(f"[Epoch {self.current_epoch}] loss_reconstruction = {m['loss_reconstruction']:.4f}")
        else:
            print(
                f"[Epoch {self.current_epoch}] "
                f"contrastive = {m['loss_contrastive']:.4f} | "
                f"reconstruction = {m['loss_reconstruction']:.4f} | "
                f"total = {m['loss_total']:.4f}"
            )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

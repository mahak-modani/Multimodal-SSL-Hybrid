import os
import json
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from models.clip_model import CLIPDualEncoder
from utils.masking import random_patch_mask
from utils.results_logger import ResultsLogger


class CLIPLightningModule(pl.LightningModule):
    def __init__(
        self,
        mode="contrastive",
        lr=1e-4,
        recon_weight=0.5,
        batch_size=32,
        save_dir="results/tmp"
    ):
        super().__init__()

        self.mode = mode
        self.lr = lr
        self.recon_weight = recon_weight
        self.batch_size = batch_size
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

        self.model = CLIPDualEncoder()

        # CSV logger (epoch-level)
        self.results_logger = ResultsLogger(
            save_dir=self.save_dir,
            filename="training_metrics.csv"
        )

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

    # ---------- Epoch logging ----------
    def on_train_epoch_end(self):
        m = self.trainer.callback_metrics
        epoch = self.current_epoch

        loss_c = m.get("loss_contrastive")
        loss_r = m.get("loss_reconstruction")
        loss_t = m.get("loss_total")

        self.results_logger.log(
            epoch,
            loss_c.item() if loss_c is not None else None,
            loss_r.item() if loss_r is not None else None,
            loss_t.item() if loss_t is not None else None
        )

        # Minimal console logging (paper-safe)
        if self.mode == "contrastive":
            print(f"[Epoch {epoch}] loss_total = {loss_t:.4f}")
        elif self.mode == "reconstruction":
            print(f"[Epoch {epoch}] loss_reconstruction = {loss_r:.4f}")
        else:
            print(
                f"[Epoch {epoch}] "
                f"contrastive = {loss_c:.4f} | "
                f"reconstruction = {loss_r:.4f} | "
                f"total = {loss_t:.4f}"
            )

    # ---------- Config saving ----------
    def on_train_start(self):
        config = self.get_run_config()

        with open(os.path.join(self.save_dir, "run_config.json"), "w") as f:
            json.dump(config, f, indent=4)

    def on_train_end(self):
        self.results_logger.close()

    def get_run_config(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "training_mode": self.mode,
            "learning_rate": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.trainer.max_epochs,
            "optimizer": "AdamW",
            "precision": "16-mixed",

            "image_encoder": "resnet18",
            "text_encoder": "MiniLM-L6-v2",
            "projection_dim": 256,
            "temperature_learnable": True,

            "reconstruction_weight": (
                self.recon_weight if self.mode == "hybrid" else None
            ),

            # PEFT / LoRA placeholders
            "peft_method": "none",
            "trainable_parameters": trainable_params,
            "total_parameters": total_params
        }

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

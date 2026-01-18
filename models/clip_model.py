import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoModel

from models.mae_decoder import MAEDecoder


class CLIPDualEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        # -------------------------
        # Vision Encoder
        # -------------------------
        resnet = models.resnet18(pretrained=True)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.vision_dim = 512

        # -------------------------
        # Text Encoder
        # -------------------------
        self.text_encoder = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_dim = self.text_encoder.config.hidden_size

        # -------------------------
        # Projection Heads
        # -------------------------
        self.image_proj = nn.Linear(self.vision_dim, embed_dim)
        self.text_proj = nn.Linear(self.text_dim, embed_dim)

        # -------------------------
        # Temperature
        # -------------------------
        self.temperature = nn.Parameter(torch.tensor(0.07))

        # -------------------------
        # MAE Decoder (Phase 2)
        # -------------------------
        self.mae_decoder = MAEDecoder(in_dim=self.vision_dim)

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, images, texts, masked_images=None):
        """
        Phase 1:
            forward(images, texts)
            -> img_emb, txt_emb

        Phase 2:
            forward(images, texts, masked_images)
            -> img_emb, txt_emb, reconstruction
        """

        # ----- Image encoder -----
        img_feat = self.vision_encoder(images).squeeze(-1).squeeze(-1)

        # ----- Text encoder -----
        txt_out = self.text_encoder(**texts)
        txt_feat = txt_out.last_hidden_state[:, 0]

        # ----- Projection -----
        img_emb = F.normalize(self.image_proj(img_feat), dim=-1)
        txt_emb = F.normalize(self.text_proj(txt_feat), dim=-1)

        # ----- Phase 1 -----
        if masked_images is None:
            return img_emb, txt_emb

        # ----- Phase 2: reconstruction -----
        masked_feat = self.vision_encoder(masked_images)

        recon = self.mae_decoder(masked_feat)

        return img_emb, txt_emb, recon

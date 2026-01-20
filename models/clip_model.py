import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoModel

from models.mae_decoder import MAEDecoder


class CLIPDualEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        # -------- Vision Encoder --------
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.vision_dim = 512

        # -------- Text Encoder --------
        self.text_encoder = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_dim = self.text_encoder.config.hidden_size

        # -------- Projection Heads --------
        self.image_proj = nn.Linear(self.vision_dim, embed_dim)
        self.text_proj = nn.Linear(self.text_dim, embed_dim)

        # -------- Temperature --------
        self.temperature = nn.Parameter(torch.tensor(0.07))

        # -------- MAE Decoder --------
        self.mae_decoder = MAEDecoder(in_dim=self.vision_dim)

    def forward(self, images, texts=None, masked_images=None):
        """
        contrastive: forward(images, texts)
        reconstruction / hybrid: forward(images, texts, masked_images)
        """

        # ---- Image features ----
        img_feat = self.vision_encoder(images)
        img_feat_vec = img_feat.mean(dim=[2, 3])

        img_emb = F.normalize(
            self.image_proj(img_feat_vec), dim=-1
        )

        # ---- Contrastive-only ----
        if texts is not None and masked_images is None:
            txt_out = self.text_encoder(**texts)
            txt_feat = txt_out.last_hidden_state[:, 0]

            txt_emb = F.normalize(
                self.text_proj(txt_feat), dim=-1
            )

            return img_emb, txt_emb

        # ---- Reconstruction-only / Hybrid ----
        masked_feat = self.vision_encoder(masked_images)
        recon = self.mae_decoder(masked_feat)

        if texts is None:
            return recon

        txt_out = self.text_encoder(**texts)
        txt_feat = txt_out.last_hidden_state[:, 0]

        txt_emb = F.normalize(
            self.text_proj(txt_feat), dim=-1
        )

        return img_emb, txt_emb, recon

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchvision.models import resnet18, ResNet18_Weights


class CLIPDualEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        # ✅ SAFE vision encoder (NO torch.hub)
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.vision_encoder = resnet18(weights=weights)
        self.vision_encoder.fc = nn.Identity()

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Projection heads
        self.image_proj = nn.Linear(512, embed_dim)
        self.text_proj = nn.Linear(384, embed_dim)

        # Temperature parameter (CLIP-style)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, images, text_inputs):
        img_features = self.vision_encoder(images)
        txt_features = self.text_encoder(**text_inputs).last_hidden_state[:, 0]

        img_emb = F.normalize(self.image_proj(img_features), dim=-1)
        txt_emb = F.normalize(self.text_proj(txt_features), dim=-1)

        return img_emb, txt_emb

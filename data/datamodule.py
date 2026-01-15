import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import torchvision.transforms as T


class ImageTextDataModule:
    def __init__(self, dataset_name="beans", batch_size=32):
        self.dataset_name = dataset_name
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def setup(self):
        self.dataset = load_dataset(self.dataset_name, split="train")
        self.dataset = self.dataset.shuffle(seed=42).select(range(1000))

    def collate_fn(self, batch):
      images = torch.stack([
          self.transform(x["image"]) for x in batch
      ])
      # Semantically meaningful captions
      label_to_caption = {
        0: "a healthy bean leaf with no visible disease",
        1: "a bean leaf affected by angular leaf spot disease",
        2: "a bean leaf infected with bean rust disease"
      }

      captions = [
        label_to_caption[int(x["labels"])] for x in batch
      ]

      texts = self.tokenizer(
        captions,
        padding=True,
        truncation=True,
        return_tensors="pt"
      ) 

      return images, texts


    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=2
        )



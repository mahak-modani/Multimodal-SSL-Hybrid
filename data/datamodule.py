import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torchvision.transforms as T

from data.dataset import Flickr30kDataset


class ImageTextDataModule:
    def __init__(
        self,
        csv_path,
        image_root,
        batch_size=32
      ):
        self.csv_path = csv_path
        self.image_root = image_root
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
        self.dataset = Flickr30kDataset(
        csv_path=self.csv_path,
        image_root=self.image_root,
        transform=self.transform
      )


    def collate_fn(self, batch):
        images = torch.stack([x[0] for x in batch])
        captions = [x[1] for x in batch]

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
            num_workers=0
        )

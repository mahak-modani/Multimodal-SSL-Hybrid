import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from data.flickr_dataset import Flickr30kDataset
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
        if self.dataset_name == "flickr30k":
            self.dataset = Flickr30kDataset(
                csv_path="/content/drive/MyDrive/Multimodal-SSL-Hybrid/flickr30k_5k.csv",
                transform=self.transform
            )
        else:
            raise ValueError("Unsupported dataset")

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
            num_workers=2
        )



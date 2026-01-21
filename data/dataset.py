import torch
import pandas as pd
from PIL import Image


class Flickr30kDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
      row = self.df.iloc[idx]

      # ---- Resolve image path robustly ----
      if "image_path" in row:
        image_path = row["image_path"]

      elif "image_name" in row:
        image_path = os.path.join(self.image_root, row["image_name"])

      elif "file_name" in row:
        image_path = os.path.join(self.image_root, row["file_name"])

      elif "image_id" in row:
        image_name = f"{int(row['image_id']):012d}.jpg"
        image_path = os.path.join(self.image_root, image_name)

      else:
        raise KeyError(
            "CSV must contain one of: image_path, image_name, file_name, image_id"
        )

      image = Image.open(image_path).convert("RGB")

      if self.transform:
        image = self.transform(image)

      caption = row["caption"]

      return image, caption

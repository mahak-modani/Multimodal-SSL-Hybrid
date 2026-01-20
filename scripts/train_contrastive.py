from data.datamodule import ImageTextDataModule
from training.lightning_module import CLIPLightningModule
import pytorch_lightning as pl


def main():
    data = ImageTextDataModule(
      csv_path="flickr30k_1k.csv",
      batch_size=32
    )
    data.setup()

    model = CLIPLightningModule(
      mode="contrastive",
      save_dir="results/flickr1k"
    )

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu",
        devices=1,
        precision="16-mixed"
    )

    trainer.fit(model, data.train_dataloader())


if __name__ == "__main__":
    main()


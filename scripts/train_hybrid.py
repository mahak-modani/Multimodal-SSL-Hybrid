from data.datamodule import ImageTextDataModule
from training.lightning_module import CLIPLightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl


def main():
    data = ImageTextDataModule(
      csv_path="flickr30k_1k.csv",
      batch_size=32
    )
    data.setup()

    model = CLIPLightningModule(
      mode="hybrid",
      batch_size=32,
      recon_weight=0.5,
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


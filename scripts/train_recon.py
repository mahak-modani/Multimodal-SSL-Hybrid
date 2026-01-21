from data.datamodule import ImageTextDataModule
from training.lightning_module import CLIPLightningModule
import pytorch_lightning as pl


def main():
    split = "1k"          # change to 2k / 5k
    mode = "reconstruction"

    csv_path = f"flickr30k_{split}.csv"
    save_dir = f"results/flickr30k/{split}/{mode}"

    data = ImageTextDataModule(
        csv_path=csv_path,
        batch_size=32
    )
    data.setup()

    model = CLIPLightningModule(
        mode=mode,
        batch_size=32,
        save_dir=save_dir
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

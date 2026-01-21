from data.datamodule import ImageTextDataModule
from training.lightning_module import CLIPLightningModule
import pytorch_lightning as pl


def main():
    split_name = "1k"
    mode = "reconstruction"

    save_dir = f"results/flickr{split_name}/{mode}"

    data = ImageTextDataModule(
        dataset_name="flickr30k",
        split=split_name,
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

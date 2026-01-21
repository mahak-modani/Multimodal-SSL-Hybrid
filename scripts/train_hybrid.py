from pytorch_lightning.callbacks import ModelCheckpoint
from data.datamodule import ImageTextDataModule
from training.lightning_module import CLIPLightningModule
import pytorch_lightning as pl


def main():
    split = "1k"          # change to 2k / 5k
    mode = "hybrid"

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

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"{mode}_flickr{split}",
        save_last=True,
        save_top_k=1
    )

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )

    trainer.fit(model, data.train_dataloader())


if __name__ == "__main__":
    main()

from data.datamodule import ImageTextDataModule
from training.lightning_module import CLIPLightningModule
import pytorch_lightning as pl

def main():
    data = ImageTextDataModule(
        dataset_name="flickr30k",
        batch_size=16
    )
    data.setup()

    model = CLIPLightningModule(phase=2)  # <-- KEY CHANGE

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        log_every_n_steps=10
    )

    trainer.fit(model, data.train_dataloader())

if __name__ == "__main__":
    main()


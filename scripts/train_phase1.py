import pytorch_lightning as pl
from data.datamodule import ImageTextDataModule
from training.lightning_module import CLIPLightningModule

def main():
    data = ImageTextDataModule(
    dataset_name="beans",
    batch_size=32
    )
    data.setup()

    model = CLIPLightningModule()

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu",
        devices=1,
        precision=16,
        log_every_n_steps=10
    )

    trainer.fit(model, data.train_dataloader())

if __name__ == "__main__":
    main()


from pytorch_lightning.callbacks import ModelCheckpoint
from data.datamodule import ImageTextDataModule
from training.lightning_module import CLIPLightningModule
import pytorch_lightning as pl


def main():
    dataset_name = "flickr30k"
    split_name = "1k"   # change to 2k / 5k when needed
    mode = "contrastive"  
    
    data = ImageTextDataModule(
        dataset_name=dataset_name,
        split=split_name,
        batch_size=32
    )
    data.setup()

    model = CLIPLightningModule(mode=mode)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"{mode}_flickr{split_name}",
        save_last=True,
        save_top_k=1,
        monitor=None
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


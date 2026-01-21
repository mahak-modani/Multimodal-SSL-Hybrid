import torch
from data.datamodule import ImageTextDataModule
from training.lightning_module import CLIPLightningModule


@torch.no_grad()
def retrieval_test(
    csv_path,
    image_root,
    checkpoint_path,
    batch_size=16
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    data = ImageTextDataModule(
        csv_path=csv_path,
        image_root=image_root,
        batch_size=batch_size
    )
    data.setup()
    loader = data.train_dataloader()

    # Load Lightning model
    model = CLIPLightningModule.load_from_checkpoint(
        checkpoint_path,
        mode="contrastive"
    ).to(device)
    model.eval()

    clip = model.model  # underlying CLIPDualEncoder

    images, texts = next(iter(loader))
    images = images.to(device)
    texts = {k: v.to(device) for k, v in texts.items()}

    img_emb, txt_emb = clip(images, texts)
    sims = img_emb @ txt_emb.T
    preds = sims.argmax(dim=1)

    recall = (preds == torch.arange(len(preds), device=device)).float().mean()
    print(f"Recall@1 (batch): {recall.item():.3f}")


if __name__ == "__main__":
    retrieval_test(
        csv_path="coco_val_1p.csv",
        image_root="/content/drive/MyDrive/coco/val2017",
        checkpoint_path="checkpoints/hybrid_flickr1k.ckpt"
    )

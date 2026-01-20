import torch
from data.datamodule import ImageTextDataModule
from models.clip_model import CLIPDualEncoder


@torch.no_grad()
def retrieval_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = ImageTextDataModule(dataset_name="flickr30k", batch_size=16)
    data.setup()
    loader = data.train_dataloader()

    model = CLIPDualEncoder().to(device)
    model.load_state_dict(torch.load("phase2_clip_model.pth", map_location=device))
    model.eval()

    images, texts = next(iter(loader))
    images = images.to(device)
    texts = {k: v.to(device) for k, v in texts.items()}

    img_emb, txt_emb = model(images, texts)
    sims = img_emb @ txt_emb.T
    preds = sims.argmax(dim=1)

    recall = (preds == torch.arange(len(preds), device=device)).float().mean()
    print(f"Recall@1 (batch): {recall.item():.3f}")


if __name__ == "__main__":
    retrieval_test()

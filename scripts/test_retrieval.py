import json
import torch
from data.datamodule import ImageTextDataModule
from models.clip_model import CLIPDualEncoder


@torch.no_grad()
def retrieval_test(csv_path, checkpoint_path, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = ImageTextDataModule(
        csv_path=csv_path,
        batch_size=16
    )
    data.setup()
    loader = data.train_dataloader()

    model = CLIPDualEncoder().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    images, texts = next(iter(loader))
    images = images.to(device)
    texts = {k: v.to(device) for k, v in texts.items()}

    img_emb, txt_emb = model(images, texts)
    sims = img_emb @ txt_emb.T
    preds = sims.argmax(dim=1)

    recall = (preds == torch.arange(len(preds), device=device)).float().mean().item()

    with open(output_path, "w") as f:
        json.dump({"Recall@1": recall}, f, indent=4)

    print(f"Recall@1: {recall:.3f}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    retrieval_test(
        csv_path="flickr30k_1k.csv",
        checkpoint_path="checkpoints/contrastive_flickr1k.ckpt",
        output_path="results/flickr1k/contrastive/retrieval.json"
    )

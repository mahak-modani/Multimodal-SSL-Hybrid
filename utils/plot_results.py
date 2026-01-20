import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_losses(csv_path, title, out_path):
    df = pd.read_csv(csv_path)

    plt.figure()
    if "loss_contrastive" in df:
        plt.plot(df["epoch"], df["loss_contrastive"], label="Contrastive")
    if "loss_reconstruction" in df:
        plt.plot(df["epoch"], df["loss_reconstruction"], label="Reconstruction")
    if "loss_total" in df:
        plt.plot(df["epoch"], df["loss_total"], label="Total")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

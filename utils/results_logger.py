import csv
import os


class ResultsLogger:
    def __init__(self, save_dir, filename):
        os.makedirs(save_dir, exist_ok=True)
        self.path = os.path.join(save_dir, filename)

        self.file = open(self.path, "w", newline="")
        self.writer = csv.writer(self.file)

        # Header
        self.writer.writerow([
            "epoch",
            "loss_contrastive",
            "loss_reconstruction",
            "loss_total"
        ])

    def log(self, epoch, loss_c=None, loss_r=None, loss_t=None):
        self.writer.writerow([
            epoch,
            loss_c,
            loss_r,
            loss_t
        ])
        self.file.flush()

    def close(self):
        self.file.close()

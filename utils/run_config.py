import json
import os


def save_run_config(save_dir, filename, config_dict):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)

    with open(path, "w") as f:
        json.dump(config_dict, f, indent=4)

    print(f"[INFO] Saved run config to {path}")

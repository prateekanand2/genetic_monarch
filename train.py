import argparse
import time
import os
import copy
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import pyjuice as juice
import pyjuice.nodes.distributions as dists


def parse_args():
    parser = argparse.ArgumentParser(description="Train HCLT with PyJuice + Early Stopping")

    # Required
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--valid_path", type=str, required=True)

    # Model
    parser.add_argument("--num_latents", type=int, default=128)
    parser.add_argument("--pseudocount", type=float, default=0.005)

    # Training
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=128)

    # Early stopping
    parser.add_argument("--patience", type=int, default=50,
                        help="Number of epochs without improvement before stopping")
    parser.add_argument("--min_delta", type=float, default=1e-4,
                        help="Minimum validation LL improvement to reset patience")

    # Misc
    parser.add_argument("--log_path", type=str, default="training.log")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)

    return parser.parse_args()


def load_txt_data(path):
    data = np.loadtxt(path, dtype=np.int8, delimiter=" ")
    return torch.tensor(data, dtype=torch.long)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)
    print("CUDA devices:", torch.cuda.device_count())
    print("CUDA version:", torch.version.cuda)

    # -------------------------
    # Load data
    # -------------------------
    train_data = load_txt_data(args.train_path)
    valid_data = load_txt_data(args.valid_path)

    print("Train shape:", train_data.shape)
    print("Valid shape:", valid_data.shape)

    train_loader = DataLoader(
        TensorDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True
    )

    valid_loader = DataLoader(
        TensorDataset(valid_data),
        batch_size=args.batch_size,
        shuffle=False
    )

    # -------------------------
    # Build HCLT (full train set)
    # -------------------------
    print("Constructing HCLT...")
    ns = juice.structures.HCLT(
        train_data.float().to(device),
        num_latents=args.num_latents,
        input_dist=dists.Categorical(num_cats=2),
    )

    pc = juice.compile(ns)
    pc.to(device)

    # Warmup for CUDA graph
    if device.type == "cuda":
        with torch.cuda.device(pc.device):
            for batch in train_loader:
                x = batch[0].to(device)
                lls = pc(x, record_cudagraph=True)
                lls.mean().backward()
                break

    # -------------------------
    # Early stopping state
    # -------------------------
    best_val_ll = -float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = None

    os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)

    with open(args.log_path, "w") as log_file:
        if device.type == "cuda":
            torch.cuda.synchronize()

        train_start_time = time.perf_counter()

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            pc.init_param_flows(flows_memory=0.0)

            # ---- Train ----
            train_ll = 0.0
            for batch in train_loader:
                x = batch[0].to(device)
                lls = pc(x)
                lls.mean().backward()
                train_ll += lls.mean().item()

            pc.mini_batch_em(
                step_size=1.0,
                pseudocount=args.pseudocount
            )

            train_ll /= len(train_loader)
            t1 = time.time()

            # ---- Validation ----
            test_ll = 0.0
            with torch.no_grad():
                for batch in valid_loader:
                    x = batch[0].to(device)
                    lls = pc(x)
                    test_ll += lls.mean().item()

            test_ll /= len(valid_loader)
            t2 = time.time()

            # ---- Early stopping check ----
            improvement = test_ll - best_val_ll

            if improvement > args.min_delta:
                best_val_ll = test_ll
                best_epoch = epoch
                patience_counter = 0
                best_state = copy.deepcopy(pc.state_dict())
                status = "✓ improved"
            else:
                patience_counter += 1
                status = f"no improve ({patience_counter}/{args.patience})"

            log_line = (
                f"[Epoch {epoch}] "
                f"[train LL: {train_ll:.4f}; val LL: {test_ll:.4f}] "
                f"[train {t1 - t0:.2f}s; val {t2 - t1:.2f}s] "
                f"{status}"
            )

            print(log_line)
            log_file.write(log_line + "\n")
            log_file.flush()

            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        # Restore best model
        if best_state is not None:
            pc.load_state_dict(best_state)
            print(f"Restored best model from epoch {best_epoch} (val LL: {best_val_ll:.4f})")

        if device.type == "cuda":
            torch.cuda.synchronize()

        total_time = time.perf_counter() - train_start_time
        print(f"\nTotal training time: {total_time/3600:.2f} hours")

        log_file.write(f"\nBest epoch: {best_epoch}\n")
        log_file.write(f"Best val LL: {best_val_ll:.6f}\n")
        log_file.write(f"Total time: {total_time:.2f} seconds\n")

    # Optional save
    if args.save_path is not None:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        juice.save(args.save_path, pc)
        print(f"Best model saved to {args.save_path}")


if __name__ == "__main__":
    main()
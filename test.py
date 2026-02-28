import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pyjuice as juice


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HCLT model on test data")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved .jpc model")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path to test .txt file")

    parser.add_argument("--batch_size", type=int, default=256)
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

    # -------------------------
    # Load model
    # -------------------------
    print("Loading model...")
    ns = juice.load(args.model_path)
    pc = juice.compile(ns)
    pc.to(device)
    pc.eval()

    # -------------------------
    # Load test data
    # -------------------------
    print("Loading test data...")
    test_data = load_txt_data(args.test_path)

    print("Test shape:", test_data.shape)

    test_loader = DataLoader(
        TensorDataset(test_data),
        batch_size=args.batch_size,
        shuffle=False
    )

    # -------------------------
    # Compute Test LL
    # -------------------------
    total_ll = 0.0

    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            lls = pc(x)
            total_ll += lls.sum().item()

    avg_ll = total_ll / len(test_data)

    print("\n=============================")
    print(f"Average Test Log-Likelihood: {avg_ll:.6f}")
    print("=============================")


if __name__ == "__main__":
    main()
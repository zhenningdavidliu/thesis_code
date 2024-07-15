import argparse
import numpy as np
import pandas as pd
import random
import math
from tqdm import tqdm

import torch
import matplotlib.pyplot as plt

import os


def test_p(
    p, input_dim, output_dim, epsilon, x, A, y, lr=1e-3, n_iter: int = 1000
) -> float:

    # Make matrix
    chosen_indices = random.sample(range(input_dim * output_dim), p)
    A_mask = torch.zeros((output_dim, input_dim))
    for i in chosen_indices:
        A_mask[i // input_dim, i % input_dim] = 1.0
    assert math.isclose(A_mask.sum().item(), p)
    A_perturb = torch.randn(output_dim, input_dim) * A_mask
    A_perturb = epsilon * A_perturb / torch.sqrt(torch.sum((A_perturb) ** 2))

    # Training
    A_perturb.requires_grad_(True)
    opt_A = torch.optim.Adam([A_perturb], lr=lr)
    for _ in range(n_iter):
        assert math.isclose(A_perturb.norm().item(), epsilon, rel_tol=1e-4), (
            A_perturb.norm().item(),
            epsilon,
        )

        # Calculate optimal perturbed input x
        with torch.no_grad():
            A_total = A + A_mask * A_perturb
            X_opt = torch.linalg.solve(A_total.T @ A_total, A_total.T @ y.T)

        # Loss
        opt_A.zero_grad()
        y_pred = X_opt.T @ (A + A_mask * A_perturb).T
        loss = torch.mean((y - y_pred) ** 2)

        # Gradient update
        loss.backward()
        opt_A.step()

        # Re-normalize
        A_perturb.data = (
            epsilon * A_perturb.data / torch.sqrt(torch.sum((A_perturb.data) ** 2))
        )

    return loss.item(), X_opt - x.T


if __name__ == "__main__":

    # Arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input_dim", type=int, default=50)
    # parser.add_argument("--output_dim", type=int, default=300)
    # parser.add_argument("--k", type=int, default=1000)
    # parser.add_argument("--epsilon", type=float, default=1e-1)
    # parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    # args = parser.parse_args()

    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("--input_dim", type=int, default=50)
    parser.add_argument("--output_dim", type=int, default=300)
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--epsilon", type=float, default=1e-1)
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--experiment_number", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 4, 5])
    args = parser.parse_args()

    # Run
    all_losses = []
    # all_max_diffs = []
    for seed in tqdm(args.seeds):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        x = torch.randn(args.k, args.input_dim)
        A = torch.randn(args.output_dim, args.input_dim)
        y = x @ A.T
        ps = np.linspace(0, args.input_dim * args.output_dim, 100).astype(int)[1:]
        outputs = [
            test_p(
                p=p,
                input_dim=args.input_dim,
                output_dim=args.output_dim,
                epsilon=args.epsilon,
                x=x,
                A=A,
                y=y,
                n_iter=args.n_iter,
            )
            for p in tqdm(ps, leave=False)
        ]
        losses = [output[0] for output in outputs]
        # max_diffs = [output[1].abs().max().item() for output in outputs]

        all_losses.append(losses)
        # all_max_diffs.append(max_diffs)

    # Create directory for experiment
    os.makedirs(f"experiments/E_{args.experiment_number}", exist_ok=True)
    filepath = "experiments/E_{args.experiment_number}"

    # Save and plot
    all_losses = np.array(all_losses)
    filename = os.path.join(
        f"experiments/E_{args.experiment_number}", "shadow_main.npz"
    )
    np.savez(filename, all_losses=all_losses, ps=ps)
    p_cutoff = args.input_dim * (args.output_dim - args.input_dim)
    plt.axvline(p_cutoff, color="red", linestyle="--")
    plt.plot(ps, all_losses.mean(0))
    plt.fill_between(
        ps,
        all_losses.mean(0) - all_losses.std(0),
        all_losses.mean(0) + all_losses.std(0),
        alpha=0.5,
    )
    plt.title(f"Experiment {args.experiment_number}")
    filename = os.path.join(
        f"experiments/E_{args.experiment_number}", "shadow_main.png"
    )
    plt.savefig(filename)
    plt.close()

    # Create log-plot
    plt.axvline(p_cutoff, color="red", linestyle="--")
    plt.plot(ps, all_losses.mean(0))
    plt.fill_between(
        ps,
        all_losses.mean(0) - all_losses.std(0),
        all_losses.mean(0) + all_losses.std(0),
        alpha=0.5,
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"Log-plot of experiment {args.experiment_number}")
    filename = os.path.join(
        f"experiments/E_{args.experiment_number}", "shadow_main_log.png"
    )
    plt.savefig(filename)
    plt.close()

    # # Plot max diffs
    # all_max_diffs = np.array(all_max_diffs)
    # plt.plot(ps, all_max_diffs.mean(0))
    # plt.fill_between(
    #     ps,
    #     all_max_diffs.mean(0) - all_max_diffs.std(0),
    #     all_max_diffs.mean(0) + all_max_diffs.std(0),
    #     alpha=0.5,
    # )
    # plt.title(f"Max diffs in experiment {args.experiment_number}")
    # filename = os.path.join(
    #     f"experiments/E_{args.experiment_number}", "shadow_main_max_diffs.png"
    # )
    # plt.savefig(filename)
    # plt.close()

    # Save parameters as txt
    filename = os.path.join(f"experiments/E_{args.experiment_number}", "params.txt")
    with open(filename, "w") as f:
        f.write(str(args))

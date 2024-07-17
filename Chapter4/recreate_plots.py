import os
import matplotlib.pyplot as plt
import numpy as np

# Parameters for recreating the plot
input_dim = 50
output_dim = 100
experiment_number = 4

# Create directory for experiment
filepath = f"experiments/E_{experiment_number}"

# Load and plot
filename = os.path.join(f"experiments/E_{experiment_number}", "shadow_main.npz")
p_cutoff = input_dim * (output_dim - input_dim)
# np.savez(filename, all_losses=all_losses, ps=ps, p_cutoff=p_cutoff)
data = np.load(filename)
all_losses = data["all_losses"]
ps = data["ps"]
# Create plot

plt.axvline(p_cutoff, color="red", linestyle="--")
plt.plot(ps, all_losses.mean(0))
plt.fill_between(
    ps,
    all_losses.mean(0) - all_losses.std(0),
    all_losses.mean(0) + all_losses.std(0),
    alpha=0.5,
)
plt.title(f"Experiment {experiment_number}")
filename = os.path.join(f"experiments/E_{experiment_number}", "shadow_main.png")
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
plt.title(f"Log-plot of experiment {experiment_number}")
filename = os.path.join(f"experiments/E_{experiment_number}", "shadow_main_log.png")
plt.savefig(filename)
plt.close()

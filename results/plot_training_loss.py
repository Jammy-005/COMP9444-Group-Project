
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# ==========================================
# CONFIGURATION
# ==========================================

log_file = "paper1/2hop.txt"
window_size = 20

# ==========================================
# EXTRACT TRAINING LOSSES
# ==========================================

pattern = re.compile(
    r"(?:Train\s+)?Epoch:\s*(\d+)\s*\[(\d+)/(\d+).*?\]\s*Loss:\s*([\d.eE+-]+)"
)

records = []

with open(log_file, "r") as file:
    for line in file:

        match = pattern.search(line)

        if match:
            epoch = int(match.group(1))
            processed = int(match.group(2))
            total = int(match.group(3))
            loss = float(match.group(4))

            records.append((epoch, processed, total, loss))

# ==========================================
# VALIDATE DATA
# ==========================================

if not records:
    raise ValueError("No training losses found.")

if len(records) < window_size:
    raise ValueError(
        f"Need at least {window_size} loss measurements."
    )

# ==========================================
# CALCULATE CUMULATIVE TRAINING EXAMPLES
# ==========================================

epochs = sorted(set(r[0] for r in records))

# Total training examples in each epoch
epoch_totals = {}

for epoch, processed, total, loss in records:
    epoch_totals[epoch] = total

# Calculate cumulative offset for each epoch
epoch_offsets = {}
offset = 0

for epoch in epochs:
    epoch_offsets[epoch] = offset
    offset += epoch_totals[epoch]

# Extract losses and global training progress
train_losses = []
train_steps = []

for epoch, processed, total, loss in records:

    train_losses.append(loss)

    global_step = epoch_offsets[epoch] + processed
    train_steps.append(global_step)

# ==========================================
# CALCULATE 20-POINT MOVING AVERAGE
# ==========================================

moving_avg = []

for i in range(window_size - 1, len(train_losses)):

    avg = sum(
        train_losses[i - window_size + 1:i + 1]
    ) / window_size

    moving_avg.append(avg)

moving_steps = train_steps[window_size - 1:]

# ==========================================
# PLOT
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    moving_steps,
    moving_avg,
    linewidth=1.8,
    label=f"Training Loss ({window_size}-point Moving Average)"
)

plt.xlabel("Training Examples Processed", fontsize=12)
plt.ylabel("Training Loss", fontsize=12)

plt.title(
    "Training Loss - Moving Average",
    fontsize=14
)

plt.ylim(bottom=0)

plt.gca().xaxis.set_major_formatter(
    StrMethodFormatter("{x:,.0f}")
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig("training_loss_moving_average.png", dpi=300)
plt.close()

# ==========================================
# SUMMARY
# ==========================================

print(f"Epochs found: {epochs}")
print(f"Total loss measurements: {len(train_losses)}")

print(f"\nInitial moving average: {moving_avg[0]:.6f}")
print(f"Final moving average:   {moving_avg[-1]:.6f}")

print("\nGraph saved: training_loss_moving_average.png")

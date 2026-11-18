
import re
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================

log_file = "paper1/2hop.txt"

# ==========================================
# REGEX PATTERNS
# ==========================================

epoch_pattern = re.compile(
    r"(?:Train\s+)?Epoch:\s*(\d+)"
)

val_pattern = re.compile(
    r"Validation set:.*?Accuracy:\s*(\d+)/(\d+)"
)

# ==========================================
# EXTRACT VALIDATION ACCURACY
# ==========================================

val_accuracy = {}
current_epoch = None

with open(log_file, "r") as file:

    for line in file:

        # Track current epoch
        epoch_match = epoch_pattern.search(line)

        if epoch_match:
            current_epoch = int(epoch_match.group(1))

        # Extract validation accuracy
        val_match = val_pattern.search(line)

        if val_match and current_epoch is not None:

            correct = int(val_match.group(1))
            total = int(val_match.group(2))

            accuracy = 100.0 * correct / total

            val_accuracy[current_epoch] = accuracy

# ==========================================
# VALIDATE DATA
# ==========================================

if not val_accuracy:
    raise ValueError("No validation accuracy found.")

epochs = sorted(val_accuracy.keys())

accuracies = [
    val_accuracy[e] for e in epochs
]

# ==========================================
# PRINT RESULTS
# ==========================================

for epoch, accuracy in zip(epochs, accuracies):
    print(f"Epoch {epoch}: Validation Accuracy = {accuracy:.2f}%")

# ==========================================
# PLOT
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    accuracies,
    marker="o",
    linewidth=2,
    markersize=8,
    label="Validation Accuracy"
)

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)

plt.title(
    "Validation Accuracy vs Epoch",
    fontsize=14
)

plt.xticks(epochs)
plt.ylim(0, 100)

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig("validation_accuracy.png", dpi=300)
plt.close()

print("\nGraph saved: validation_accuracy.png")

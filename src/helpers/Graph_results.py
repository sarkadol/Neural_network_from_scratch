import matplotlib.pyplot as plt

with open("../../Java_to_Python.txt", "r") as file:
    lines = file.readlines()

# Parse the fields from the file
losses = eval(lines[0].split('=')[1].strip())
learning_rates = eval(lines[1].split('=')[1].strip())
train_vector_count = eval(lines[2].split('=')[1].strip())
layers = eval(lines[3].split('=')[1].strip())
batch_size = eval(lines[4].split('=')[1].strip())
decay_rate = eval(lines[5].split('=')[1].strip())
momentum = eval(lines[6].split('=')[1].strip())
learning_decay_rate = eval(lines[7].split('=')[1].strip())
clip_value = eval(lines[8].split('=')[1].strip())


# Create a figure with two subplots (stacked vertically)
fig, axs = plt.subplots(2, 1, figsize=(8, 8))  # 2 rows, 1 column

# Plot the losses in the first subplot
axs[0].plot(losses, marker='o', linestyle='-', color='blue', label='Cross-Entropy loss')
axs[0].set_title('Cross-Entropy loss', fontsize=16)
axs[0].set_xlabel('Epoch', fontsize=12)
axs[0].set_ylabel('Cross-Entropy loss', fontsize=12)
axs[0].legend(fontsize=10)
axs[0].grid(True, linestyle='--', alpha=0.7)
axs[0].set_ylim(bottom=0)  # For the losses plot

# Plot the learning rates in the second subplot
axs[1].plot(learning_rates, marker='o', linestyle='-', color='red', label='Learning rate')
axs[1].set_title('Learning rate', fontsize=16)
axs[1].set_xlabel('Epoch', fontsize=12)
axs[1].set_ylabel('Learning Rate', fontsize=12)
axs[1].legend(fontsize=10)
axs[1].grid(True, linestyle='--', alpha=0.7)

def evaluate():
    """
    Copied from the official evaluator, just the arguments (filenames) are static.
    :return:
    """
    with open('../../NEW_test_predictions.csv', 'r') as rf:
        pred = rf.read().split()
    with open('../../data/fashion_mnist_test_labels.csv', 'r') as rf:
        truth = rf.read().split()
    assert len(pred) == len(truth)
    hits = sum([p == t for p, t in zip(pred, truth)])
    return hits / len(truth)

# Add separate text boxes for hyperparameters and network details
hyperparameter_text = f"""
Hyperparameters:
- Batch size: {batch_size}
- Decay rate: {decay_rate}
- Momentum: {momentum}
- Learning decay rate: {learning_decay_rate}
- Clip value: {clip_value}
- Epochs: {len(losses)}
- Initial learning rate: {learning_rates[0]}
"""

network_text = f"""
Network:
- Layers: {layers}
- Training images: {train_vector_count}

ACCURACY: {evaluate()}
"""

# Add the text boxes to the figure
fig.text(0.02, 0.02, hyperparameter_text, ha='left', va='bottom', fontsize=12, multialignment='left',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

fig.text(0.5, 0.02, network_text, ha='left', va='bottom', fontsize=12, multialignment='left',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

# Adjust layout for better readability
plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space for the text boxes

# Display the plots
plt.show()

import matplotlib.pyplot as plt

# Paths when running with different environments
# Java_to_Python.txt
with open("Java_to_Python.txt", "r") as file:
    lines = file.readlines()

# Parse the fields from the file in the correct order
losses = eval(lines[0].split('=')[1].strip())
learning_rates = eval(lines[1].split('=')[1].strip())
train_vector_count = eval(lines[2].split('=')[1].strip())
layers = eval(lines[3].split('=')[1].strip())
batch_size = eval(lines[4].split('=')[1].strip())
decay_rate = eval(lines[5].split('=')[1].strip())
momentum = eval(lines[6].split('=')[1].strip())
learning_decay_rate = eval(lines[7].split('=')[1].strip())
clip_value = eval(lines[8].split('=')[1].strip())

# Define the evaluate function
def evaluate():
    """
    Copied from the official evaluator, just the arguments (filenames) are static.
    :return:
    """
    with open('NEW_test_predictions.csv', 'r') as rf:
        pred = rf.read().split()
    with open('data/fashion_mnist_test_labels.csv', 'r') as rf:
        truth = rf.read().split()
    assert len(pred) == len(truth)
    hits = sum([p == t for p, t in zip(pred, truth)])
    return hits / len(truth)

# Print the hyperparameters and network details
print(f"""PYTHON
Hyperparameters:
- Batch size: {batch_size}
- Decay rate: {decay_rate}
- Momentum: {momentum}
- Learning decay rate: {learning_decay_rate}
- Clip value: {clip_value}
- Epochs: {len(losses)}
- Initial learning rate: {learning_rates[0]}

Network:
- Layers: {layers}
- Training images: {train_vector_count}
- ACCURACY: {evaluate()}
""")

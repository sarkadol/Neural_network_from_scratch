import matplotlib.pyplot as plt

# paths when running with run_ours.sh:
#Java_to_Python.txt
# paths when running with IDE:
#../../Java_to_Python.txt
with open("Java_to_Python.txt", "r") as file:
    lines = file.readlines()

# Parse losses and learning_rates
losses = eval(lines[0].split('=')[1].strip())
learning_rates = eval(lines[1].split('=')[1].strip())
train_vector_count = eval(lines[2].split('=')[1].strip())
batch_size = eval(lines[3].split('=')[1].strip())
decay_rate = eval(lines[4].split('=')[1].strip())
layers = eval(lines[5].split('=')[1].strip())


def evaluate():
    """
    Copied from the official evaluator, just the arguments (filenames) are static.
    :return:
    """
    with open('NEW_test_predictions.csv','r') as rf:
        pred = rf.read().split()
    with open('data/fashion_mnist_test_labels.csv','r') as rf:
        truth = rf.read().split()
    assert len(pred) == len(truth)
    hits = sum([p == t for p, t in zip(pred, truth)])
    return hits/len(truth)

print(f"""PYTHON
Hyperparameters:
- Batch size: {batch_size}
- Decay rate: {decay_rate}
- Training images: {train_vector_count}
- Epochs: {len(losses)}
- Initial learning rate: {learning_rates[0]}
- Layers: {layers}
ACCURACY: {evaluate()}
""")
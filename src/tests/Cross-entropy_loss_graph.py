import matplotlib.pyplot as plt

# Your data

with open("../../losses_and_learning_rates.txt", "r") as file:
    lines = file.readlines()

# Parse losses and learning_rates
losses = eval(lines[0].split('=')[1].strip())
learning_rates = eval(lines[1].split('=')[1].strip())

print("Losses:", losses)
print("Learning Rates:", learning_rates)

# need to paste the output arrays here ->
#losses = [2.303363, 2.2959313, 2.2846365, 2.2814102, 2.2799768, 2.2650354, 2.240415, 2.2378201, 2.2033048, 2.176597, 2.1402047, 2.1119206, 2.0638845, 1.993593, 1.9360859, 1.8867111, 1.8016211, 1.7550396, 1.7313391, 1.6593345, 1.69958, 1.6353155, 1.5815704, 1.527669, 1.4967921, 1.4323189, 1.3679982, 1.3370432, 1.3548478, 1.2330565, 1.2144641, 1.1890142, 1.1389239, 1.1490458, 1.0713606, 1.0455558, 1.0139507, 0.9780038, 0.9860202, 0.9355294, 0.9172052, 0.89120543, 0.8664216, 0.84784824, 0.8237543, 0.8053271, 0.8347564, 0.76479304, 0.76079434, 0.79507935, 0.74129754, 0.7643122, 0.7256332, 0.75541997, 0.70544916, 0.7349442, 0.8238748, 0.69558585, 0.77246845, 0.8770758]
# = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

# Create a figure with two subplots (stacked vertically)
fig, axs = plt.subplots(2, 1, figsize=(8, 8))  # 2 rows, 1 column

# Plot the losses in the first subplot
axs[0].plot(losses, marker='o', linestyle='-', color='blue', label='Cross-Entropy loss')
axs[0].set_title('Cross-Entropy loss during epochs', fontsize=16)
axs[0].set_xlabel('Epoch', fontsize=12)
axs[0].set_ylabel('Cross-Entropy loss', fontsize=12)
axs[0].legend(fontsize=10)
axs[0].grid(True, linestyle='--', alpha=0.7)

# Plot the learning rates in the second subplot
axs[1].plot(learning_rates, marker='o', linestyle='-', color='red', label='Learning rate')
axs[1].set_title('Learning rate during epochs', fontsize=16)
axs[1].set_xlabel('Epoch', fontsize=12)
axs[1].set_ylabel('Learning Rate', fontsize=12)
axs[1].legend(fontsize=10)
axs[1].grid(True, linestyle='--', alpha=0.7)

# Adjust layout for better readability
plt.tight_layout()

# Display the plots
plt.show()

import matplotlib.pyplot as plt

# Your data
# need to paste the output array here ->
values = [2.5775754, 2.303266, 2.23492, 2.1820006, 2.1663268, 2.092114, 2.1279647, 2.1008713, 2.0565999, 1.9445829, 2.0102732, 1.935393, 1.9226673, 1.9138842, 1.8963734, 1.8167839, 1.8129399, 1.7086494, 1.6830381, 1.6897633, 1.6778904, 1.5378683, 1.6110089, 1.5714992, 1.5527709, 1.481962, 1.4106404, 1.4608526, 1.4082791, 1.367927, 1.342597, 1.313468, 1.3650645, 1.2927868, 1.2579275, 1.2198771, 1.1987131, 1.1702611, 1.1404196, 1.1149118, 1.1058905, 1.144952, 1.0491015, 1.0031599, 0.9975132, 0.98230535, 0.97021115, 0.937317, 0.92603266, 0.92634284]

# Create the plot
plt.figure(figsize=(10, 6))  # Set the figure size
plt.plot(values, marker='o', linestyle='-', color='blue', label='Values')  # Line plot with markers

# Add labels, title, and legend
plt.title('Plot of Given Values', fontsize=16)
plt.xlabel('Index', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(fontsize=10)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Display the plot
plt.show()

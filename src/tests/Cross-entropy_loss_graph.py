import matplotlib.pyplot as plt

# Your data
# need to paste the output array here ->
values = [2.254477, 2.089683, 1.9463774, 1.8286957, 1.8256452, 1.7021631, 1.7165549, 1.6172447, 1.5429804, 1.5158443, 1.4575006, 1.4329863, 1.4062792, 1.3411039, 1.3349245, 1.3440049, 1.308402, 1.2794291, 1.212161, 1.2384081, 1.1809322, 1.1894743, 1.0912066, 1.1002855, 1.0760548, 1.0471863, 1.0179743, 1.0142102, 0.9812814, 0.9336551, 0.9459568, 0.88264287, 0.8066495, 0.801771, 0.7989741, 0.7475809, 0.7182392, 0.68439007, 0.62305397, 0.62833047, 0.5847073, 0.5712376, 0.52885133, 0.5201405, 0.48312587, 0.4886573, 0.45512143, 0.44095284, 0.43013358, 0.41224274]

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

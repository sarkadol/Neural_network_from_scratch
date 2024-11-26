import matplotlib.pyplot as plt

# Your data
# need to paste the output array here ->
values = [2.3621786, 2.2302864, 2.2005231, 2.128506, 2.0230405, 1.8684168, 1.7182635, 1.6654751, 1.7110237, 2.0203538, 2.8114138, 4.948678, 7.7098675, 10.465697, 11.909109, 12.717258, 13.622315, 12.974894, 14.077723, 14.101967, 13.858874, 14.243974, 14.736539, 13.27026, 14.963444, 15.578411, 15.104953, 15.473366, 14.736539, 14.736539, 15.473366, 15.841779, 13.999713, 15.104953, 15.104953, 15.104953, 14.960836, 15.473366, 15.473366, 15.473366, 15.473366, 15.473366, 15.473366, 16.210192, 15.473366, 15.473366, 14.736539, 16.210192, 15.841779, 15.841779]

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

import matplotlib.pyplot as plt

# Your data
# need to paste the output array here ->
values = [0.1, 0.09954055, 0.098627955, 0.09727473, 0.09549926, 0.093325436, 0.090782054, 0.087902255, 0.08472275, 0.08128306, 0.07762472, 0.07379043, 0.06982325, 0.06576579, 0.061659504, 0.057543997, 0.05345644, 0.049431074, 0.04549881, 0.04168694, 0.038018942, 0.034514375, 0.031188896, 0.028054336, 0.025118863, 0.02238721, 0.019860948, 0.017538803, 0.015417003, 0.013489627, 0.011748974, 0.010185913, 0.008790224, 0.007550921, 0.006456541, 0.0054954076, 0.00465586, 0.0039264485, 0.0032960966, 0.0027542282, 0.0022908673, 0.0018967055, 0.0015631473, 0.0012823303, 0.0010471283, 8.5113785E-4, 6.8865216E-4, 5.546256E-4, 4.4463115E-4, 3.548133E-4, 2.818382E-4, 2.2284345E-4, 1.7538799E-4, 1.3740415E-4, 1.07151885E-4, 8.317635E-5, 6.4268745E-5, 4.9431048E-5, 3.7844242E-5, 2.8840303E-5]

# Create the plot
plt.figure(figsize=(10, 6))  # Set the figure size
plt.plot(values, marker='o', linestyle='-', color='blue', label='Values')  # Line plot with markers

# Add labels, title, and legend
#plt.title('Cross-entropy losses during epochs', fontsize=16)
plt.title('Learning rate losses during epochs', fontsize=16)

plt.xlabel('Epoch', fontsize=12)
#plt.ylabel('Cross entropy loss ', fontsize=12)
plt.ylabel('Learning rate ', fontsize=12)

plt.legend(fontsize=10)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Display the plot
plt.show()

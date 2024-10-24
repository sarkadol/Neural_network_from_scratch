import csv
import numpy as np
import matplotlib.pyplot as plt

# Path to your CSV file
csv_file_path = "../data/fashion_mnist_test_vectors.csv"
#csv_file_path = "../data/fashion_mnist_train_vectors.csv"

# Function to read a specific row (image) from the CSV
def display_image_from_csv(row_number):
    # Open the CSV file
    with open(csv_file_path, newline='') as file:
        csv_reader = csv.reader(file)

        # Skip to the specific row
        for idx, row in enumerate(csv_reader):
            if idx == row_number:
                # Convert the row to a numpy array and reshape it to 28x28
                image = np.array(row, dtype=np.uint8).reshape(28, 28)

                # Display the image using matplotlib
                plt.imshow(image, cmap='gray')  # Use a grayscale colormap
                plt.title(f"Image at row {row_number}")
                plt.show()
                break

# Example: Display the image from the first row (index 0)
display_image_from_csv(0)

display_image_from_csv(100)

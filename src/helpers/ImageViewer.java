package src.helpers;

import src.DataLoader;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;

/**
 * The {@code ImageViewer} class provides functionality for displaying
 * a single grayscale image from a dataset loaded by the {@code DataLoader}.
 */
public class ImageViewer extends JPanel {

    private BufferedImage image;

    /**
     * Constructor that creates a grayscale BufferedImage from pixel data.
     *
     * @param pixels Array of pixel values in the range [0, 1] (normalized).
     *               Should contain exactly 784 elements (28x28).
     */
    public ImageViewer(float[] pixels) {
        image = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                int pixelIndex = i * 28 + j;
                int grayValue = (int) (pixels[pixelIndex] * 255); // Scale to [0, 255]
                int rgb = new Color(grayValue, grayValue, grayValue).getRGB();
                image.setRGB(j, i, rgb);
            }
        }
    }

    /**
     * Paints the image onto the panel.
     */
    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        g.drawImage(image, 0, 0, this.getWidth(), this.getHeight(), null);
    }

    /**
     * Displays the image in a new window.
     *
     * @param title The title of the window.
     */
    public void display(String title) {
        JFrame frame = new JFrame(title);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(this);
        frame.setSize(300, 300); // Adjust the window size as needed
        frame.setVisible(true);
    }

    /**
     * Displays a grid of images in a new window.
     *
     * @param images List of float arrays representing normalized images.
     * @param labels List of labels corresponding to each image.
     * @param rows   Number of rows in the grid.
     * @param cols   Number of columns in the grid.
     */
    public static void displayImageGrid(List<float[]> images, List<Integer> labels, int rows, int cols) {
        JFrame frame = new JFrame("Image Grid");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // Set the layout of the frame to a grid
        frame.setLayout(new GridLayout(rows, cols, 5, 5)); // 5px gap between images

        // Add each image to the grid
        for (int i = 0; i < rows * cols && i < images.size(); i++) {
            float[] pixels = images.get(i);
            String label = DataLoader.getLabelName(labels.get(i));

            ImageViewer viewer = new ImageViewer(pixels);
            viewer.setPreferredSize(new Dimension(28, 28)); // Set the size of each image panel

            // Add label information at the bottom of each image
            JPanel panel = new JPanel(new BorderLayout());
            panel.add(viewer, BorderLayout.CENTER);
            panel.add(new JLabel(label, SwingConstants.CENTER), BorderLayout.SOUTH);

            frame.add(panel);
        }

        // Adjust frame size based on the number of images
        frame.pack();
        frame.setVisible(true);
    }

    /**
     * Loads the image data and labels, normalizes the images,
     * and displays a specified image in a separate window.
     *
     * @param args Command line arguments.
     */
    public static void main(String[] args) {
        try {
            String imagePath = "data/fashion_mnist_test_vectors.csv"; // Replace with actual path
            String labelPath = "data/fashion_mnist_test_labels.csv"; // Replace with actual path

            // Load images and labels using DataLoader
            // Load images and labels using DataLoader
            List<float[]> images = DataLoader.loadVectors(imagePath);
            List<Integer> labels = DataLoader.loadLabels(labelPath);

            // Normalize images
            images = DataLoader.normalizeVectors(images);

            // Display the images in a grid (e.g., 5x5 grid for 25 images)
            displayImageGrid(images, labels, 13, 20);

        } catch (IOException e) {
            System.out.println("Error loading images or labels: " + e.getMessage());
        }
    }
}

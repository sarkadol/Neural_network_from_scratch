package src;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * The {@code DataLoader} class provides utility methods for loading image data
 * and labels from files in CSV format and for printing images to the console.
 */
public class DataLoader {
    // Array of label names corresponding to each label number
    private static final String[] LABEL_NAMES = {
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };

    /**
     * Loads 28x28 pixel images from a CSV file where each line contains 784
     * comma-separated integers representing an image.
     *
     * @param filePath Path to the image data file.
     * @return List of int arrays, each array representing an image.
     * @throws IOException if an error occurs during file reading.
     */
    public static List<int[]> loadVectors(String filePath) throws IOException {
        List<int[]> images = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] pixelStrings = line.split(",");
            int[] pixels = new int[784];
            for (int i = 0; i < pixelStrings.length; i++) {
                pixels[i] = Integer.parseInt(pixelStrings[i]);
            }
            images.add(pixels);
        }
        reader.close();
        return images;
    }

    /**
     * Loads labels from a file where each line contains a single integer label.
     *
     * @param filePath Path to the label data file.
     * @return List of integers representing labels.
     * @throws IOException if an error occurs during file reading.
     */
    public static List<Integer> loadLabels(String filePath) throws IOException {
        List<Integer> labels = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        String line;
        while ((line = reader.readLine()) != null) {
            labels.add(Integer.parseInt(line));
        }
        reader.close();
        return labels;
    }

    /**
     * Prints a 28x28 pixel image from an array of integers.
     *
     * @param pixels Array of 784 integers representing an image.
     */
    public static void printData(int[] pixels) {
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                System.out.printf("%3d ", pixels[i * 28 + j]);
            }
            System.out.println();
        }
    }
    /**
     * Prints a 28x28 normalized image from an array of floats, each representing a pixel
     * value in the range [0, 1].
     *
     * @param pixels Array of 784 floats representing a normalized image.
     */
    public static void printNormalizedData(float[] pixels) {
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                System.out.printf("%4.2f ", pixels[i * 28 + j]);
            }
            System.out.println();
        }
    }
    /**
     * Normalizes a list of images by scaling pixel values to the range [0, 1].
     * Each pixel value is divided by 255.
     *
     * @param images List of int arrays, where each array represents an image.
     * @return List of float arrays, where each array represents a normalized image.
     */
    public static List<float[]> normalizeVectors(List<int[]> images) {
        List<float[]> normalizedImages = new ArrayList<>();

        for (int[] image : images) {
            float[] normalizedImage = new float[784];
            for (int i = 0; i < image.length; i++) {
                normalizedImage[i] = image[i] / 255.0f; // Normalize by dividing by 255
            }
            normalizedImages.add(normalizedImage);
        }

        return normalizedImages;
    }

    public static String getLabelName(int label) {
        if (label >= 0 && label < LABEL_NAMES.length) {
            return LABEL_NAMES[label];
        } else {
            return "Unknown";
        }
    }

}

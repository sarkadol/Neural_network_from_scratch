package src;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.stream.Collectors;

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
    public static List<float[]> loadVectors(String filePath) throws IOException {
        List<float[]> images = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] pixelStrings = line.split(",");
            float[] pixels = new float[784];
            for (int i = 0; i < pixelStrings.length; i++) {
                pixels[i] = Integer.parseInt(pixelStrings[i]);
            }
            images.add(pixels);
        }
        reader.close();
        return images;
    }


    /**
     * Writes and array to given csv file.
     * @param array
     * @param filePath
     */
    public static void writeArrayToCSV(int[] array, String filePath) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            // Iterate through the array and write each element to a new line
            for (int value : array) {
                writer.write(Integer.toString(value)); // Convert the integer to a string
                writer.newLine(); // Add a newline character
            }
            System.out.println("Results written to " + filePath);
        } catch (IOException e) {
            System.err.println("Error writing array to CSV: " + e.getMessage());
        }
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
    public static void printData(float[] pixels) {
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
    public static List<float[]> normalizeVectors(List<float[]> images) {
        List<float[]> normalizedImages = new ArrayList<>();

        for (float[] image : images) {
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


    /**
     * Loads 28x28 pixel images from a CSV file where each line contains 784
     * comma-separated integers representing an image, and normalizes it.
     *
     * @param filePath Path to the image data file.
     * @return List of int arrays, each array representing a normalized image.
     * @throws IOException if an error occurs during file reading.
     */
    public static List<float[]> loadAndNormalizeVectors(String filePath) throws IOException {
        List<float[]> images = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] pixelStrings = line.split(",");
            float[] pixels = new float[784];
            for (int i = 0; i < pixelStrings.length; i++) {
                //pixels[i] = Integer.parseInt(pixelStrings[i]);
                pixels[i] = Integer.parseInt(pixelStrings[i]) / 255.0f;
            }
            images.add(pixels);
        }
        reader.close();
        return images;
    }

    public static void writeToFileForPython(float[] losses, float[] learningRates, int train_vector_count, String layers, Hyperparameters hp) {
        try (FileWriter writer = new FileWriter("Java_to_Python.txt")) {
            writer.write("losses = " + Arrays.toString(losses) + "\n");
            writer.write("learning_rates = " + Arrays.toString(learningRates) + "\n");
            writer.write("train_vector_count = " + train_vector_count + "\n");
            writer.write("layers = " + layers + "\n");
            writer.write("batch_size = " + hp.getBatchSize() + "\n");
            writer.write("decay_rate = " + hp.getWeightDecay() + "\n");
            writer.write("momentum = " + hp.getMomentum() + "\n");
            writer.write("learning_decay_rate = " + hp.getLearningDecayRate() + "\n");
            writer.write("clip_value = " + hp.getClipValue() + "\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
        //System.out.println("Data written to file for Python graph evaluation.");
    }
    public static void writeToCsvForComparison(int train_vector_count, String layers, Hyperparameters hp,long totalTime) {

        try (FileWriter writer = new FileWriter("all_tries.csv",true)) {
            // Prepare the data in a single row with semicolons as delimiters
            String dataRow = String.join(";",
                    Integer.toString(train_vector_count),
                    layers,
                    Integer.toString(hp.getBatchSize()),
                    Float.toString(hp.getWeightDecay()),
                    Float.toString(hp.getMomentum()),
                    Float.toString(hp.getLearningDecayRate()),
                    Float.toString(hp.getClipValue()),
                    Float.toString(evaluate("NEW_test_predictions.csv","data/fashion_mnist_test_labels.csv")),
                    Long.toString(totalTime),
                    Integer.toString(hp.getEpochs())
            );

            writer.write(dataRow + "\n"); // Write the data to the file
        } catch (IOException e) {
            e.printStackTrace();
        }
        //System.out.println("Data written to CSV file for comparison.");
    }

    // Evaluate accuracy
    private static float evaluate(String predictionsPath, String truthPath) throws IOException {
        List<String> predictions = readFileAsList(predictionsPath);
        List<String> truth = readFileAsList(truthPath);

        if (predictions.size() != truth.size()) {
            throw new IllegalArgumentException("Predictions and truth sizes do not match!");
        }

        int hits = 0;
        for (int i = 0; i < predictions.size(); i++) {
            if (predictions.get(i).equals(truth.get(i))) {
                hits++;
            }
        }

        return (float) hits / predictions.size();
    }
    // Read file as a list of strings
    private static List<String> readFileAsList(String filePath) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            return br.lines().collect(Collectors.toList());
        }
    }


}

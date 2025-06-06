package src.helpers;

import src.Hyperparameters;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Just for us for debugging purposes and tuning of parameters
 */
public class Helper {

    public static void writeToFileForPython(float[] losses, float[] learningRates, int train_vector_count, String layers, Hyperparameters hp, LocalDateTime time) {
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
            writer.write("time = " + time + "\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
        //System.out.println("Data written to file for Python graph evaluation.");
    }
    public static void writeToCsvForComparison(int train_vector_count, String layers, Hyperparameters hp,long totalTime,LocalDateTime time) {

        try (FileWriter writer = new FileWriter("all_tries_new.csv",true)) {
            // Prepare the data in a single row with semicolons as delimiters
            String dataRow = String.join(";",
                    Integer.toString(train_vector_count),
                    layers,
                    Float.toString(evaluate("NEW_test_predictions.csv","data/fashion_mnist_test_labels.csv")),
                    Float.toString(hp.getLearningRate()),
                    Integer.toString(hp.getBatchSize()),
                    Boolean.toString(hp.useLearningDecayRate()),
                    Float.toString(hp.getLearningDecayRate()),
                    Float.toString(hp.getClipValue()),
                    Float.toString(hp.getMomentum()),
                    Float.toString(hp.getWeightDecay()),
                    Long.toString(totalTime),
                    Integer.toString(hp.getEpochs()),
                    time.toString()
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

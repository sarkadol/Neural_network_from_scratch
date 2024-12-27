package src;

import src.helpers.Helper;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class Main {
    /**
     * This main method creates a neural network, trains it and then predicts the labels
     * according to PV021 Neural Networks project.
     * @param args
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        long total_start_time = System.currentTimeMillis();
        System.out.println("Initializing layers...");
        Layer layer0 = new Layer(28*28);
        Layer layer1 = new Layer(layer0, 256, "relu");
        Layer layer2 = new Layer(layer1, 128, "relu");
        Layer layer3 = new Layer(layer2, 10, "softmax");
        Layer[] layers = new Layer[] {layer0, layer1, layer2, layer3};
        System.out.println("Layers initialized");

        Network network = new Network(layers);

        int number_of_images = 100; //max 60000
        System.out.println("Loading and normalizing a subset of data...");
        //training set of 60,000 examples
        //test set of 10,000 examples
        List<float[]> train_vectors = DataLoader.loadAndNormalizeVectors("data/fashion_mnist_train_vectors.csv").subList(0, number_of_images);
        List<Integer> train_labels = DataLoader.loadLabels("data/fashion_mnist_train_labels.csv").subList(0, number_of_images);
        System.out.println("Loading completed");

        Hyperparameters hyperparameters = new Hyperparameters(
                10,
                0.001f,
                20,
                false,
                100,
                5.0f,
                0.8F,
                0.01F,
                true);

        long startTime = System.currentTimeMillis();

        network.trainNetwork(train_vectors, train_labels, hyperparameters, false);

        long endTime = System.currentTimeMillis();
        long elapsedMillis = endTime - startTime;
        long minutes = (elapsedMillis / 1000) / 60;
        long seconds = (elapsedMillis / 1000) % 60;
        long millis = elapsedMillis % 1000;

        System.out.println(String.format("Training time: %02d:%02d:%03d", minutes, seconds, millis));

        if (false) {        //PREDICTING TRAIN DATASET
            System.out.println("Predicting train dataset of 60,000 images...");
            List<float[]> test_vectors = DataLoader.loadAndNormalizeVectors("data/fashion_mnist_train_vectors.csv");
            long startTimePrediction = System.currentTimeMillis();
            int[] predicted_labels = network.predictAll(test_vectors);
            long endTimePrediction = System.currentTimeMillis();

            long totalTime = endTime - startTime+endTimePrediction - startTimePrediction;

            DataLoader.writeArrayToCSV(predicted_labels,"train_predictions.csv"); //for evaluation (see README)
            Helper.writeToCsvForComparison(number_of_images, Arrays.toString(network.getLayersLength()), hyperparameters,totalTime);

            System.out.println("Predicting train dataset in "+(endTimePrediction - startTimePrediction) + " milliseconds");
        }
        if (true) {        //PREDICTING TEST DATASET
            System.out.println("Predicting test dataset of 10,000 images...");
            List<float[]> test_vectors = DataLoader.loadAndNormalizeVectors("data/fashion_mnist_test_vectors.csv");
            long startTimePrediction = System.currentTimeMillis();
            int[] predicted_labels = network.predictAll(test_vectors);
            long endTimePrediction = System.currentTimeMillis();

            long totalTime = endTime - startTime+endTimePrediction - startTimePrediction;

            DataLoader.writeArrayToCSV(predicted_labels,"NEW_test_predictions.csv"); //for us
            DataLoader.writeArrayToCSV(predicted_labels,"test_predictions.csv"); //for evaluation (see README)
            Helper.writeToCsvForComparison(number_of_images, Arrays.toString(network.getLayersLength()), hyperparameters,totalTime);

            System.out.println("Predicting test dataset in "+(endTimePrediction - startTimePrediction) + " milliseconds");
        }

        long total_end_time = System.currentTimeMillis();
        long total_time = total_end_time-total_start_time;
        // Convert total_time to minutes, seconds, and milliseconds
        minutes = (total_time / 1000) / 60;
        seconds = (total_time / 1000) % 60;
        millis = total_time % 1000;

        // Format and print the result
        System.out.printf("\nProject completed in %02d:%02d:%03d\n", minutes, seconds, millis);
        System.out.println("Šárka Blaško, 567774");
        System.out.println("Kryštof Zamazal, 514304");

    }
}

package src;

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
        System.out.println("Initializing layers...");
        Layer layer0 = new Layer(28*28);
        Layer layer1 = new Layer(layer0, 128, "relu");
        Layer layer2 = new Layer(layer1, 16, "relu");
        Layer layer3 = new Layer(layer2, 10, "softmax");
        Layer[] layers = new Layer[] {layer0, layer1, layer2, layer3};
        System.out.println("Layers initialized");

        Network network = new Network(layers);

        if (true) {
            int number_of_images = 10; //max 60000
            System.out.println("Loading and normalizing a subset of data...");
            //training set of 60,000 examples
            //test set of 10,000 examples
            List<float[]> train_vectors = DataLoader.loadAndNormalizeVectors("data/fashion_mnist_train_vectors.csv").subList(0, number_of_images);
            List<Integer> train_labels = DataLoader.loadLabels("data/fashion_mnist_train_labels.csv").subList(0, number_of_images);
            System.out.println("Loading completed");

            Hyperparameters hyperparameters = new Hyperparameters(
                    20,
                    0.01f,
                    50,
                    5.0f,
                    64,
                    0.5F,
                    0.0005F);
            //momentum 0 = momentum not used

            long startTime = System.currentTimeMillis();
            network.trainNetwork(train_vectors, train_labels, hyperparameters, false);
            long endTime = System.currentTimeMillis();

            if (true) {        //PREDICTING
                System.out.println("Predicting...");
                List<float[]> test_vectors = DataLoader.loadAndNormalizeVectors("data/fashion_mnist_test_vectors.csv");
                long startTimePrediction = System.currentTimeMillis();
                int[] predicted_labels = network.predictAll(test_vectors);
                long endTimePrediction = System.currentTimeMillis();

                //System.out.println("Predicted labels:\n"+Arrays.toString(predicted_labels));
                DataLoader.writeArrayToCSV(predicted_labels,"NEW_test_predictions.csv");

                System.out.println("Predicting "+(endTimePrediction - startTimePrediction) + " milliseconds");
            }
            System.out.println("Training: "+ (endTime - startTime) + " milliseconds");
        }
    }

    private static void evaluateNetwork(Network network, List<float[]> testVectors, List<Integer> testLabels) {
        int correct = 0;
        for (int i = 0; i < testVectors.size(); i++) {
            float[] outputs = network.forwardPass(testVectors.get(i));
            //if (Util.argmax(outputs) == testLabels.get(i)) {
             //   correct++;
            //}
        }
        System.out.println("Accuracy: " + (correct / (float) testVectors.size()) * 100 + "%");
    }

}

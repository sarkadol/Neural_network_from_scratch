package src;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static java.lang.Float.POSITIVE_INFINITY;

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
        Layer layer2 = new Layer(layer1, 128, "relu");
        Layer layer3 = new Layer(layer2, 128, "relu");
        Layer layer4 = new Layer(layer3, 128, "relu");
        Layer layer5 = new Layer(layer4, 10, "softmax");
        Layer[] layers = new Layer[] {layer0, layer1, layer2, layer3, layer4, layer5};
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
                    10,
                    0.01f,
                    64,
                    true,
                    500,
                    5.0f,
                    0.8F,
                    0.01F);
            //momentum 0 = momentum not used
            //weight decay rate 0 = not used

            long startTime = System.currentTimeMillis();
            network.trainNetwork(train_vectors, train_labels, hyperparameters, false);
            long endTime = System.currentTimeMillis();

            if (true) {        //PREDICTING TRAIN DATASET
                System.out.println("Predicting train dataset of 60,000 images...");
                List<float[]> test_vectors = DataLoader.loadAndNormalizeVectors("data/fashion_mnist_train_vectors.csv");
                long startTimePrediction = System.currentTimeMillis();
                int[] predicted_labels = network.predictAll(test_vectors);
                long endTimePrediction = System.currentTimeMillis();

                long totalTime = endTime - startTime+endTimePrediction - startTimePrediction;

                DataLoader.writeArrayToCSV(predicted_labels,"train_predictions.csv"); //for evaluation (see README)
                DataLoader.writeToCsvForComparison(number_of_images, Arrays.toString(network.getLayersLength()), hyperparameters,totalTime);

                System.out.println("Predicting train dataset in "+(endTimePrediction - startTimePrediction) + " milliseconds");
            }
            if (true) {        //PREDICTING TEST DATASET
                System.out.println("Predicting test dataset of 10,000 images...");
                List<float[]> test_vectors = DataLoader.loadAndNormalizeVectors("data/fashion_mnist_test_vectors.csv");
                long startTimePrediction = System.currentTimeMillis();
                int[] predicted_labels = network.predictAll(test_vectors);
                long endTimePrediction = System.currentTimeMillis();

                long totalTime = endTime - startTime+endTimePrediction - startTimePrediction;

                //System.out.println("Predicted labels:\n"+Arrays.toString(predicted_labels));
                DataLoader.writeArrayToCSV(predicted_labels,"NEW_test_predictions.csv"); //for us
                DataLoader.writeArrayToCSV(predicted_labels,"test_predictions.csv"); //for evaluation (see README)
                DataLoader.writeToCsvForComparison(number_of_images, Arrays.toString(network.getLayersLength()), hyperparameters,totalTime);

                System.out.println("Predicting test in "+(endTimePrediction - startTimePrediction) + " milliseconds");
            }
            System.out.println("\nTraining: "+ (endTime - startTime) + " milliseconds");
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

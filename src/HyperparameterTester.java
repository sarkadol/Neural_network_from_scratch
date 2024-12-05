package src;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class HyperparameterTester {

    public static void main(String[] args) throws IOException {
        System.out.println("Initializing layers...");
        Layer layer0 = new Layer(28 * 28);
        Layer layer1 = new Layer(layer0, 128, "relu");
        Layer layer2 = new Layer(layer1, 16, "relu");
        Layer layer3 = new Layer(layer2, 10, "softmax");
        Layer[] layers = new Layer[]{layer0, layer1, layer2, layer3};
        System.out.println("Layers initialized");

        Network network = new Network(layers);

        int number_of_images = 1000; // max 60000
        System.out.println("Loading and normalizing a subset of data...");
        List<float[]> train_vectors = DataLoader.loadAndNormalizeVectors("data/fashion_mnist_train_vectors.csv").subList(0, number_of_images);
        List<Integer> train_labels = DataLoader.loadLabels("data/fashion_mnist_train_labels.csv").subList(0, number_of_images);
        List<float[]> test_vectors = DataLoader.loadAndNormalizeVectors("data/fashion_mnist_test_vectors.csv");
        List<Integer> test_labels = DataLoader.loadLabels("data/fashion_mnist_test_labels.csv");
        System.out.println("Loading completed");

        HyperparameterTester tester = new HyperparameterTester(network, train_vectors, train_labels, test_vectors, test_labels, layers);
        tester.runTests(number_of_images);
    }

    private final Network network;
    private final List<float[]> trainVectors;
    private final List<Integer> trainLabels;
    private final List<float[]> testVectors;
    private final List<Integer> testLabels;
    private final Layer[] layers;

    public HyperparameterTester(Network network, List<float[]> trainVectors, List<Integer> trainLabels, List<float[]> testVectors, List<Integer> testLabels, Layer[] layers) {
        this.network = network;
        this.trainVectors = trainVectors;
        this.trainLabels = trainLabels;
        this.testVectors = testVectors;
        this.testLabels = testLabels;
        this.layers = layers;
    }

    public void runTests(int number_of_images) {
        // Define ranges for hyperparameters
        int[] batchSizes = {32, 64, 128};
        float[] learningRates = {0.01f,0.1f};
        float[] momentums = {0.0f, 0.8f, 0.9f};
        float[] weightDecays = {0.01f, 0.0f};
        float[] learningDecayRates = {500, 1000};
        float[] clipValues = {5f, 1.0f};

        // Automate testing with a grid search
        for (int batchSize : batchSizes) {
            for (float learningRate : learningRates) {
                for (float momentum : momentums) {
                    for (float weightDecay : weightDecays) {
                        for (float learningDecayRate : learningDecayRates) {
                            for (float clipValue : clipValues) {
                                for (int i=0; i<5; i++) {
                                    Hyperparameters hp = new Hyperparameters(
                                            60,
                                            learningRate,
                                            batchSize,
                                            true,
                                            learningDecayRate,
                                            clipValue,
                                            momentum,
                                            weightDecay
                                    );
                                    //System.out.println("Testing hyperparameters: " + hp);
                                    try {
                                        long startTime = System.currentTimeMillis();
                                        network.trainNetwork(trainVectors, trainLabels, hp, false);
                                        long endTime = System.currentTimeMillis();
                                        System.out.println("Predicting...");
                                        List<float[]> test_vectors = DataLoader.loadAndNormalizeVectors("data/fashion_mnist_test_vectors.csv");
                                        long startTimePrediction = System.currentTimeMillis();
                                        int[] predicted_labels = network.predictAll(test_vectors);
                                        long endTimePrediction = System.currentTimeMillis();

                                        //System.out.println("Predicted labels:\n"+Arrays.toString(predicted_labels));
                                        DataLoader.writeArrayToCSV(predicted_labels,"NEW_test_predictions.csv");
                                        DataLoader.writeToCsvForComparison(number_of_images, Arrays.toString(network.getLayersLength()), hp);

                                        System.out.println("Predicting "+(endTimePrediction - startTimePrediction) + " milliseconds");
                                        System.out.println("Training: "+ (endTime - startTime) + " milliseconds");



                                    } catch (Exception e) {
                                        System.err.println("Error testing hyperparameters: " + hp);
                                        e.printStackTrace();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private float evaluateNetworkAccuracy(int[] predictedLabels, List<Integer> trueLabels) {
        int correct = 0;
        for (int i = 0; i < predictedLabels.length; i++) {
            if (predictedLabels[i] == trueLabels.get(i)) {
                correct++;
            }
        }
        return (correct / (float) predictedLabels.length) * 100;
    }
}

package src;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class Main {

    public static void main(String[] args) throws IOException {
        try {

            if (false) { //just for debugging purposes
                System.out.println("loading the images...");

                // load the data
                List<float[]> test_vectors = DataLoader.loadAndNormalizeVectors("data/fashion_mnist_test_vectors.csv");
                List<Integer> test_labels = DataLoader.loadLabels("data/fashion_mnist_test_labels.csv");
                List<float[]> train_vectors = DataLoader.loadAndNormalizeVectors("data/fashion_mnist_train_vectors.csv");
                List<Integer> train_labels = DataLoader.loadLabels("data/fashion_mnist_train_labels.csv");
                System.out.println("images loaded...");

                // Display the X-th image and its label
                int image = 11;
                DataLoader.printData(test_vectors.get(image));
                //label
                System.out.println(test_labels.get(image)+": "+DataLoader.getLabelName(test_labels.get(image)));
                //DataLoader.printNormalizedData(normalized_test_vectors.get(image));

                Neuron neuron1 = new Neuron();
                neuron1.printInfo();
                neuron1.printInfoLine();
            }
        } catch (Exception e) {
            System.out.println("An error occurred: " + e.getMessage());
        }
        //creating the layers
        System.out.println("Initializing layers...");
        Layer layer0 = new Layer(28*28);
        Layer layer1 = new Layer(layer0, 16, "relu");
        Layer layer2 = new Layer(layer1, 8, "relu");
        Layer layer3 = new Layer(layer2, 10, "softmax");
        Layer[] layers = new Layer[] {layer0, layer1, layer2, layer3};
        System.out.println("Layers initialized");

        Network network = new Network(layers);

        List<float[]> test_vectors = DataLoader.loadAndNormalizeVectors("data/fashion_mnist_test_vectors.csv");
        int[] predicted_labels = network.predictAll(test_vectors);
        System.out.println("\nPredicted labels:\n"+Arrays.toString(predicted_labels));
        DataLoader.writeArrayToCSV(predicted_labels,"NEW_test_predictions.csv");

        if (false) {
            System.out.println("Loading data...");
            float[] image = DataLoader.loadVectors("data/fashion_mnist_train_vectors.csv").get(0);
            float[] output = network.forwardPass(image);
            float[] target = Util.labelToVector(0);
            System.out.println("Desired predictions: " + Arrays.toString(target) );
            System.out.println("Predictions: " + Arrays.toString(output));
            System.out.println("Predictions Length: " + output.length);
            float sum = 0;
            for (float single_output: output) {
                sum += single_output;
            }
            System.out.println("Predictions sum: " + sum);
            network.train(0.01F,target,output);
        }

        if (false) {
            int number_of_images = 50;
            System.out.println("Loading and normalizing a subset of data...");
            List<float[]> trainVectors = DataLoader.loadAndNormalizeVectors("data/fashion_mnist_train_vectors.csv").subList(0, number_of_images);
            List<Integer> trainLabels = DataLoader.loadLabels("data/fashion_mnist_train_labels.csv").subList(0, number_of_images);

            //System.out.println("Training on 5 images for debugging...");
            network.trainNetwork(trainVectors, trainLabels, 50, 0.01f, false);

            //System.out.println("Label of image: "+network.predict(DataLoader.loadVectors("data/fashion_mnist_train_vectors.csv").get(0)));



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

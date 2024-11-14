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
                List<float[]> test_vectors = DataLoader.loadVectors("data/fashion_mnist_test_vectors.csv");
                List<Integer> test_labels = DataLoader.loadLabels("data/fashion_mnist_test_labels.csv");
                //List<int[]> train_vectors = DataLoader.loadVectors("data/fashion_mnist_train_vectors.csv");
                //List<Integer> train_labels = DataLoader.loadLabels("data/fashion_mnist_train_labels.csv");

                List<float[]> normalized_test_vectors = DataLoader.normalizeVectors(test_vectors);
                //List<float[]> normalized_train_vectors = DataLoader.normalizeVectors(train_vectors);
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
        Layer layer0 = new Layer(28*28);
        Layer layer1 = new Layer(layer0, 16, "relu");
        Layer layer2 = new Layer(layer1, 8, "relu");
        Layer layer3 = new Layer(layer2, 10, "softmax");
        Layer[] layers = new Layer[] {layer0, layer1, layer2, layer3};
        System.out.println("Layers initialized");

        Network network = new Network(layers);
        float[] output = network.ForwardPass(DataLoader.loadVectors("data/fashion_mnist_train_vectors.csv").get(0));
        System.out.println("List: " + Arrays.toString(output));
        System.out.println("List Length: " + output.length);
        float sum = 0;
        for (float single_output: output) {
            sum += single_output;
        }
        System.out.println("List sum: " + sum);

        layer0.printInfo(false);
        layer1.printInfo(false);
        layer2.printInfo(false);
        layer3.printInfo(false);
    }
}

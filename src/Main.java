package src;
import java.util.List;

public class Main {

    public static void main(String[] args) {
        try {

            if (false) { //just for debugging purposes
                System.out.println("loading the images...");

                // load the data
                List<int[]> test_vectors = DataLoader.loadVectors("data/fashion_mnist_test_vectors.csv");
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
        layer0.printInfo(false);
        Layer layer1 = new Layer(layer0, 16, "relu");
        layer1.printInfo(false);
        Layer layer2 = new Layer(layer1, 8, "relu");
        layer2.printInfo(false);
        Layer layer3 = new Layer(layer2, 24, "softmax");
        layer3.printInfo(false);
        System.out.println("layers initialized");
        layer1.InitializeWeights();
        layer2.InitializeWeights();

        layer1.printInfo(true);
        layer2.printInfo(true);



    }
}

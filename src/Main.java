package src;
import java.util.List;

public class Main {

    public static void main(String[] args) {
        try {
            // load the data
            List<int[]> test_vectors = DataLoader.loadVectors("data/fashion_mnist_test_vectors.csv");
            List<Integer> test_labels = DataLoader.loadLabels("data/fashion_mnist_test_labels.csv");
            List<int[]> train_vectors = DataLoader.loadVectors("data/fashion_mnist_train_vectors.csv");
            List<Integer> train_labels = DataLoader.loadLabels("data/fashion_mnist_train_labels.csv");

            List<float[]> normalized_test_vectors = DataLoader.normalizeVectors(test_vectors);
            List<float[]> normalized_train_vectors = DataLoader.normalizeVectors(train_vectors);

            // Display the X-th image and its label
            int image = 10;
            DataLoader.printData(test_vectors.get(image));
            //label
            System.out.println(test_labels.get(image)+": "+DataLoader.getLabelName(test_labels.get(image)));
            //DataLoader.printNormalizedData(normalized_test_vectors.get(image));

        } catch (Exception e) {
            System.out.println("An error occurred: " + e.getMessage());
        }
    }
}

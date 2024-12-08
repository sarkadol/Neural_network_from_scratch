package src;

import java.util.List;
import java.util.Collections;
import java.util.ArrayList;
import java.util.stream.Collectors;

/**
 * Class containing two lists of vectors and labels representing images
 * from Fashion MNIST dataset
 */
public class Dataset {
    List<float[]> vectors;
    List<Integer> labels;

    public Dataset(List<float[]> vectors, List<Integer> labels) {
        this.vectors = vectors;
        this.labels = labels;
    }
    public List<float[]> getVectors() {
        return vectors;
    }
    public List<Integer> getLabels() {
        return labels;
    }

    /**
     * Randomly shuffles the dataset, ensuring vectors and labels remain synchronized.
     * Useful for removing order bias before training.
     */
    public void shuffle() {
        // Create a list of indices
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < vectors.size(); i++) {
            indices.add(i);
        }

        // Shuffle the indices
        Collections.shuffle(indices);

        // Reorder the vectors and labels based on the shuffled indices
        List<float[]> shuffledVectors = indices.stream()
                                               .map(vectors::get)
                                               .collect(Collectors.toList());
        List<Integer> shuffledLabels = indices.stream()
                                              .map(labels::get)
                                              .collect(Collectors.toList());
        // Update the attributes
        this.vectors = shuffledVectors;
        this.labels = shuffledLabels;
    }
}

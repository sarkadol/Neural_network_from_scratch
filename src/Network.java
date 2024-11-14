package src;

public class Network {
    public Layer[] layers;

    /**
     * Creates a network with layers
     * @param layers
     */
    public Network(Layer[] layers) {

    }

    /**
     * Performs forward propagation
     * Calculates the output of the network by sequentially passing data through each layer.
     *
     * @param inputs float array representing the input features for a single image
     * @return array representing the network's output (class probabilities)
     */
    public float[] ForwardPass(float[] inputs){
        for (int i = 0; i < layers.length; i++){
            inputs = layers[i].computeOutput(inputs);
        }
        return inputs;
    }
    public void BackPropagation(float learning_rate, float[] training_data){

    }
}

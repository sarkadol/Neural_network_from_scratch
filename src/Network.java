package src;

import java.util.Arrays;

import static src.Util.activationDerivative;
import static src.Util.crossEntropy;

public class Network {
    public Layer[] layers;

    /**
     * Creates a network with layers
     * @param layers Layers of the network
     */
    public Network(Layer[] layers) {
        this.layers = layers;
        this.initializeWeights();
    }

    /**
     * Performs forward propagation
     * Calculates the output of the network by sequentially passing data through each layer.
     *
     * @param inputs float array representing the input features for a single image
     * @return array representing the network's output (class probabilities)
     */
    public float[] ForwardPass(float[] inputs){
        System.out.println("Forward pass proceeding...");
        for (int i = 1; i < layers.length; i++){    // V tuto chvíli je nepotřebná vstupní vrstva
            inputs = layers[i].computeOutput(inputs);
        }
        System.out.println("Forward pass complete.");
        return inputs;
    }

    private float[] computeOutputLayerGradient(float[] predicted, float[] target) {
        float[] gradient = new float[predicted.length];
        for (int i = 0; i < predicted.length; i++) {
            gradient[i] = predicted[i] - target[i];
        }
        return gradient;
    }

    private float[] backpropagateHiddenLayer(float[] nextLayerGradient, Layer currentLayer, Layer previousLayer) {
        int currentNeuronCount = currentLayer.neurons.length;
        int previousNeuronCount = previousLayer.neurons.length;

        float[] currentGradient = new float[previousNeuronCount]; // Gradient for previous layer

        for (int i = 0; i < previousNeuronCount; i++) {
            currentGradient[i] = 0; // Initialize gradient
            for (int j = 0; j < currentNeuronCount; j++) {
                // Backpropagate gradient from current layer
                float weightGradient = nextLayerGradient[j] * currentLayer.neurons[j].weights[i];
                currentGradient[i] += weightGradient;
            }
            // Multiply by activation function derivative
            currentGradient[i] *= activationDerivative(previousLayer.neurons[i].getInnerPotential(), previousLayer.activation_function);
        }
        return currentGradient;
    }



    /**
     *
     * @param learning_rate
     * @param target list of desired probabilities given by label
     * @param outputs list of computed probabilities from forward pass
     */
    public void BackPropagation(float learning_rate, float[] target, float[] outputs){
        System.out.println("\nBack Propagation");

        // Step 2: Compute the gradient of the loss function at the output layer - see the improvement during training
        float loss = Util.crossEntropy(target, outputs);
        System.out.println("cross entropy: "+loss);

        // Step 3: Compute gradients for the output layer using softmax + cross-entropy derivative
        float[] outputLayerGradient = computeOutputLayerGradient(target, outputs);
        System.out.println("output layer gradients: "+ Arrays.toString(outputLayerGradient));

        // Step 4: Backward pass through hidden layers
        float[] currentGradient = outputLayerGradient;
        for (int i = layers.length - 1; i > 0; i--) {
            Layer currentLayer = layers[i];
            Layer previousLayer = layers[i - 1];
            System.out.println("Passing from " + i + " to " + (i-1));
            // Update weights and biases for the current layer
            //System.out.println("Gradients size: " + currentGradient.length);
            //System.out.println("Number of neurons: " + previousLayer.y.length);

            //System.out.println("Gradients size: " + currentGradient.length);
            //System.out.println("Previous layer output size: " + previousLayer.y.length);
            //System.out.println("Neurons in current layer: " + currentLayer.neurons.length);
            //System.out.println("Weights per neuron in current layer: " + currentLayer.neurons[0].getWeights().length);
            currentLayer.updateWeights(currentGradient, previousLayer.y, learning_rate);
            //currentGradient = backpropagateHiddenLayer(currentGradient, currentLayer, previousLayer);
            if (i > 1) { // Stop backpropagation before reaching the input layer
                currentGradient = backpropagateHiddenLayer(currentGradient, currentLayer, previousLayer);
            }
        }
        System.out.println("Backpropagation competed.");
    }

    /**
     * Initializes weights and biases for all layers in the network, excluding the input layer.
     */
    public void initializeWeights(){
        for (int i = 1; i < layers.length; i++){//skip 0th because it has no weights nor bias
            layers[i].InitializeWeights();
        }
        System.out.println("Initialization complete");
    }

    public void printInfo() {
        System.out.println("---Network - number of layers: " + layers.length);
        for (Layer layer : layers){
            layer.printInfoLine();
        }

    }
}

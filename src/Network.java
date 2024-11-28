package src;

import static src.Util.activationDerivative;

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
        //System.out.println("Inputs before forward pass:" + Arrays.toString(inputs));
        for (int i = 1; i < layers.length; i++){    // V tuto chvíli je nepotřebná vstupní vrstva
            layers[i].setX(inputs); // Save respective attributes
            inputs = layers[i].computeOutput(inputs);
        }
        //System.out.println("Inputs after forward pass:" + Arrays.toString(inputs));

        System.out.println("Forward pass complete.");
        return inputs;
    }

    /**
     * Computes gradients of the output layer. (How should the y of output layer change)
     * @param predicted array of predicted probabilities
     * @param target array of desired probabilities
     * @return array of gradients
     */
    private float[] computeOutputLayerGradients(float[] predicted, float[] target) {
        float[] gradients = new float[predicted.length];
        for (int i = 0; i < predicted.length; i++) {
            gradients[i] = predicted[i] - target[i];    // TODO check if it is truly true; it is the same as partial
                                                        // derivative wrt. outputs and we are computing partial
                                                        // derivative wrt. inner potentials
        }
        return gradients;
    }

    /**
     * Backpropagation of a hidden layer:
     * (∂E_k / ∂y_j) = Σ_(r ∈ j→) [(∂E_k / ∂y_r) * σ'_r(ξ_r) * w_rj]
     *
     * r ∈ j→ indicates that r is in the set of nodes connected to j. (arch from j to r)
     * σ'_r(ξ_r) is the derivative of the activation function with respect to ξ_r.
     * w_rj represents the weight connecting nodes r and j.
     *
     * @param currentGradients gradient for the next layer neuron
     * @param currentLayer Layer object of current layer
     * @param previousLayer Layer just before the current layer
     * @return
     */
    private float[] backpropagateHiddenLayer(float[] currentGradients, Layer currentLayer, Layer previousLayer) {
        int currentNeuronCount = currentLayer.neurons.length;
        int previousNeuronCount = previousLayer.neurons.length;

        float[] previousGradients = new float[previousNeuronCount]; // Gradient for previous layer

        for (int j = 0; j < previousNeuronCount; j++) {
            previousGradients[j] = 0; // Initialize gradient
            for (int r = 0; r < currentNeuronCount; r++) {
                // Backpropagate gradient from current layer
                previousGradients[j] +=
                        currentGradients[r] *
                        currentLayer.neurons[r].weights[j] *
                        activationDerivative(currentLayer.neurons[r].getInnerPotential(), currentLayer.activation_function);
            }
        }
        return previousGradients;
    }

    /**
     * Clips the gradients to prevent exploding.
     * −clipValue ≤ gradients[i][j] ≤ clipValue
     * (see the slide "Issues in gradient descent – too fast descent")
     * @param gradients 2D array f gradients
     * @param clipValue treshold value
     * @return
     */
    public static float[][] clipGradients(float[][] gradients, float clipValue) {
        for (int i = 0; i < gradients.length; i++) {
            for (int j = 0; j < gradients[i].length; j++) {
                gradients[i][j] = Math.max(-clipValue, Math.min(clipValue, gradients[i][j]));
            }
        }
        return gradients;
    }


    /**
     * BACKPROPAGATION and WEIGHTS UPDATE
     * First, it handles the output layer separately, and then it loops over the hidden layers.
     * At each layer:
     * 1) output gradients are computed,
     * 2) weight gradients are computed and
     * 3) weights are updated.
     * @param learning_rate the rate by which the weights change
     * @param target list of desired probabilities given by label
     * @param outputs list of computed probabilities from forward pass
     */
    public void train(float learning_rate, float[] target, float[] outputs){
        System.out.println("\nTraining...");

        float loss = Util.crossEntropy(target, outputs);
        //System.out.println("cross entropy: "+loss);

        // ---------------OUTPUT LAYER -------------------
        Layer outputLayer = layers[layers.length - 1];

        // 1) output gradients
        float[] output_layer_gradients = computeOutputLayerGradients(outputs,target); //gradient wrt y

        // 2) weight gradients
        float[][] output_layer_weight_gradients = outputLayer.computeOutputLayerWeightGradients(output_layer_gradients); //gradient wrt w
        // Clip gradients for output layer
        output_layer_weight_gradients = clipGradients(output_layer_weight_gradients, 5.0f); // Example clip value
        //TODO how to choose a good clip value? recommended 1-5, but possible up to 20... - HYPERPARAMETER

        // 3) weight update
        outputLayer.updateWeights(output_layer_weight_gradients, learning_rate);
        System.out.println("Training of the output layer completed");

        // -----------------HIDDEN LAYERS----------------------

        float[] current_output_gradient = output_layer_gradients; //move from output layer

        for (int i = layers.length - 1; i > 1; i--) {
            Layer currentLayer = layers[i];
            Layer previousLayer = layers[i - 1];
            System.out.println("Passing from " + i + " to " + (i-1));

            // 1) output gradients
            float[] previous_output_gradient = backpropagateHiddenLayer(current_output_gradient, currentLayer, previousLayer);

            // 2) weight gradients
            float[][] previous_weight_gradients = previousLayer.computeWeightGradients(previous_output_gradient);
            // Clip gradients for the current hidden layer
            previous_weight_gradients = clipGradients(previous_weight_gradients, 5.0f); // Example clip value

            // 3) weight update
            previousLayer.updateWeights(previous_weight_gradients, learning_rate);

            current_output_gradient = previous_output_gradient; //move to another layer
        }
        System.out.println("Training completed");
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

    /**
     * Prints brief information about a network instance.
     */
    public void printInfo() {
        System.out.println("---Network - number of layers: " + layers.length);
        for (Layer layer : layers){
            layer.printInfoLine();
        }

    }
}

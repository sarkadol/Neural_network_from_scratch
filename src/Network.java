package src;

import java.util.Arrays;

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
            gradients[i] = predicted[i] - target[i];
        }
        return gradients;
    }

    /**
     * Computes the weight gradients of the output layer (including bias gradient)
     *
     * For the formula, see the {@link #computeOutputLayerWeightGradients(float[], float[])} computeWeightGradients}
     *
     * @param output_layer_gradients array of output layer gradients
     * @param previous_layer_outputs array of previous layer outputs
     * @return weight gradients for output layer including bias gradient
     */
    private float[][] computeOutputLayerWeightGradients(float[] output_layer_gradients, float[] previous_layer_outputs) {
        int num_neurons = output_layer_gradients.length; // Number of neurons in the output layer
        int num_inputs = previous_layer_outputs.length; // Number of inputs to the output layer

        float[][] weight_gradients = new float[num_neurons][num_inputs + 1]; // +1 for bias gradient
        for (int i = 0; i < num_neurons; i++) { // For each neuron in the output layer
            for (int j = 0; j < num_inputs; j++) { // For each weight of that neuron
                weight_gradients[i][j] = output_layer_gradients[i] * previous_layer_outputs[j];
            }
            // Compute the bias gradient as the last element in the row
            weight_gradients[i][num_inputs] = output_layer_gradients[i];
        }

        return weight_gradients;
    }

    /**
     * Backpropagation of hidden layer:
     * (∂E_k / ∂y_j) = Σ_(r ∈ j→) [(∂E_k / ∂y_r) * σ'_r(ξ_r) * w_rj]
     *
     * r ∈ j→ indicates that r is in the set of nodes connected to j.
     * σ'_r(ξ_r) is the derivative of the activation function with respect to ξ_r.
     * w_rj represents the weight connecting nodes r and j.
     *
     * @param nextLayerGradient gradient for the next layer neuron
     * @param currentLayer Layer object of current layer
     * @param previousLayer Layer just before the current layer
     * @return
     */
    private float[] backpropagateHiddenLayer(float[] nextLayerGradient, Layer currentLayer, Layer previousLayer) {
        int currentNeuronCount = currentLayer.neurons.length;
        int previousNeuronCount = previousLayer.neurons.length;

        float[] currentGradient = new float[previousNeuronCount]; // Gradient for previous layer

        for (int j = 0; j < previousNeuronCount; j++) {
            currentGradient[j] = 0; // Initialize gradient
            for (int r = 0; r < currentNeuronCount; r++) {
                // Backpropagate gradient from current layer
                float weightGradient = nextLayerGradient[r] * currentLayer.neurons[r].weights[j];
                currentGradient[j] += weightGradient;
            }
            // Multiply by activation function derivative
            currentGradient[j] *= activationDerivative(previousLayer.neurons[j].computeInnerPotential(), previousLayer.activation_function);
        }
        return currentGradient;
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
     * BACKPROPAGATION.
     * First, it handles the output layer separately and then it loops over the hidden layers.
     * @param learning_rate the rate by which the weights change
     * @param target list of desired probabilities given by label
     * @param outputs list of computed probabilities from forward pass
     */
    public void BackPropagation(float learning_rate, float[] target, float[] outputs){
        System.out.println("\nBackpropagation...");

        // Step 2: Compute the gradient of the loss function at the output layer - see the improvement during training
        float loss = Util.crossEntropy(target, outputs);
        //System.out.println("cross entropy: "+loss);

        // Step 3: Compute gradients for the output layer using softmax + cross-entropy derivative
        // OUTPUT LAYER separately
        Layer outputLayer = layers[layers.length - 1];
        Layer hiddenLayerBeforeOutput = layers[layers.length - 2];
        //outputLayer.printInfoLine();
        //hiddenLayerBeforeOutput.printInfoLine();
        System.out.println("Passing from output layer (" + (layers.length - 1)+ ") to " + (layers.length - 2));

        float[] output_layer_gradients = computeOutputLayerGradients(target, outputs); //gradient wrt y
        float[][] output_layer_weight_gradients = computeOutputLayerWeightGradients(output_layer_gradients, hiddenLayerBeforeOutput.y); //gradient wrt w

        // Clip gradients for output layer
        output_layer_weight_gradients = clipGradients(output_layer_weight_gradients, 5.0f); // Example clip value
        //TODO how to choose a good clip value? recommended 1-5, but possible up to 20... - HYPERPARAMETER

        outputLayer.updateWeights(output_layer_weight_gradients, learning_rate);
        System.out.println("Backpropagation of the output layer completed");

        // Step 4: Backward pass through hidden layers
        // HIDDEN LAYERS
        float[] current_output_gradient = backpropagateHiddenLayer(output_layer_gradients, outputLayer, hiddenLayerBeforeOutput);
        for (int i = layers.length - 2; i > 0; i--) {
            Layer currentLayer = layers[i];
            Layer previousLayer = layers[i - 1];

            System.out.println("Passing from " + i + " to " + (i-1));
            //currentLayer.printInfoLine();
            //previousLayer.printInfoLine();

            float[][] current_weight_gradients = currentLayer.computeWeightGradients(current_output_gradient);

            // Clip gradients for the current hidden layer
            current_weight_gradients = clipGradients(current_weight_gradients, 5.0f); // Example clip value

            currentLayer.updateWeights(current_weight_gradients, learning_rate);
            // Stop backpropagation before reaching the input layer
            if (i > 1) {
                current_output_gradient = backpropagateHiddenLayer(current_output_gradient, currentLayer, previousLayer);
            }
        }
        System.out.println("Backpropagation completed");
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

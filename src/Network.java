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
        for (int i = 1; i < layers.length; i++){    // V tuto chvíli je nepotřebná vstupní vrstva
            layers[i].setX(inputs); // Save respective attributes
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

    private float[][] computeOutputLayerWeightGradients(float[] target, float[] outputs, float[] previous_layer_outputs) {
        // Step 1: Compute the gradient of the loss function with respect to the outputs
        float[] output_layer_gradient = computeOutputLayerGradient(target, outputs);

        // Step 2: Compute weight gradients for the output layer
        int num_neurons = outputs.length; // Number of neurons in the output layer
        int num_inputs = previous_layer_outputs.length; // Number of inputs to the output layer

        float[][] weight_gradients = new float[num_neurons][num_inputs + 1]; // +1 for bias gradient
        for (int i = 0; i < num_neurons; i++) { // For each neuron in the output layer
            for (int j = 0; j < num_inputs; j++) { // For each weight of that neuron
                weight_gradients[i][j] = output_layer_gradient[i] * previous_layer_outputs[j];
            }
            // Compute the bias gradient as the last element in the row
            weight_gradients[i][num_inputs] = output_layer_gradient[i];
        }

        return weight_gradients;
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
            currentGradient[i] *= activationDerivative(previousLayer.neurons[i].computeInnerPotential(), previousLayer.activation_function);
        }
        return currentGradient;
    }



    /**
     *
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
        Layer outputLayer = layers[layers.length - 1];
        Layer hiddenLayerBeforeOutput = layers[layers.length - 2];
        outputLayer.printInfoLine();
        hiddenLayerBeforeOutput.printInfoLine();

        float[] output_layer_gradients = computeOutputLayerGradient(target, outputs);
        //System.out.println("output layer gradients: "+ Arrays.toString(output_layer_gradients));
        float[][] output_layer_weight_gradients = computeOutputLayerWeightGradients(target, outputs, hiddenLayerBeforeOutput.y);
        outputLayer.updateWeights(output_layer_weight_gradients, learning_rate);
        System.out.println("Backpropagation of the output layer completed");

        // Step 4: Backward pass through hidden layers
        float[] current_output_gradient = output_layer_gradients;
        current_output_gradient = backpropagateHiddenLayer(current_output_gradient, outputLayer, hiddenLayerBeforeOutput);

        for (int i = layers.length - 2; i > 0; i--) {
            Layer currentLayer = layers[i];
            Layer previousLayer = layers[i - 1];

            System.out.println("Passing from " + i + " to " + (i-1));
            currentLayer.printInfoLine();
            previousLayer.printInfoLine();

            float[][] current_weight_gradients = currentLayer.computeWeightGradients(current_output_gradient);
            System.out.println("output gradients "+current_output_gradient.length);
            System.out.println("weight gradients "+current_weight_gradients.length);
            currentLayer.updateWeights(current_weight_gradients, learning_rate);
            //current_output_gradient = backpropagateHiddenLayer(current_output_gradient, currentLayer, previousLayer);
            if (i > 1) { // Stop backpropagation before reaching the input layer
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

    public void printInfo() {
        System.out.println("---Network - number of layers: " + layers.length);
        for (Layer layer : layers){
            layer.printInfoLine();
        }

    }
}

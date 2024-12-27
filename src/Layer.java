package src;

import java.util.Arrays;
import java.util.Random;


public class Layer {
    String activation_function;
    float[] x; //inputs
    float[] y; //outputs
    float[][] layer_weights; //TODO weights of neuron w[j][i] from i to j. w[j][0] = -bias[j]
    float [] layer_inner_potentials;
    float [][] prev_weight_updates;

    /**
     * creates a hidden/output layer
     *
     * @param activation_function specifies activation function used in this layer
     * @param previous_layer      the previous layer
     * @param neurons_number      number of neurons in this layer
     */
    public Layer(Layer previous_layer, int neurons_number, String activation_function) {
        this.activation_function = activation_function;
        int input_length = previous_layer.getOutputLength();
        this.x = new float[input_length]; // number of inputs for this layer
        this.y = new float[neurons_number];

        this.layer_weights = new float[neurons_number][input_length+1]; //one more for bias
    }

    /**
     * Creates an input layer
     *
     * @param input_neurons_number number of values e.g. from csv (needed for the next layer for weights)
     */
    public Layer(int input_neurons_number) {
        this.y = new float[input_neurons_number];
    }

    public void setX(float[] x) {
        this.x = x;
    }

    /**
     * Returns the length of an input ( = number of neurons in the previous layer)
     *
     * @return length of an array y (inputs)
     */
    public int getInputLength() {
        if (x != null) {
            return x.length;
        } else {
            return 0;
        }
    }

    /**
     * Returns the length of an output ( = number of neurons in the previous layer)
     *
     * @return length of an array y (outputs)
     */
    public int getOutputLength() {
        if (y != null) {
            return y.length;
        } else {
            return 0;
        }
    }

    public float[] getYs(){
        return y;
    }

    /*
    public Neuron getNeuron(int index){
        return null;
    }*/

    /**
     * Randomly initializes weights and biases for each neuron in this layer.
     * The method uses different initialization strategies depending on the activation function of the layer:
     * - He Initialization for ReLU activation to prevent exploding/vanishing gradients during training.
     * - He formula: w_i ∈ 𝒩(0, 2 / n)
     * - Glorot (Xavier) Initialization for softmax, tanh, or linear activations to maintain variance consistency.
     * - Glorot fomula: w_i ∼ 𝒩(0, 2 / (m + n)),
     * where n is the number of inputs to the layer, m is the number of neurons in the layer above
     */
    public void InitializeWeights() {
        Random random = new Random(); //TODO check if it is correct
        boolean useReLU = activation_function.equals("relu"); // If the activation function is relu, it uses different intialization

        // Calculate the standard deviation for the normal distribution
        float stddev = useReLU
                ? (float) Math.sqrt(2.0 / x.length) // He Initialization
                : (float) Math.sqrt(2.0 / (x.length + y.length)); // Xavier Normal Initialization

        // Initialize weights and biases
        for (int j = 0; j < y.length; j++) {
            float[] neuron_weights = new float[x.length]; // array of weights of one neuron
            for (int i = 0; i < x.length; i++) {
                // Sample weights from a normal distribution with the calculated standard deviation
                neuron_weights[i] = (float) (random.nextGaussian() * stddev);
                //System.out.printf("Initializing weight for neuron[%d] from input[%d]: %f\n", j, i, neuron_weights[i]);

                layer_weights[j][i+1] = neuron_weights[i];
            }
            layer_weights[j][0] = 0; //bias is 0
        }
        //System.out.println(Arrays.deepToString(layer_weights));
        //System.out.println("Weights initialized");
    }

    /**
     * Computes the inner potentials (ξ_j) for all neurons in the layer.
     * Formula: computes the inner potential of neuron j, with inputs i
     * ξ_j = ∑ w_ji * y_i
     * Careful! bias of neuron j is stored in w_j0, so it needs to be shifted by 1 when computing.
     * @return
     */
    public float[] layerComputeInnerPotentials() {
        int neuron_count = layer_weights.length; // Number of neurons in the layer: j
        int input_count = layer_weights[0].length - 1; // Number of inputs (excluding bias): i - 1
        layer_inner_potentials = new float[neuron_count]; // Array to store inner potentials: ξ_j

        for (int j = 0; j < neuron_count; j++) {
            float inner_potential = 0; // Initialize inner potential for neuron j
            for (int i = 0; i < input_count; i++) {
                inner_potential += layer_weights[j][i+1] * x[i];
                // Sum over inputs and weights (and take into account the bias - skip it for now)
            }
            inner_potential += layer_weights[j][0];// bias
            layer_inner_potentials[j] = inner_potential; // Store computed inner potential
        }
        return layer_inner_potentials; // Return the array of inner potentials
    }

    /**
     * Computes the output of this layer, by applying the layer's activation function to the neuron's inner potentials
     * obtained from the input array and the weights
     * Formula: y_j = σ_j(ξ_j)
     *
     * @param input array of input floats
     * @return output
     */
    public float[] computeOutput(float[] input) {
        int output_length = getOutputLength();

        layer_inner_potentials = layerComputeInnerPotentials();
        float[] outputs;
        if (activation_function.equals("softmax")) { //softmax is a special case
            outputs = Util.softmax(layer_inner_potentials);
        } else {
            outputs = new float[output_length];
            for (int j = 0; j < output_length; j++) { //for each neuron j
                outputs[j] = Util.activationFunction(layer_inner_potentials[j], activation_function);
            }
        }
        y = outputs; // Save neurons' outputs
        return outputs;
    }


    /**
     * Computes the weight gradients of the output layer (including bias gradient)
     *
     * For the formula, see the {@link #computeWeightGradients(float[])}
     *
     * @param output_layer_inner_potential_gradients array of output layer inner potential gradients
     * @return weight gradients for output layer including bias gradient
     */
    public float[][] computeOutputLayerWeightGradients(float[] output_layer_inner_potential_gradients) {
        int num_neurons = output_layer_inner_potential_gradients.length; // Number of neurons in the output layer
        int num_inputs = x.length; // Number of inputs to the output layer

        float[][] weight_gradients = new float[num_neurons][num_inputs + 1]; // +1 for bias gradient
        for (int j = 0; j < num_neurons; j++) { // For each neuron in the output layer
            for (int i = 1; i < num_inputs+1; i++) { // For each weight of that neuron (skipping bias)
                weight_gradients[j][i] = output_layer_inner_potential_gradients[j] * x[i-1];
            }
            // Compute the bias gradient as the last element in the row
            weight_gradients[j][0] = output_layer_inner_potential_gradients[j];
        }
        return weight_gradients;
    }

    /**
     * Initializes weight gradients as a zero matrix for all neurons and their inputs.
     * @return A 2D float array representing the initialized weight gradients.
     */
    public float[][] initializeWeightGradients() {
        return new float[y.length][x.length + 1];
    }

    /**
     * Adds current weight gradients to the total weight gradients for accumulation.
     * @param weight_gradients Current weight gradients.
     * @param total_weight_gradients Accumulated weight gradients.
     * @return Updated total weight gradients.
     */
    public float[][] addWeightGradients(float[][] weight_gradients, float[][] total_weight_gradients) {
        for (int i = 0; i < total_weight_gradients.length; i++) {
            for (int j = 0; j < total_weight_gradients[i].length; j++) {
                total_weight_gradients[i][j] += weight_gradients[i][j];
            }
        }
        return total_weight_gradients;
    }

    /**
     * Converts gradient of outputs of neurons to gradient of their weights; works for layers with a simple activation
     * function, that takes as an argument only the inner potential of the neuron
     * <p>
     * (∂E_k / ∂w_ji) = (∂E_k / ∂y_j) * σ'_j(ξ_j) * y_i
     *
     * @param output_gradients an array of output gradients of each neuron in the layer
     * @return a matrix of gradients of weights where on each row there are all the weights of a single neuron
     * in the layer
     */
    public float[][] computeWeightGradients(float[] output_gradients) {
        if (output_gradients.length != y.length) {
            throw new IllegalArgumentException("Gradients length must match the number of neurons. \n" +
                    "Gradients length: " + output_gradients.length + ", Neurons length: " + y.length);
        }

        float[][] weight_gradients = new float[y.length][x.length + 1];// plus bias
        float weight_independent_part;
        for (int j = 0; j < y.length; j++) { //For each neuron
            //weight_independent_part = output_gradients[j] * Util.activationFunctionDerivative(neurons[j].getInnerPotential(), activation_function);
            weight_independent_part = output_gradients[j] * Util.activationFunctionDerivative(layer_inner_potentials[j], activation_function);

            for (int i = 0; i < x.length; i++) {    //For each weight (skipping bias)
                weight_gradients[j][i+1] = weight_independent_part * x[i];
            }
            // Add bias gradient as the last element
            weight_gradients[j][0] = weight_independent_part;
        }
        //System.out.println("Weight gradients: " + Arrays.deepToString(weight_gradients));
        return weight_gradients;
    }

    /**
     * Updates the weights and biases of neurons using the weight gradients.
     * <p>
     * Formula: Δw_ji(t) = -ε · ∂E / ∂w_ji + α · Δw_ji(t-1)
     * (see the slide 274 "SGD with momentum")
     *
     * @param weight_gradients ∂E / ∂w_ji - gradients for each neuron including bias gradient
     * @param learningRate     ε - the rate at which weights and biases are adjusted
     * @param momentum         α - momentum at which weights and biases are adjusted
     * @param weight_decay     additive weight decay
     */
    public void updateWeights(float[][] weight_gradients, float learningRate, float momentum, float weight_decay) {
        // Validate parameters
        if (weight_gradients == null || y == null) {
            throw new IllegalArgumentException("Gradients and inputs must not be null.");}
        if (weight_gradients[0].length != x.length + 1) {
            throw new IllegalArgumentException("Gradients width must match the number of weights plus 1 for bias. \n" +
                    "Gradients width: " + weight_gradients[0].length + ", Expected width: " + (x.length + 1));}
        if (learningRate <= 0) {
            throw new IllegalArgumentException("Learning rate must be greater than 0.");}

        // Initialize `prev_weight_updates` if not already done
        if (prev_weight_updates == null) {
            prev_weight_updates = new float[y.length][x.length + 1];
        }

        for (int j = 0; j < y.length; j++) { // For each neuron in this layer
            for (int i = 0; i <= x.length; i++) { // For each weight and bias
                float regularization = (i == 0) ? 0 : 2 * weight_decay * layer_weights[j][i]; // Regularization for weights, not bias
                float current_update = -learningRate * (weight_gradients[j][i] + regularization)
                        + momentum * prev_weight_updates[j][i];

                // Update weight/bias
                layer_weights[j][i] += current_update;

                // Store current update for momentum
                prev_weight_updates[j][i] = current_update;
            }
        }
    }

    /**
     * Prints information about the layer
     *
     *
     */
    public void printInfo() {
        System.out.println("---Layer Info:");
        System.out.println("activation function: " + activation_function);
        System.out.println("x count: " + (x != null ? x.length : 0));
        System.out.println("y count: " + (y != null ? y.length : 0));
    }

    /**
     * SINGLE LINE.
     * Prints the activation function, count of inputs, neurons, and outputs for a quick overview.
     */
    public void printInfoLine() {
        System.out.println(
                "-Layer - " +
                        "activation function: " + activation_function + ", " +
                        "x count: " + (x != null ? x.length : 0) + ", " +
                        "y count: " + (y != null ? y.length : 0));
    }
}

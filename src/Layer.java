package src;

import java.util.Arrays;
import java.util.Random;


public class Layer {
    public Neuron[] neurons;
    String activation_function;
    float[] x; //inputs
    float[] y; //outputs

    /**
     * creates a hidden/output layer
     *
     * @param activation_function specifies activation function used in this layer
     * @param previous_layer      the previous layer
     * @param neurons_number      number of neurons in this layer
     */
    public Layer(Layer previous_layer, int neurons_number, String activation_function) {
        this.activation_function = activation_function;
        this.neurons = new Neuron[neurons_number]; // Initialize the neurons array
        for (int i = 0; i < neurons_number; i++) {
            //float weights = 0;
            //float bias = 0;
            this.neurons[i] = new Neuron();
        }
        int input_length = previous_layer.getOutputLength();
        this.x = new float[input_length]; // number of inputs for this layer
        this.y = new float[this.neurons.length];
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
        boolean useReLU = false;

        if (activation_function.equals("relu")) {  // If the activation function is relu, it uses different intialization
            useReLU = true;
        }

        float range = 0.05f; // Adjust this to control how close to 0 the weights should be
        float stddev = 0;

        if (useReLU) {
            stddev = (float) Math.sqrt(2.0 / x.length); // He Initialization for ReLU
        } else {
            range = (float) Math.sqrt(6.0 / (x.length + neurons.length)); // Xavier (Glorot) Initialization for softmax/tanh/linear
        }

        for (int i = 0; i < neurons.length; i++) {//each neuron i

            float[] neuron_weights = new float[x.length];// array of weights of one neuron
            float bias = 0; // Initialize bias to zero (or small constant)
            //float bias = (random.nextFloat() * 2 - 1) * range;

            for (int j = 0; j < x.length; j++) {//each input j in particular neuron
                // Generate weights based on the chosen method
                if (useReLU) {
                    neuron_weights[j] = (float) (random.nextGaussian() * stddev);
                } else {
                    neuron_weights[j] = random.nextFloat() * 2 * range - range;
                }
            }
            neurons[i].setWeights(neuron_weights);
            neurons[i].setBias(bias);
            //neurons[i].printInfo();
        }
        System.out.println("Weights initialized");
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
        float[] inner_potentials = new float[output_length];
        for (int j = 0; j < output_length; j++) { //for each neuron j
            neurons[j].setX(input);
            inner_potentials[j] = neurons[j].computeInnerPotential();
            //System.out.println("Neuron " + j + " inner potential: " + inner_potentials[j]);
        }
        float[] outputs;
        if (activation_function.equals("softmax")) { //softmax is a special case
            outputs = Util.softmax(inner_potentials);
        } else {
            outputs = new float[output_length];
            for (int j = 0; j < output_length; j++) { //for each neuron j
                outputs[j] = Util.activationFunction(inner_potentials[j], activation_function);
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
     * @param output_layer_gradients array of output layer gradients
     * @return weight gradients for output layer including bias gradient
     */
    public float[][] computeOutputLayerWeightGradients(float[] output_layer_gradients) {
        int num_neurons = output_layer_gradients.length; // Number of neurons in the output layer
        int num_inputs = x.length; // Number of inputs to the output layer

        float[][] weight_gradients = new float[num_neurons][num_inputs + 1]; // +1 for bias gradient
        for (int j = 0; j < num_neurons; j++) { // For each neuron in the output layer
            for (int i = 1; i < num_inputs+1; i++) { // For each weight of that neuron (skipping bias)
                weight_gradients[j][i] = output_layer_gradients[j] * x[i-1];
            }
            // Compute the bias gradient as the last element in the row
            weight_gradients[j][0] = output_layer_gradients[j];
        }
        return weight_gradients;
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
        if (output_gradients.length != neurons.length) {
            throw new IllegalArgumentException("Gradients length must match the number of neurons. \n" +
                    "Gradients length: " + output_gradients.length + ", Neurons length: " + neurons.length);
        }

        float[][] weight_gradients = new float[neurons.length][x.length + 1];// plus bias
        float weight_independent_part;
        for (int j = 0; j < neurons.length; j++) { //For each neuron
            weight_independent_part = output_gradients[j] * Util.activationFunctionDerivative(neurons[j].getInnerPotential(), activation_function);
            for (int i = 1; i < x.length+1; i++) {    //For each weight (skipping bias)
                weight_gradients[j][i] = weight_independent_part * x[i-1];
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

        for (int i = 0; i < neurons.length; i++) { //for each neuron in this layer:
            float[] weights = neurons[i].getWeights();

            if (neurons[i].getPrevWeightUpdate() == null) { // Initialize prevWeightUpdate if not already done
                neurons[i].setPrevWeightUpdate(new float[weights.length+1]); // all 0 instead of null
            }
            float[] prevWeightUpdate = neurons[i].getPrevWeightUpdate();
            // Update bias
            float biasGradient = weight_gradients[i][0];
            float biasRegularization = 0; // No regularization applied to the bias
            float biasUpdate = -learningRate * (biasGradient + biasRegularization) + momentum * prevWeightUpdate[0];
            neurons[i].setBias(neurons[i].getBias() + biasUpdate);
            prevWeightUpdate[0] = biasUpdate; // Store the update for momentum

            //update weights and bias of a neuron
            for (int j = 1; j < weights.length+1; j++) {
                //change every weight according to its gradient
                //hyperparameter learning rate
                //hyperparameter momentum
                //hyperparameter weight decay
                float regularizationTerm = 2 * weight_decay * weights[j-1];
                float currentUpdate = -learningRate * (weight_gradients[i][j-1] + regularizationTerm) + momentum * prevWeightUpdate[j-1];
                prevWeightUpdate[j] = currentUpdate; // Store the current update as the new "previous update"
            }
            // Update weights
            neurons[i].setWeights(weights);
            neurons[i].setPrevWeightUpdate(prevWeightUpdate); //for momentum
        }
    }

    /**
     * Prints information about the layer
     *
     * @param includeNeurons if true, then it prints also property of each neuron in this layer,
     *                       see {@link Neuron#printInfoLine()}
     */
    public void printInfo(boolean includeNeurons) {
        System.out.println("---Layer Info:");
        System.out.println("activation function: " + activation_function);
        System.out.println("x count: " + (x != null ? x.length : 0));
        System.out.println("neurons: " + (neurons != null ? neurons.length : 0));
        System.out.println("y count: " + (y != null ? y.length : 0));


        if (includeNeurons && neurons != null) {
            for (int i = 0; i < neurons.length; i++) {
                //System.out.print("Neuron " + (i + 1) + ": ");
                System.out.printf("Neuron %-4d ", (i + 1)); // Why different indexing?
                neurons[i].printInfoLine();
            }
        }
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
                        "neurons count: " + (neurons != null ? neurons.length : 0) + ", " +
                        "y count: " + (y != null ? y.length : 0));
    }
}

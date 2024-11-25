package src;

import java.util.Arrays;
import java.util.Random;


public class Layer {
    public Neuron[] neurons;
    String activation_function;
    float[] x; //number of inputs
    float[] y; //number of inputs
    /**
     * creates a hidden/output layer
     * @param activation_function specifies activation function used in this layer
     * @param previous_layer the previous layer
     * @param neurons_number number of neurons in this layer
     */
    public Layer(Layer previous_layer, int neurons_number, String activation_function) {
        this.activation_function = activation_function;
        this.neurons = new Neuron[neurons_number]; // Initialize the neurons array
        for(int i = 0; i < neurons_number; i++) {
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
     * @return length of an array y (inputs)
     */
    public int getInputLength(){
        if (x !=null) {return x.length;}
        else{return 0;}
    }
    /**
     * Returns the length of an output ( = number of neurons in the previous layer)
     * @return length of an array y (outputs)
     */
    public int getOutputLength(){
        if (y !=null) {return y.length;}
        else{return 0;}
    }
    /*
    public Neuron getNeuron(int index){
        return null;
    }*/

    /**
     * Randomly initializes weights and biases for each neuron in this layer.
     * The method uses different initialization strategies depending on the activation function of the layer:
     * - He Initialization for ReLU activation to prevent exploding/vanishing gradients during training.
     * - Xavier (Glorot) Initialization for softmax, tanh, or linear activations to maintain variance consistency.
     *
     */
    public void InitializeWeights(){
        Random random = new Random(); //TODO check if it is correct
        boolean useReLU = false;

        if(activation_function.equals("relu")){  // If the activation function is relu, it uses different intialization
            useReLU = true;}

        float range = 0.05f; // Adjust this to control how close to 0 the weights should be
        float stddev = 0;

        if (useReLU) {
            stddev = (float) Math.sqrt(2.0 / x.length); // He Initialization for ReLU
        } else {
            range = (float) Math.sqrt(6.0 / (x.length + neurons.length)); // Xavier (Glorot) Initialization for softmax/tanh/linear
        }

        for(int i = 0; i < neurons.length; i++){//each neuron i

            float[] neuron_weights = new float[x.length];// array of weights of one neuron
            float bias = 0; // Initialize bias to zero (or small constant)
            //float bias = (random.nextFloat() * 2 - 1) * range;

            for(int j = 0; j < x.length; j++){//each input j in particular neuron
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
     * @param input array of input floats
     * @return output
     */
    public float[] computeOutput(float[] input){
        int output_length = getOutputLength();
        float[] inner_potentials = new float[output_length];
        for(int i = 0; i < output_length; i++){
            neurons[i].setX(input);
            inner_potentials[i] = neurons[i].computeInnerPotential();
            //System.out.println("Neuron " + i + " inner potential: " + inner_potentials[i]);

        }
        float[] output;
        if (activation_function.equals("softmax")){
            //System.out.println("Inner potentials before softmax: " + Arrays.toString(inner_potentials));
            output = Util.softmax(inner_potentials);
            //System.out.println("Outputs after softmax: " + Arrays.toString(output));

        }
        else{
            output = new float[output_length];
            for(int i = 0; i < output_length; i++){
                output[i] = Util.activationFunction(inner_potentials[i], activation_function);
                //System.out.println("Activation output for neuron " + i + ": " + output[i]);

            }
        }
        y = output; // Save respective attribute
        return output;
    }

    /**
     * Prints information about the layer
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
                "-Layer - "+
                "activation function: " + activation_function + ", " +
                        "x count: " + (x != null ? x.length : 0) + ", " +
                        "neurons count: " + (neurons != null ? neurons.length : 0) + ", " +
                        "y count: " + (y != null ? y.length : 0));
    }

    /**
     * Converts gradient of outputs of neurons to gradient of their weights; works for layers with a simple activation
     * function, that takes as an argument only the inner potential of the neuron
     *
     * (∂E_k / ∂w_ji) = (∂E_k / ∂y_j) * σ'_j(ξ_j) * y_i
     *
     * @param output_gradients an array of output gradients of each neuron in the layer
     * @return a matrix of gradients of weights where on each row there are all the weights of a single neuron
     * in the layer
     */
    public float[][] computeWeightGradients(float[] output_gradients) {
        if (output_gradients.length != neurons.length) {
            throw new IllegalArgumentException("Gradients length must match the number of neurons. \n"+
                    "Gradients length: " + output_gradients.length + ", Neurons length: " + neurons.length);}

        float[][] weight_gradients = new float[neurons.length][x.length+1];// plus bias
        float weight_independent_part;
        for (int j = 0; j < neurons.length; j++) { //For each neuron
            weight_independent_part = output_gradients[j] * Util.activationFunctionDerivative(neurons[j].getInnerPotential(), activation_function);
            for (int i = 0; i < x.length; i++) {    //For each weight
                weight_gradients[j][i] = weight_independent_part * x[i];
            }
            // Add bias gradient as the last element
            weight_gradients[j][x.length] = weight_independent_part;
        }
        return weight_gradients;
    }



    /**
     * Updates the weights and biases of neurons using the weight gradients.
     *
     * @param weight_gradients    the gradients for each neuron including bias gradient
     * @param learningRate the rate at which weights and biases are adjusted
     */
    public void updateWeights(float[][] weight_gradients, float learningRate) {
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
            //update weights
            for (int j = 1; j < weights.length; j++) { //for each weight of a neuron except the first one which is bias
                //change every weight according to its gradient
                weights[j] = weights[j] - learningRate * weight_gradients[i][j];}
            // Update weights
            neurons[i].setWeights(weights);
            // Update bias
            neurons[i].setBias(neurons[i].getBias() - learningRate * weight_gradients[i][0]);
        }
        System.out.println("Weights and bias updated");
    }

}

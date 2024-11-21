package src;

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

    public String getActivation_function(){
        return this.activation_function;
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
     * Randomly initializes weights and biases for each neuron in this layer
     */
    public void InitializeWeights(){
        Random random = new Random(); //TODO check if it is correct
        float range = 0.05f; // Adjust this to control how close to 0 the weights should be

        for(int i = 0; i < neurons.length; i++){//each neuron i ...= for(Neuron neuron : neurons)

            float[] neuron_weights = new float[x.length];// array of weights of one neuron
            float bias = (random.nextFloat() * 2 - 1) * range;
            for(int j = 0; j < x.length; j++){//each input j in particular neuron
                //random.nextFloat() generates a number between 0 and 1.
                neuron_weights[j] = (random.nextFloat() * 2 - 1) * range; // Generate weights in the range -0.05 to 0.05
                //Multiplying by 2 and subtracting 1 shifts the range to [-1,1]
                //multiplying by range (0.05 in this case) scales it to [-0.05,0.05]
                //neuron_weights[j] = weights[i][j];
            }
            neurons[i].setWeights(neuron_weights);
            neurons[i].setBias(bias);
            //neurons[i].printInfo();
        }
        System.out.println("Weights initialized");
    }

    /**
     * Computes the output of this layer, by ...
     * @param input array of input floats
     * @return output
     */
    public float[] computeOutput(float[] input){
        int output_length = getOutputLength();
        float[] inner_potentials = new float[output_length];
        for(int i = 0; i < output_length; i++){
            neurons[i].setX(input);
            inner_potentials[i] = neurons[i].computeInnerPotential();
        }
        float[] output;
        if (activation_function.equals("softmax")){
            output = Util.softmax(inner_potentials);
        }
        else{
            output = new float[output_length];
            for(int i = 0; i < output_length; i++){
                output[i] = Util.activationFunction(inner_potentials[i], activation_function);
            }
        }
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

    public float[][] outputGradientsToWeightGradients(float[] output_gradients) {
        if (output_gradients.length != neurons.length) {
            throw new IllegalArgumentException("Gradients length must match the number of neurons.");}
        float[][] result = new float[neurons.length][x.length];
        float weight_independent_part;
        for (int i = 0; i < neurons.length; i++) { //For each neuron
            weight_independent_part = output_gradients[i] * Util.activationFunctionDerivative(neurons[i].getInnerPotential(), activation_function);
            for (int j = 0; j < x.length; j++) {    //For each weight
                result[i][j] = weight_independent_part * x[j];
            }
        }
        return result;
    }

    /**
     * Updates the weights and biases of neurons using the provided gradients and inputs.
     *
     * @param weight_gradients    the gradients for each neuron
     * @param learningRate the rate at which weights and biases are adjusted
     */
    public void updateWeights(float[][] weight_gradients, float learningRate) {
        // Validate parameters
        if (weight_gradients == null || y == null) {
            throw new IllegalArgumentException("Gradients and inputs must not be null.");}
        if (weight_gradients[0].length != x.length + 1) {
            throw new IllegalArgumentException("Gradients width must match the number of weights plus 1 for bias.");}
        // We can check also the height of gradient matrix, it should be equal to the number of neurons
        if (learningRate <= 0) {
            throw new IllegalArgumentException("Learning rate must be greater than 0.");}

        for (int i = 0; i < neurons.length; i++) { //for each neuron in this layer:
            float[] weights = neurons[i].getWeights();
            //update weights
            for (int j = 1; j < weights.length; j++) { //for each weight of a neuron
                //weights[j] -= learningRate * gradients[i] * inputs[j];}
                weights[j] = weights[j] - learningRate * weight_gradients[i][j];}  // Why is there y

        // Update bias
            neurons[i].setWeights(weights);
            neurons[i].setBias(neurons[i].getBias() - learningRate * weight_gradients[i][0]);
        }
        System.out.println("Weights updated");
    }

}

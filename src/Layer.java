package src;

import java.util.Random;


public class Layer {
    public Neuron[] neurons;
    String activation_function; //TODO acivation function is just here but not used
    float[] y; //number of inputs
    //TODO rozlišit y - výsledky předchozí vrstvy a y - output této vrstvy
    float[][] weights; //i - number of y from the first layer, j - number of neurons second layer

    /**
     * creates a hidden/output layer
     * @param activation_function specifies activation function used in this layer
     * @param layer the previous layer
     * @param neurons_number number of neurons in this layer
     */
    public Layer(Layer layer, int neurons_number, String activation_function) {
        this.activation_function = activation_function;
        this.neurons = new Neuron[neurons_number]; // Initialize the neurons array
        for(int i = 0; i < neurons_number; i++) {
            this.neurons[i] = new Neuron();
            //weight number for each neuron = input_length needed for initialization
        }
        // if this is the first layer which has no outputs (and therefore no length), then take its inputs as input of next layer
        int input_length = 0;
        if (layer.getLength()==0) {
            input_length = layer.getInputLength();}
        else {
            input_length = layer.getLength();}
        this.y = new float[input_length]; // number of inputs for this layer
    }

    /**
     * Creates an input layer
     * @param input_neurons_number number of values e.g. from csv (needed for the next layer for weights)
     */
    public Layer(int input_neurons_number) {
        this.y = new float[input_neurons_number];
    }

    /**
     * Returns the length of a layer (number of neurons in this layer)
     * @return length of a layer
     */
    public int getLength(){
        if (neurons!=null) {
            return neurons.length;
        }
        else{
            return 0;
        }
    }
    /**
     * Returns the length of an input ( = number of neurons in the previous layer)
     * @return length of an array y (inputs)
     */
    public int getInputLength(){
        return y.length;
    }
    /*
    public Neuron getNeuron(int index){
        return null;
    }*/

    /**
     * Randomly initializes weights for each neuron in this layer
     */
    public void InitializeWeights(){
        Random random = new Random(); //TODO check if it is correct
        for(int i = 0; i < neurons.length; i++){//each neuron i ...= for(Neuron neuron : neurons)
            for(int j = 0; j < y.length; j++){//each input j in particular neuron
                weights[i][j] = random.nextFloat();
                neurons[i].weights[j] = weights[i][j]; //save this weight into one Neuron
            }
        }

    }

    /**
     * Prints information about the layer
     * @param includeNeurons if true, then it prints also property of each neuron in this layer,
     *                       see {@link Neuron#printInfoLine()}
     */
    public void printInfo(boolean includeNeurons) {
        System.out.println("---Layer Info:");
        System.out.println("Activation function: " + activation_function);
        System.out.println("Number of inputs: " + (y != null ? y.length : 0));
        System.out.println("Number of neurons: " + (neurons != null ? neurons.length : 0));

        if (includeNeurons && neurons != null) {
            for (int i = 0; i < neurons.length; i++) {
                //System.out.print("Neuron " + (i + 1) + ": ");
                System.out.printf("Neuron %-4d ", (i + 1));
                neurons[i].printInfoLine();
            }
        }
    }
}

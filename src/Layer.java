package src;

public class Layer {
    public Neuron[] neurons;
    String activation_function;
    float[] y;

    /**
     * creates a hidden/output layer
     * @param activation_function specifies activation function used in this layer
     * @param input_length length of the previous layer
     * @param neurons_number number of neurons in this layer
     */
    public Layer(int input_length, int neurons_number, String activation_function) {
        this.activation_function = activation_function;
        for(int i = 0; i < neurons_number; i++) {
            this.neurons[i] = new Neuron();
            //weight number for each neuron = input_length needed for initialization
        }
    }

    /**
     * Creates an input layer
     * @param input_neurons_number number of values e.g. from csv (needed for the next layer for weights)
     */
    public Layer(int input_neurons_number) {
        this.y = new float[input_neurons_number];
    }

    public int getLength(){
        return neurons.length;
    }
    /*
    public Neuron getNeuron(int index){
        return null;
    }*/
}

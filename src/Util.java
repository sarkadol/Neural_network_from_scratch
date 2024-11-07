package src;

import java.util.Arrays;

public class Util {
    //--------------------------------------
    /**
     * temporary main for debugging
     */
    public static void main(String[] args) {
        Util util = new Util();

        System.out.println(util.ReLU(-5));
        System.out.println(util.ReLU(0));
        System.out.println(util.ReLU(5));
        for(int i=0; i<10;i++){
            System.out.println(Arrays.toString(util.labelToVector(i)));
        }
        float[] pole = {1, 2, 3};
        float[] result = util.Softmax(pole);
        System.out.println(Arrays.toString(result)); //má být cca [0.0900,0.2447,0.6652] pro 1,2,3

    }
    //-------------------------------------------------------
    /**
     * Performs matrix multiplication.
     * If A is of size m x n and B is of size n x p, then the resulting matrix C will be of size m x p
     * @param A first matrix (m x n)
     * @param B second matrix (n x p)
     * @return matrix C (m x p)
     */
    public float[][] MatrixMultiplication(float[][] A, float[][] B)
    {// NOT SURE IF THIS IS CORRECT, CHECK IT :) TODO
        int m = A.length;          // Number of rows in A
        int n = A[0].length;       // Number of columns in A (and rows in B)
        int p = B[0].length;       // Number of columns in B

        // Check if multiplication is possible
        if (B.length != n) {
            throw new IllegalArgumentException("Number of columns in A must match number of rows in B.");
        }

        // Initialize the result matrix with dimensions m x p
        float[][] C = new float[m][p];

        // Perform matrix multiplication
        for (int i = 0; i < m; i++) {            // Iterate over rows of A
            for (int j = 0; j < p; j++) {        // Iterate over columns of B
                C[i][j] = 0;
                for (int k = 0; k < n; k++) {    // Multiply row of A by column of B
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return null; //C
    }

    /**
     *
     */
    public void InitializeWeights(){

    }
    public void ActivationFunction(){


    }

    /**
     * Softmax activation function.
     * Converts a vector of K real numbers into probability distributions of K possible outcomes
     * @param inner_potentials (ksi) vector of inner potentials
     * @return array of probabilities
     */
    public float[] Softmax(float[] inner_potentials){
        int n = inner_potentials.length;
        float[] y = new float[n];
        float sum_e = 0;

        // Step 1: Calculate the exponential of each element and sum them up
        for (int i = 0; i < n; i++) {
            y[i] = (float) Math.exp(inner_potentials[i]); //e^ksi_i
            sum_e += y[i];
        }
        // Step 2: Divide each exponential by the sum to get probabilities
        for (int i = 0; i < n; i++) {
            y[i] /= sum_e;
        }
        return y;
    }

    /**
     * Rectified Linear Unit activation function
     * @param inner_potential weighted sum + bias of a single Neuron
     * @return ReLU
     */
    public float ReLU(float inner_potential){
        return Math.max(0, inner_potential);
    }

    /** TODO
     * Calculates the categorical cross entropy loss
     * @return
     */
    public float CrossEntropy(float[] desired_output,float[] real_output){
        int p=0; //number of samples in the dataset
        Neuron[] neurons = null; // neurons
        //TBD
        float result = 0;
        return result;
    }

    /**
     * Transforms an input label into an ideal probability vector (e.g. label 2 into vector 0,0,1,0,...)
     * @param label the output label in range 0-9
     * @return vector of probabilities where exactly one value is 1, others are 0
     */
    public float[] labelToVector(int label){
        //Fashion MNIST has 10 labels 0-9:
        //"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        //"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        int classes_number = 10;
        float[] vector = new float[classes_number]; // Array is initialized to all 0s by default
        if (label >= 0 && label < classes_number) {
            vector[label] = 1;
        } else {
            throw new IllegalArgumentException("Label must be in the range 0-9");
        }
        return vector;
    }

}

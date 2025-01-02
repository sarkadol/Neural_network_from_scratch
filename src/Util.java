package src;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class Util {

    /**
     * Performs matrix multiplication.
     * If A is of size m x n and B is of size n x p, then the resulting matrix C will be of size m x p
     * @param A first matrix (m x n)
     * @param B second matrix (n x p)
     * @return matrix C (m x p)
     */
    public static float[][] matrixMultiplication(float[][] A, float[][] B)
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
        return C;
    }

    /**
     * Performs scalar multiplication of the input vectors.
     * @param u first vector of size n
     * @param v second vector of size n
     * @return the resulting product
     */
    public static float scalarProduct(float[] u, float[] v) {
        float product = 0;
        int n = u.length;

        // Check if the scalar product is possible
        if (v.length != n) {
            throw new IllegalArgumentException("Length of the vectors in scalar product must be the same.");
        }

        for (int i = 0; i < n; i++) {
            product += u[i] * v[i];
        }
        return product;
    }

    /**
     * Switch for the derivative of a simple activation function that takes only one inner potential as an argument
     * see {@link Util#activationFunction(float, String)}
     * @param input the input to the activation function derivative - inner potential
     * @param fun_name the name of the function, we want to use the derivative of
     * @return the result of applying the desired activation function derivative to the input
     */
    public static float activationFunctionDerivative(float input, String fun_name){
        switch (fun_name){
            case "relu":
                return ReLUDerivative(input);
            //case "softmax":
                //return SoftmaxDerivative();
        }
        throw new IllegalArgumentException("Unknown function: " + fun_name);
    }

    /**
     * Switch for the simple activation function that takes only one inner potential as an argument
     * @param input the input to the activation function - inner potential
     * @param function_name the name of the function, we want to use
     * @return the result of applying the desired activation function to the input
     */
    public static float activationFunction(float input, String function_name){
        switch (function_name){
            case "relu":
                return ReLU(input);
        }
        throw new IllegalArgumentException("Unknown function: " + function_name);
    }

    /**
     * Softmax activation function. Prevents overflowing by subtracting maximum value.
     * Converts a vector of K real numbers into probability distributions of K possible outcomes
     * @param inner_potentials (ksi) vector of inner potentials
     * @return array of probabilities
     */
    public static float[] softmax(float[] inner_potentials){
        int n = inner_potentials.length;
        float[] y = new float[n];
        float sum_e = 0;

        float max_potential = Float.NEGATIVE_INFINITY;

        // Step 1: Find the maximum value in the array to stabilize computations
        for (float potential : inner_potentials) {
            if (potential > max_potential) {
                max_potential = potential;
            }
        }

        // Step 2: Calculate the exponential of each element and sum them up
        for (int i = 0; i < n; i++) {
            y[i] = (float) Math.exp(inner_potentials[i]- max_potential); //e^ksi_i - max_potential
            sum_e += y[i];
        }
        // Step 3: Divide each exponential by the sum to get probabilities
        for (int i = 0; i < n; i++) {
            y[i] /= sum_e;
        }
        return y;
    }

    /**
     * Rectified Linear Unit activation function
     * Formula σ(ξ) = max(ξ, 0)
     * @param inner_potential weighted sum + bias of a single Neuron
     * @return ReLU
     */
    public static float ReLU(float inner_potential){
        return Math.max(0, inner_potential);
    }

    /**
     * Formula:
     * f(x) = { 0 if x < 0
     *          1 if x > 0 }
     * @param inner_potential
     * @return derivative of a ReLU function
     */
    public static float ReLUDerivative(float inner_potential){
        return inner_potential > 0 ? 1 : 0;
    }

    /**
     * Transforms a float array to a float array of logarithms
     * @param inputArray the array to be transformed
     * @return resulting array of natural logarithms
     */
    public static float[] mapToNaturalLog(float[] inputArray) {
        float[] resultArray = new float[inputArray.length];
        for (int i = 0; i < inputArray.length; i++) {
            resultArray[i] = (float) Math.log(inputArray[i]);
        }
        return resultArray;
    }

    /**
     * Calculates the categorical cross entropy loss
     * @param desired_output the ground truth output
     * @param real_output the output of the neural network
     * @return value of cross entropy error function
     */
    public static float crossEntropy(float[] desired_output, float[] real_output){
        //log(0) is not defined!
        //log(1) = 0
        float epsilon = 1e-8f; // Small constant to avoid log(0)
        float result = 0;

        // Step 1: Clip the real_output to avoid 0 or 1 probabilities
        float[] clipped_real_output = new float[real_output.length];
        for (int i = 0; i < real_output.length; i++) {
            clipped_real_output[i] = Math.max(epsilon, Math.min(1 - epsilon, real_output[i]));
            //Ensure all probabilities are within the range [epsilon,1−epsilon]
            //if real output is 0, it sets it very close to 0
            //if real output is 1, it sets it slightly less than one
        }
        // Step 2: Compute the logarithm of the clipped probabilities
        float[] log_real_output = mapToNaturalLog(clipped_real_output);

        // Step 3: Compute the cross-entropy as the negative scalar product
        result = -scalarProduct(log_real_output, desired_output);

        return result;
    }

    /**
     * Transforms an input label into an ideal probability vector (e.g. label 2 into vector 0,0,1,0,...)
     * @param label the output label in range 0-9
     * @return vector of probabilities where exactly one value is 1, others are 0
     */
    public static float[] labelToVector(int label){
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

    public static float activationDerivative(float input, String functionName) {
        switch (functionName.toLowerCase()) {
            case "relu":
                return input > 0 ? 1 : 0; // Derivative of ReLU
            case "softmax":
                return 1; // Not used for hidden layers in backprop
            default:
                throw new IllegalArgumentException("Unknown activation function: " + functionName);
        }
    }

    /**
     * Divides a list into smaller batches of a specified size.
     *
     * @param <T> The type of elements in the list.
     * @param list The list to be divided into batches.
     * @param batchSize The maximum size of each batch.
     * @return A list of sublists (batches) containing the elements of the original list.
     *         The last batch may contain fewer elements if the list size is not a multiple of batchSize.
     * @throws IllegalArgumentException if batchSize is less than or equal to 0.
     */
    public static <T> List<List<T>> divideToBatches(List<T> list, int batchSize) {
        int currentBatchStart = 0;
        int listLength = list.size();
        int numberOfBatches = listLength / batchSize + (listLength % batchSize == 0 ? 0 : 1);
        List<List<T>> batches = new ArrayList<>(numberOfBatches);
        for (int i = 0; i < numberOfBatches; i++) {
            batches.add(i, list.subList(currentBatchStart, Math.min(currentBatchStart + batchSize, listLength)));
            currentBatchStart += batchSize;
        }
        return batches;
    }
}

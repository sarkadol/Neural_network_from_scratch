package src;

public class Hyperparameters {
    private int epochs;
    private float learningRate;
    private float clipValue;
    private int batchSize;

    public Hyperparameters(int epochs, float learningRate, float clipValue, int batchSize) {
        this.epochs = epochs;
        this.learningRate = learningRate;
        this.clipValue = clipValue;
        this.batchSize = batchSize;
    }

    public int getEpochs() {
        return epochs;
    }
    public float getLearningRate() {
        return learningRate;
    }
    public float getClipValue() {
        return clipValue;
    }
    public int getBatchSize() {
        return batchSize;
    }
}

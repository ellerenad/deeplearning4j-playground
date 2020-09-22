package dev.ienjoysoftware.classification;

public class Iris {
    private float sepalLength;
    private float sepalWidth;
    private float petalLength;
    private float petalWidth;

    public Iris(float sepalLength, float sepalWidth, float petalLength, float petalWidth) {
        this.sepalLength = sepalLength;
        this.sepalWidth = sepalWidth;
        this.petalLength = petalLength;
        this.petalWidth = petalWidth;
    }

    public float getSepalLength() {
        return sepalLength;
    }

    public float getSepalWidth() {
        return sepalWidth;
    }

    public float getPetalLength() {
        return petalLength;
    }

    public float getPetalWidth() {
        return petalWidth;
    }
}

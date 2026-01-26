package org.brain4j.examples.emotion;

public class PADEmotionalState {
    private final double pleasure;
    private final double arousal;
    private final double dominance;

    public PADEmotionalState(double pleasure, double arousal, double dominance) {
        this.pleasure = clamp(pleasure, -1.0, 1.0);
        this.arousal = clamp(arousal, -1.0, 1.0);
        this.dominance = clamp(dominance, -1.0, 1.0);
    }

    private double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    public double getPleasure() { return pleasure; }
    public double getArousal() { return arousal; }
    public double getDominance() { return dominance; }

    @Override
    public String toString() {
        return String.format("PAD(P: %.2f, A: %.2f, D: %.2f)", pleasure, arousal, dominance);
    }

    public double euclideanDistance(PADEmotionalState other) {
        return Math.sqrt(
                Math.pow(this.pleasure - other.pleasure, 2) +
                        Math.pow(this.arousal - other.arousal, 2) +
                        Math.pow(this.dominance - other.dominance, 2)
        );
    }
}
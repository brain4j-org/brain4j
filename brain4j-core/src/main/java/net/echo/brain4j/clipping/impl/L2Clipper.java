package net.echo.brain4j.clipping.impl;

import net.echo.brain4j.clipping.GradientClipper;
import net.echo.math.tensor.Tensor;

public class L2Clipper implements GradientClipper {

    private final double scale;

    public L2Clipper(double scale) { this.scale = scale; }

    @Override
    public void clip(Tensor grad) {
        double threshold = scale * Math.sqrt(grad.elements());
        clipL2(grad, threshold);
    }

    public double l2Norm(Tensor input) {
        double sumOfSquares = 0.0;

        for (int i = 0; i < input.elements(); i++) {
            sumOfSquares += Math.pow(input.getData()[i], 2);
        }

        return Math.sqrt(sumOfSquares);
    }

    public void clipL2(Tensor input, double threshold) {
        double norm = l2Norm(input);

        if (norm > threshold) {
            double scaleFactor = threshold / norm;
            input.mul(scaleFactor);
        }
    }
}

package net.echo.brain4j.loss;

import net.echo.math.tensor.Tensor;

/**
 * Also known as cost function, measures the model's performance and is minimized during training.
 */
public interface LossFunction {

    double calculate(Tensor actual, Tensor predicted);

    Tensor getDelta(Tensor error, Tensor derivative);
}

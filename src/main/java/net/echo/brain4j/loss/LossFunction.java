package net.echo.brain4j.loss;

import net.echo.brain4j.utils.Vector;

public interface LossFunction {
    
    double calculate(Vector actual, Vector predicted);
}

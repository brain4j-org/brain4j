package org.brain4j.math.operators;

import org.brain4j.math.tensor.Shape;

public interface OperatorSpec {
    int inputCount();
    int outputCount();
    void validateInputShapes(Shape... inputShapes) throws ShapeException;
    Shape inferOutputShape(int outputIndex, Shape... inputShapes);
    boolean isInPlace(int inputIndex, int outputIndex);
}

package net.echo.brain4j.layer.impl.conv;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.StatesCache;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

public class FlattenLayer extends Layer {

    public FlattenLayer() {
    }

    public void preConnect(Layer previous, Layer next) {
        if (!(previous instanceof ConvLayer convLayer)) {
            throw new UnsupportedOperationException("Layer before must be a convolutional layer!");
        }

        int filters = convLayer.getFilters();
        int kernelWidth = convLayer.getOutputWidth();
        int kernelHeight = convLayer.getOutputHeight();

        this.bias = TensorFactory.create(filters * kernelWidth * kernelHeight);
    }

    @Override
    public Tensor forward(StatesCache cache, Layer lastLayer, Tensor input, boolean training) {
        return input.reshape(input.elements());
    }
}

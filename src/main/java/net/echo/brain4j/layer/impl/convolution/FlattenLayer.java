package net.echo.brain4j.layer.impl.convolution;

import com.google.common.base.Preconditions;
import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.convolution.Kernel;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.cache.StatesCache;

public class FlattenLayer extends Layer {

    public FlattenLayer(int input) {
        super(input, Activations.LINEAR);
    }

    @Override
    public Kernel forward(StatesCache cache, Layer lastLayer, Kernel input) {
        super.forward(cache, lastLayer, input);
        return null;
    }

    public void flatten(StatesCache cache, Layer layer, Kernel input) {
        Preconditions.checkNotNull(input, "Last convolutional input is null! Missing an input layer?");

        boolean isConvolutional = layer instanceof ConvLayer || layer instanceof PoolingLayer;

        Preconditions.checkState(isConvolutional, "Flatten layer is not preceded by convolutional layer!");
        Preconditions.checkState(size() == input.size(),
                "Flatten dimension != Conv dimension (" + size() + " != " + input.size() + ")");

        for (int h = 0; h < input.getHeight(); h++) {
            for (int w = 0; w < input.getWidth(); w++) {
                double value = input.getValue(w, h);

                getNeuronAt(h * input.getWidth() + w).setValue(cache, value);
            }
        }
    }
}

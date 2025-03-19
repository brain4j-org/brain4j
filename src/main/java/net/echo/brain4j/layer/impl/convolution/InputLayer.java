package net.echo.brain4j.layer.impl.convolution;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.convolution.Kernel;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.utils.math.vector.Vector;

public class InputLayer extends Layer<Vector, Vector> {

    private final int width;
    private final int height;

    public InputLayer(int width, int height) {
        super(width * height, Activations.LINEAR);
        this.width = width;
        this.height = height;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    @Override
    public boolean isConvolutional() {
        return true;
    }

    @Override
    public int getTotalNeurons() {
        return width * height;
    }

    public Kernel getImage(StatesCache cache) {
        Kernel result = new Kernel(width, height);

        for (int x = 0; x < width; x++) {
            for (int h = 0; h < height; h++) {
                Neuron neuron = neurons.get(h * width + x);

                result.getValues()[h].set(x, neuron.getValue(cache));
            }
        }

        return result;
    }
}

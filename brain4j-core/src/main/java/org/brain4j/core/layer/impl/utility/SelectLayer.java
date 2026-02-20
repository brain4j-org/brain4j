package org.brain4j.core.layer.impl.utility;

import org.brain4j.core.layer.Layer0;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;

/**
 * A utility layer that selects a single tensor from multiple inputs.
 *
 * <p>This layer is useful in models that have multiple input branches
 * and need to select one of the tensors for further processing. It
 * acts as a multiplexer, forwarding only the tensor at the specified
 * index position.
 *
 * @author xEcho1337
 */
public class SelectLayer extends Layer0 {

    private final int index;

    /**
     * Creates a new select layer.
     *
     * @param index the index of the input tensor to forward
     */
    public SelectLayer(int index) {
        this.index = index;
    }

    @Override
    public void connect() {
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        return new Tensor[] { inputs[index] };
    }

    @Override
    public int size() {
        return 0;
    }

    /**
     * Gets the selection index used by this layer.
     * @return the index of the input tensor being selected
     */
    public int index() {
        return index;
    }
}

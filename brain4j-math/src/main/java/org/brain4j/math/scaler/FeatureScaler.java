package org.brain4j.math.scaler;

import org.brain4j.math.commons.JsonAdapter;
import org.brain4j.math.tensor.Tensor;

import java.util.List;

public interface FeatureScaler {

    void fit(List<Tensor> tensors);
    Tensor transform(Tensor tensor);

    default List<Tensor> fitAndTransform(List<Tensor> tensors) {
        fit(tensors);
        return transform(tensors);
    }

    default List<Tensor> transform(List<Tensor> tensors) {
        return tensors.stream().map(this::transform).toList();
    }
}

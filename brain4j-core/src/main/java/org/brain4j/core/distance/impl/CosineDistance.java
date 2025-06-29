package org.brain4j.core.distance.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.Tensors;
import org.brain4j.core.distance.DistanceOperator;

public class CosineDistance implements DistanceOperator {
    @Override
    public Tensor distance(Tensor a, Tensor b) {
        Tensor dot = a.mul(b).sum(1, true);

        Tensor normA = a.pow(2).sum(1, true).sqrt();
        Tensor normB = b.pow(2).sum(1, true).sqrt();

        Tensor cosineSim = dot.div(normA.mul(normB).add(1e-8));

        return Tensors.ones(cosineSim.shape()).sub(cosineSim);
    }
}

package org.brain4j.core.loss.impl;

import org.brain4j.core.loss.LossFunction;
import org.brain4j.math.tensor.Tensor;

public class CosineEmbeddingLoss implements LossFunction {

    @Override
    public double calculate(Tensor expected, Tensor predicted) {
        int batchSize = expected.shape()[0];
        int dim = expected.shape()[1];

        double totalLoss = 0.0;

        for (int batch = 0; batch < batchSize; batch++) {
            double dot = 0;
            double normE = 0;
            double normP = 0;

            for (int i = 0; i < dim; i++) {
                double e = expected.get(batch, i);
                double p = predicted.get(batch, i);

                dot += e * p;
                normE += e * e;
                normP += p * p;
            }

            normE = Math.sqrt(normE);
            normP = Math.sqrt(normP);

            double cosineSim = dot / (normE * normP + 1e-8);

            // Loss = 1 - cosine similarity
            double loss = 1.0 - cosineSim;
            totalLoss += loss;
        }

        return totalLoss / batchSize;
    }

    @Override
    public Tensor getDelta(Tensor error, Tensor derivative) {
        return error.mul(derivative);
    }
}

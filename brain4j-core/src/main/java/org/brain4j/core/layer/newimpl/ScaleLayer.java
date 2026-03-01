package org.brain4j.core.layer.newimpl;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.scaler.FeatureScaler;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.AutogradContext;

import java.util.List;
import java.util.Set;
import java.util.random.RandomGenerator;

public class ScaleLayer extends Layer {

    private final FeatureScaler scaler;
    private Set<Integer> enabledInputs;

    public ScaleLayer(FeatureScaler scaler) {
        this.scaler = scaler;
    }

    public ScaleLayer(FeatureScaler scaler, Set<Integer> enabledInputs) {
        this.scaler = scaler;
        this.enabledInputs = enabledInputs;
    }

    @Override
    public void build(List<Shape> inputShapes) {
    }

    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
    }

    @Override
    public List<Shape> inferOutputShapes(List<Shape> inputShapes) {
        return inputShapes;
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        boolean scaleAll = enabledInputs == null;

        if (!scaleAll) {
            for (int i : enabledInputs) {
                if (i < 0 || i >= inputs.length) {
                    throw Commons.illegalState("Enabled input index out of range: %s", i);
                }
            }
        }

        Tensor[] outputs = new Tensor[inputs.length];

        for (int i = 0; i < outputs.length; i++) {
            Tensor input = inputs[i];

            if (!scaleAll && !enabledInputs.contains(i)) {
                outputs[i] = input;
                continue;
            }

            Tensor result = scaler.transform(input);

            AutogradContext context = input.getAutogradContext();
            result.setAutogradContext(context);

            outputs[i] = result;
        }

        return outputs;
    }

    @Override
    public Layer copy() {
        return new ScaleLayer(scaler, enabledInputs);
    }

    public FeatureScaler scaler() {
        return scaler;
    }

    public Set<Integer> enabledInputs() {
        return enabledInputs;
    }
}

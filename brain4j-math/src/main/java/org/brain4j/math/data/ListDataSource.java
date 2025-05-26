package org.brain4j.math.data;

import org.brain4j.math.Pair;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Consumer;

public class ListDataSource implements Cloneable {

    protected final List<Sample> samples;
    protected final List<Tensor> batchedInputs;
    protected final List<Tensor> batchedLabels;
    protected final int batchSize;
    protected final int batches;
    protected int cursor;

    public ListDataSource(List<Sample> samples, boolean shuffle, int batchSize) {
        this.samples = samples;
        this.batchedInputs = new ArrayList<>();
        this.batchedLabels = new ArrayList<>();
        this.batches = samples.size() / batchSize;
        this.batchSize = batchSize;

        if (shuffle) {
            Collections.shuffle(this.samples);
        }

        computeBatches();
    }

    public List<Sample> getSamples() {
        return samples;
    }

    public boolean hasNext() {
        return cursor < batches;
    }

    public void reset() {
        cursor = 0;
    }

    public ListDataSource normalize() {
        List<Tensor> inputs = new ArrayList<>();
        List<Tensor> labels = new ArrayList<>();

        for (Sample sample : samples) {
            inputs.add(sample.input());
            labels.add(sample.label());
        }

        Tensor first = inputs.getFirst();
        int features = first.elements();

        float[] means = new float[features];
        float[] stds = new float[features];

        for (int i = 0; i < features; i++) {
            double mean = 0;
            double std = 0;

            for (Tensor input : inputs) {
                mean += input.get(i);
                std += input.get(i) * input.get(i);
            }

            mean /= inputs.size();
            std = Math.sqrt(std / inputs.size() - mean * mean);

            means[i] = (float) mean;
            stds[i] = (float) Math.max(std, 1e-8);
        }

        Tensor mean = Tensors.vector(means);
        Tensor std = Tensors.vector(stds);

        for (Tensor input : inputs) {
            input.sub(mean).div(std);
        }

        samples.clear();

        for (int i = 0; i < inputs.size(); i++) {
            Tensor input = inputs.get(i);
            Tensor label = labels.get(i);
            samples.add(new Sample(input, label));
        }

        batchedInputs.clear();
        batchedLabels.clear();

        computeBatches();

        return this;
    }

    public void propagate(Consumer<Pair<Tensor, Tensor>> task) {
        reset();

        while (hasNext()) {
            task.accept(nextBatch());
        }
    }

    public Pair<Tensor, Tensor> nextBatch() {
        if (!hasNext()) return null;

        Tensor input = batchedInputs.get(cursor);
        Tensor label = batchedLabels.get(cursor);

        cursor++;
        return new Pair<>(input, label);
    }

    public int size() {
        return samples.size();
    }

    private void computeBatches() {
        int size = size();
        int index = 0;

        while (index < size) {
            int end = Math.min(index + batchSize, size);
            List<Sample> subSet = samples.subList(index, end);

            List<Tensor> inputs = new ArrayList<>();
            List<Tensor> labels = new ArrayList<>();

            for (Sample sample : subSet) {
                inputs.add(sample.input());
                labels.add(sample.label());
            }

            Tensor mergedInput = Tensors.mergeTensors(inputs);
            Tensor mergedLabels = Tensors.mergeTensors(labels);

            batchedInputs.add(mergedInput);
            batchedLabels.add(mergedLabels);

            index += batchSize;
        }
    }

    @Override
    public ListDataSource clone() {
        return new ListDataSource(samples, false, batchSize);
    }
}

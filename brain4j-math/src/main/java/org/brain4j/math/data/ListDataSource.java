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

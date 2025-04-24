package net.echo.math.data;

import net.echo.math.Pair;
import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.Tensors;

import java.util.ArrayList;
import java.util.List;

public class ListDataSource {

    private final List<Sample> samples;
    private final List<Tensor> batchedInputs;
    private final List<Tensor> batchedLabels;
    private final int batches;
    private int cursor;

    public ListDataSource(List<Sample> samples, int batches) {
        this.samples = samples;
        this.batchedInputs = new ArrayList<>();
        this.batchedLabels = new ArrayList<>();
        this.batches = batches;

        if (samples.size() % batches != 0) {
            throw new IllegalArgumentException("The number of samples must be a multiple of the number of batches.");
        }

        computeBatches();
    }

    public List<Sample> getSamples() {
        return samples;
    }

    public boolean hasNext() {
        return cursor < batches;
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
        int batchSize = size / batches;

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
}

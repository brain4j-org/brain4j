package org.brain4j.math.data;

import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Batch;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**
 * A data source implementation that manages a list of samples for training or evaluation
 * in machine learning contexts. It supports batching, optional shuffling, normalization,
 * and iteration over samples or batches.
 * <p>
 * This class partitions the underlying dataset into batches of fixed size,
 * provides functionality to iterate through batches sequentially,
 * and supports normalization of input features across the entire dataset.
 * <p>
 * It also supports cloning to create deep copies of the data source, preserving the sample data integrity.
 * <p>
 * Typical usage involves constructing the data source with a list of {@link Sample} objects,
 * optionally shuffling them, then iterating through batches during training.
 *
 * <p><b>Thread Safety:</b> This class is not thread-safe. Synchronization is required if accessed concurrently.
 *
 * @author xEcho1337
 * @author Adversing
 */
public class ListDataSource implements Cloneable, Iterable<Sample> {

    protected final List<Tensor[]> batchedInputs;
    protected final List<Tensor[]> batchedLabels;
    protected final int batches;

    protected List<Sample> samples;
    protected Device device;
    protected int cursor;
    protected int batchSize;

    /**
     * Constructs a new ListDataSource from a given list of samples.
     * Optionally shuffles the samples and partitions them into batches of the specified size.
     *
     * @param samples the list of samples to use as the dataset
     * @param shuffle if true, the samples list will be shuffled before batching
     * @param batchSize the size of each batch for iteration
     */
    public ListDataSource(List<Sample> samples, boolean shuffle, int batchSize) {
        this.samples = samples;
        this.batchedInputs = new ArrayList<>();
        this.batchedLabels = new ArrayList<>();
        this.batches = (samples.size() + batchSize - 1) / batchSize;
        this.batchSize = batchSize;

        if (shuffle) {
            Collections.shuffle(this.samples);
        }

        computeBatches();
    }
    
    public ListDataSource(Device device, List<Sample> samples, boolean shuffle, int batchSize) {
        this.device = device;
        this.samples = samples;
        this.batchedInputs = new ArrayList<>();
        this.batchedLabels = new ArrayList<>();
        this.batches = (samples.size() + batchSize - 1) / batchSize;
        this.batchSize = batchSize;

        if (shuffle) {
            Collections.shuffle(this.samples);
        }

        computeBatches();
    }
    
    /**
     * Copy constructor.
     */
    private ListDataSource(List<Sample> samples, Device device, int batches, int cursor, int batchSize) {
        this.device = device;
        this.samples = samples;
        this.batchedInputs = new ArrayList<>();
        this.batchedLabels = new ArrayList<>();
        this.cursor = cursor;
        this.batches = batches;
        this.batchSize = batchSize;
        
        computeBatches();
    }

    /**
     * Returns true if there are remaining batches to iterate over.
     * @return true if more batches are available, false otherwise
     */
    public boolean hasNext() {
        return cursor < batches;
    }

    /**
     * Resets the batch iteration cursor to the beginning.
     */
    public void reset() {
        cursor = 0;
    }

    /**
     * Normalizes the input features of all samples by applying the z-score normalization.
     * This operation modifies the samples in place and recomputes batches accordingly.
     *
     * @return this ListDataSource instance after normalization
     */
    public ListDataSource normalize() {
        if (samples.isEmpty()) return this;

        int numInputs = samples.getFirst().inputs().length;
        List<List<Tensor>> inputStreams = new ArrayList<>(numInputs);
        
        for (int i = 0; i < numInputs; i++) {
            inputStreams.add(new ArrayList<>());
        }

        for (Sample sample : samples) {
            Tensor[] inputs = sample.inputs();
            
            for (int i = 0; i < inputs.length; i++) {
                inputStreams.get(i).add(inputs[i]);
            }
        }

        for (int i = 0; i < numInputs; i++) {
            List<Tensor> tensors = inputStreams.get(i);
            int features = tensors.getFirst().elements();

            float[] means = new float[features];
            float[] stds = new float[features];

            for (int f = 0; f < features; f++) {
                double mean = 0;
                double std = 0;
                
                for (Tensor tensor : tensors) {
                    float value = tensor.get(f);
                    mean += value;
                    std += value * value;
                }

                mean /= tensors.size();
                std = Math.sqrt(std / tensors.size() - mean * mean);

                means[f] = (float) mean;
                stds[f] = (float) Math.max(std, 1e-8);
            }

            Tensor meanTensor = Tensors.vector(means);
            Tensor stdTensor = Tensors.vector(stds);

            for (Tensor tensor : tensors) {
                tensor.sub(meanTensor).div(stdTensor);
            }
        }

        batchedInputs.clear();
        batchedLabels.clear();

        computeBatches();

        return this;
    }

    /**
     * Retrieves the next batch of data (input and label tensors) and advances the cursor.
     * @return a Pair containing input tensor and label tensor for the next batch,
     *         or null if no more batches are available
     */
    public Batch nextBatch() {
        if (!hasNext()) return null;

        Tensor[] input = batchedInputs.get(cursor);
        Tensor[] label = batchedLabels.get(cursor);

        cursor++;

        return new Batch(input, label);
    }

    /**
     * Computes batched inputs and labels by partitioning the samples list according
     * to the batch size. Merges tensors within each batch for efficient processing.
     * This method is called during construction and after normalization.
     */
    private void computeBatches() {
        int size = getSize();
        int index = 0;

        while (index < size) {
            int end = Math.min(index + batchSize, size);
            Batch batch = createBatch(index, end);

            batchedInputs.add(batch.first());
            batchedLabels.add(batch.second());

            index += batchSize;
        }
    }

    private Batch createBatch(int start, int end) {
        List<Sample> subSet = samples.subList(start, end);

        Sample first = subSet.getFirst();

        int inputCount = first.inputs().length;
        int labelCount = first.labels().length;

        List<List<Tensor>> mergedInputs = new ArrayList<>(inputCount);
        List<List<Tensor>> mergedLabels = new ArrayList<>();

        for (int i = 0; i < inputCount; i++) {
            mergedInputs.add(new ArrayList<>());
        }

        for (int i = 0; i < labelCount; i++) {
            mergedLabels.add(new ArrayList<>());
        }

        for (Sample sample : subSet) {
            Tensor[] inputs = sample.inputs();
            Tensor[] labels = sample.labels();


            for (int i = 0; i < inputs.length; i++) {
                mergedInputs.get(i).add(inputs[i]);
            }

            for (int i = 0; i < labels.length; i++) {
                mergedLabels.get(i).add(labels[i]);
            }
        }

        Tensor[] batchedInputTensors = new Tensor[inputCount];

        for (int i = 0; i < inputCount; i++) {
            Tensor merged = Tensors.mergeTensors(mergedInputs.get(i));
            batchedInputTensors[i] = device == null ? merged : GpuTensor.persistent(merged, device);
        }

        Tensor[] batchedLabelTensors = new Tensor[labelCount];

        for (int i = 0; i < labelCount; i++) {
            Tensor merged = Tensors.mergeTensors(mergedLabels.get(i));
            batchedLabelTensors[i] = device == null ? merged : GpuTensor.persistent(merged, device);
        }

        return new Batch(batchedInputTensors, batchedLabelTensors);
    }

    @Override
    public ListDataSource clone() {
        try {
            ListDataSource clone = (ListDataSource) super.clone();

            clone.samples = new ArrayList<>();
            clone.batchSize = batchSize;
            
            for (Sample sample : samples) {
                clone.samples.add(sample.copy());
            }
            
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }
    
    public ListDataSource to(Device device) {
        return new ListDataSource(samples, device, batches, cursor, batchSize);
    }
    
    /**
     * Returns the total number of samples in the data source.
     * @return number of samples
     */
    public int getSize() {
        return samples.size();
    }

    /**
     * Returns the underlying list of samples.
     * @return the samples list
     */
    public List<Sample> getSamples() {
        return samples;
    }

    /**
     * Returns the list of batched input tensors.
     * @return list of input batches
     */
    public List<Tensor[]> getBatchedInputs() {
        return batchedInputs;
    }

    /**
     * Returns the list of batched label tensors.
     * @return list of label batches
     */
    public List<Tensor[]> getBatchedLabels() {
        return batchedLabels;
    }

    /**
     * Returns the configured batch size.
     * @return batch size
     */
    public int getBatchSize() {
        return batchSize;
    }

    /**
     * Returns the total number of batches.
     * @return number of batches
     */
    public int getBatches() {
        return batches;
    }

    /**
     * Returns the current batch cursor index.
     * @return current cursor position
     */
    public int getCursor() {
        return cursor;
    }

    /**
     * Returns an iterator over the individual samples in the data source.
     * @return an iterator of {@link Sample}
     */
    @Override
    public Iterator<Sample> iterator() {
        return samples.iterator();
    }

    public Iterable<Batch> batchIterator() {
        List<Batch> result = new ArrayList<>();
        int cursor = 0;

        while (cursor < batches) {
            Tensor[] input = batchedInputs.get(cursor);
            Tensor[] label = batchedLabels.get(cursor);

            cursor++;

            result.add(new Batch(input, label));
        }

        return result;
    }
}

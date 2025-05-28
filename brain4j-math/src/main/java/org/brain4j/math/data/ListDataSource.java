package org.brain4j.math.data;

import org.brain4j.datasets.core.dataset.Dataset;
import org.brain4j.datasets.core.dataset.Dataset.DatasetFile;
import org.brain4j.math.Pair;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;

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
 * @version 3.0
 */
public class ListDataSource implements Cloneable, Iterable<Sample> {

    protected final List<Sample> samples;
    protected final List<Tensor> batchedInputs;
    protected final List<Tensor> batchedLabels;
    protected final int batchSize;
    protected final int batches;
    protected int cursor;

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

    /**
     * Creates a ListDataSource from a {@link Dataset} object.
     * 
     * @param dataset the dataset to load data from
     * @param inputFeatures a function that converts a line of data to input features tensor
     * @param outputLabels a function that converts a line of data to output labels tensor
     * @param shuffle whether to shuffle the data
     * @param batchSize the size of each batch
     * @param fileFormat the format of files to use (e.g., "csv", "json")
     * @return a new ListDataSource containing the loaded data
     * @throws IOException if an error occurs while reading the dataset files
     */
    public static ListDataSource fromDataset(
            Dataset dataset,
            Function<String, Tensor> inputFeatures,
            Function<String, Tensor> outputLabels,
            boolean shuffle,
            int batchSize,
            String fileFormat
    ) throws IOException {
        
        List<Sample> samples = new ArrayList<>();
        List<DatasetFile> dataFiles = dataset.getFilesByFormat(fileFormat);
        
        for (DatasetFile file : dataFiles) {
            Path filePath = file.path();
            try (BufferedReader reader = Files.newBufferedReader(filePath)) {
                String line;
                while ((line = reader.readLine()) != null) {
                    Tensor input = inputFeatures.apply(line);
                    Tensor label = outputLabels.apply(line);
                    samples.add(new Sample(input, label));
                }
            }
        }
        
        return new ListDataSource(samples, shuffle, batchSize);
    }
    
    /**
     * Creates a ListDataSource from a {@link Dataset} object with a custom parser.
     * 
     * @param dataset the dataset to load data from
     * @param parser a function that converts a line of data to a Sample
     * @param shuffle whether to shuffle the data
     * @param batchSize the size of each batch
     * @param fileFormat the format of files to use (e.g., "csv", "json")
     * @return a new ListDataSource containing the loaded data
     * @throws IOException if an error occurs while reading the dataset files
     */
    public static ListDataSource fromDataset(
            Dataset dataset,
            Function<String, Sample> parser,
            boolean shuffle,
            int batchSize,
            String fileFormat
    ) throws IOException {
        
        List<Sample> samples = new ArrayList<>();
        List<DatasetFile> dataFiles = dataset.getFilesByFormat(fileFormat);
        
        for (DatasetFile file : dataFiles) {
            Path filePath = file.path();
            try (BufferedReader reader = Files.newBufferedReader(filePath)) {
                String line;
                while ((line = reader.readLine()) != null) {
                    Sample sample = parser.apply(line);
                    samples.add(sample);
                }
            }
        }
        
        return new ListDataSource(samples, shuffle, batchSize);
    }
    
    /**
     * Creates a ListDataSource from a {@link Dataset} object with a split function.
     * 
     * @param dataset the dataset to load data from
     * @param lineSplitter a function that splits a line into input and label tensors
     * @param shuffle whether to shuffle the data
     * @param batchSize the size of each batch
     * @param fileFormat the format of files to use (e.g., "csv", "json")
     * @return a new ListDataSource containing the loaded data
     * @throws IOException if an error occurs while reading the dataset files
     */
    public static ListDataSource fromDataset(
            Dataset dataset,
            BiFunction<String, Integer, Pair<Tensor, Tensor>> lineSplitter,
            boolean shuffle,
            int batchSize,
            String fileFormat
    ) throws IOException {
        
        List<Sample> samples = new ArrayList<>();
        List<DatasetFile> dataFiles = dataset.getFilesByFormat(fileFormat);
        
        for (DatasetFile file : dataFiles) {
            Path filePath = file.path();
            try (BufferedReader reader = Files.newBufferedReader(filePath)) {
                String line;
                int lineNum = 0;
                while ((line = reader.readLine()) != null) {
                    Pair<Tensor, Tensor> pair = lineSplitter.apply(line, lineNum++);
                    samples.add(new Sample(pair.first(), pair.second()));
                }
            }
        }
        
        return new ListDataSource(samples, shuffle, batchSize);
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

    /**
     * Performs the specified task on every batch sequentially.
     * @param task a Consumer that accepts a pair of input and label tensors representing a batch
     */
    public void accept(Consumer<Pair<Tensor, Tensor>> task) {
        reset();

        while (hasNext()) {
            task.accept(nextBatch());
        }
    }
    /**
     * Retrieves the next batch of data (input and label tensors) and advances the cursor.
     * @return a Pair containing input tensor and label tensor for the next batch,
     *         or null if no more batches are available
     */
    public Pair<Tensor, Tensor> nextBatch() {
        if (!hasNext()) return null;

        Tensor input = batchedInputs.get(cursor);
        Tensor label = batchedLabels.get(cursor);

        cursor++;
        return new Pair<>(input, label);
    }

    /**
     * Computes batched inputs and labels by partitioning the samples list according
     * to the batch size. Merges tensors within each batch for efficient processing.
     * This method is called during construction and after normalization.
     */
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
        List<Sample> copied = samples.stream()
                .map(s -> new Sample(s.input().clone(), s.label().clone()))
                .collect(Collectors.toList());
        return new ListDataSource(copied, false, batchSize);
    }

    /**
     * Returns the total number of samples in the data source.
     * @return number of samples
     */
    public int size() {
        return samples.size();
    }

    /**
     * Returns the underlying list of samples.
     * @return the samples list
     */
    public List<Sample> samples() {
        return samples;
    }

    /**
     * Returns the list of batched input tensors.
     * @return list of input batches
     */
    public List<Tensor> batchedInputs() {
        return batchedInputs;
    }

    /**
     * Returns the list of batched label tensors.
     * @return list of label batches
     */
    public List<Tensor> batchedLabels() {
        return batchedLabels;
    }

    /**
     * Returns the configured batch size.
     * @return batch size
     */
    public int batchSize() {
        return batchSize;
    }

    /**
     * Returns the total number of batches.
     * @return number of batches
     */
    public int batches() {
        return batches;
    }

    /**
     * Returns the current batch cursor index.
     * @return current cursor position
     */
    public int cursor() {
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
}

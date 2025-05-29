package org.brain4j.math.data;

import org.brain4j.datasets.core.dataset.Dataset;
import org.brain4j.math.LineSplitting;
import org.brain4j.math.Pair;
import org.brain4j.math.tensor.Tensor;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;
import java.util.function.Function;

public class AsyncDataSource extends ListDataSource {

    public static final int PROCESSORS = Runtime.getRuntime().availableProcessors();
    public static final ExecutorService EXECUTOR = Executors.newFixedThreadPool(PROCESSORS);

    public AsyncDataSource(List<Sample> samples, boolean shuffle, int batchSize) {
        super(samples, shuffle, batchSize);
    }

    /**
     * Creates an AsyncDataSource from a {@link Dataset} object.
     * 
     * @param dataset the dataset to load data from
     * @param inputFeatures a function that converts a line of data to input features tensor
     * @param outputLabels a function that converts a line of data to output labels tensor
     * @param shuffle whether to shuffle the data
     * @param batchSize the size of each batch
     * @param fileFormat the format of files to use (e.g., "csv", "json")
     * @return a new AsyncDataSource containing the loaded data
     * @throws IOException if an error occurs while reading the dataset files
     */
    public static AsyncDataSource fromDataset(
            Dataset dataset,
            Function<String, Tensor> inputFeatures,
            Function<String, Tensor> outputLabels,
            boolean shuffle,
            int batchSize,
            String fileFormat
    ) throws IOException {
        ListDataSource dataSource = ListDataSource.fromDataset(
                dataset, inputFeatures, outputLabels, shuffle, batchSize, fileFormat);
        
        return new AsyncDataSource(dataSource.samples(), shuffle, batchSize);
    }
    
    /**
     * Creates an AsyncDataSource from a {@link Dataset} object with a custom parser.
     * 
     * @param dataset the dataset to load data from
     * @param parser a function that converts a line of data to a Sample
     * @param shuffle whether to shuffle the data
     * @param batchSize the size of each batch
     * @param fileFormat the format of files to use (e.g., "csv", "json")
     * @return a new AsyncDataSource containing the loaded data
     * @throws IOException if an error occurs while reading the dataset files
     */
    public static AsyncDataSource fromDataset(
            Dataset dataset,
            Function<String, Sample> parser,
            boolean shuffle,
            int batchSize,
            String fileFormat
    ) throws IOException {
        
        ListDataSource dataSource = ListDataSource.fromDataset(
                dataset, parser, shuffle, batchSize, fileFormat);
        
        return new AsyncDataSource(dataSource.samples(), shuffle, batchSize);
    }
    
    /**
     * Creates an AsyncDataSource from a {@link Dataset} object with a split function.
     * 
     * @param dataset the dataset to load data from
     * @param lineSplitter a function that splits a line into input and label tensors
     * @param shuffle whether to shuffle the data
     * @param batchSize the size of each batch
     * @param fileFormat the format of files to use (e.g., "csv", "json")
     * @return a new AsyncDataSource containing the loaded data
     * @throws IOException if an error occurs while reading the dataset files
     */
    public static AsyncDataSource fromDataset(
            Dataset dataset,
            LineSplitting lineSplitter,
            boolean shuffle,
            int batchSize,
            String fileFormat
    ) throws IOException {
        ListDataSource dataSource = ListDataSource.fromDataset(
                dataset, lineSplitter, shuffle, batchSize, fileFormat);
        
        return new AsyncDataSource(dataSource.samples(), shuffle, batchSize);
    }

    public void accept(Consumer<Pair<Tensor, Tensor>> task) {
        reset();

        List<CompletableFuture<Void>> futures = new ArrayList<>();

        while (hasNext()) {
            Pair<Tensor, Tensor> partition = nextBatch();
            futures.add(CompletableFuture.runAsync(() -> task.accept(partition), EXECUTOR));
        }

        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();
    }

    @Override
    public AsyncDataSource clone() {
        return new AsyncDataSource(samples, false, batchSize);
    }
}

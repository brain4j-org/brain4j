package org.brain4j.datasets;

import ar.com.hjg.pngj.ImageLineInt;
import ar.com.hjg.pngj.PngReader;
import org.apache.parquet.example.data.Group;
import org.brain4j.datasets.api.Dataset;
import org.brain4j.datasets.core.dataset.HFDataset;
import org.brain4j.datasets.core.dataset.DatasetFile;
import org.brain4j.datasets.core.loader.DatasetLoader;
import org.brain4j.datasets.core.loader.config.LoadConfig;
import org.brain4j.datasets.download.callback.ProgressCallback;
import org.brain4j.datasets.format.FileFormat;
import org.brain4j.datasets.format.RecordParser;
import org.brain4j.datasets.format.impl.ParquetFormat;
import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Batch;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.Sample;
import org.brain4j.math.tensor.Tensor;

import javax.xml.crypto.Data;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * High-level helpers for loading and converting datasets into framework-friendly
 * data sources.
 *
 * <p>This class contains convenience methods to obtain commonly used datasets (for
 * example MNIST) and utility methods to transform dataset files into the
 * library's {@link ListDataSource} implementations.
 */
public final class Datasets {

    private Datasets() {
    }
    
    /**
     * Convenience loader for the MNIST dataset.
     *
     * @param shuffle whether to shuffle the dataset
     * @param batchSize the batch size to use in the returned data source
     * @return a {@link ListDataSource} yielding MNIST (input, label) pairs
     */
    public static Dataset mnist(boolean shuffle, int batchSize, double trainTestSplit) {
        if (trainTestSplit < 0 || trainTestSplit > 1) {
            throw new IllegalArgumentException("trainTestSplit must be between 0 and 1");
        }

        try {
            HFDataset dataset = loadDataset("ylecun/mnist");
            RecordParser<Group> parser = (record, index) -> {
                int label = (int) record.getLong("label", 0);
                Group imageGroup = record.getGroup("image", 0);
                byte[] pngBytes = imageGroup.getBinary("bytes", 0).getBytes();
                
                PngReader reader = new PngReader(new ByteArrayInputStream(pngBytes));
                
                int w = reader.imgInfo.cols;
                int h = reader.imgInfo.rows;
                
                float[] normalized = new float[w * h];
                
                for (int row = 0; row < h; row++) {
                    ImageLineInt line = (ImageLineInt) reader.readRow(row);
                    int[] scanline = line.getScanline();
                    
                    for (int col = 0; col < w; col++) {
                        int gray = scanline[col] & 0xFF;
                        normalized[row * w + col] = gray / 255.0f;
                    }
                }
                
                reader.end();
                
                Tensor input = Tensors.vector(normalized);
                Tensor output = Tensors.zeros(10);
                
                output.set(1, label);
                return new Batch(new Tensor[] { input }, new Tensor[] { output });
            };
            
            return createDataSet(dataset, new ParquetFormat(), parser, shuffle, batchSize, trainTestSplit);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
    
    /**
     * Creates a ListDataSource from a {@link HFDataset} object with a custom parser.
     *
     * @param dataset the dataset to convert
     * @param parser a function that parses a sample
     * @param format the format of files to use
     * @param shuffle whether to shuffle the data
     * @param batchSize the size of each batch
     * @return a new ListDataSource containing the loaded data
     * @throws IOException if an error occurs while reading the dataset files
     */
    public static <T> Dataset createDataSet(
        HFDataset dataset,
        FileFormat<T> format,
        RecordParser<T> parser,
        boolean shuffle,
        int batchSize,
        double trainTestSplit
    ) throws IOException {
        if (trainTestSplit < 0 || trainTestSplit > 1) {
            throw new IllegalArgumentException("trainTestSplit must be between 0 and 1");
        }

        List<Sample> samples = new ArrayList<>();
        List<DatasetFile> dataFiles = dataset.filesByFormat(format.format());
        
        for (DatasetFile file : dataFiles) {
            for (T record : format.read(file.path().toFile())) {
                try {
                    Batch pair = parser.parse(record, samples.size());
                    
                    if (pair == null) continue;
                    
                    samples.add(new Sample(pair.first(), pair.second()));
                } catch (Exception e) {
                    e.printStackTrace(System.err);
                    break;
                }
            }
        }

        int division = (int) (samples.size() * trainTestSplit);

        var firstList = samples.subList(0, division);
        var secondList = samples.subList(division, samples.size());

        var firstSource =  new ListDataSource(firstList, shuffle, batchSize);
        var secondSource =  new ListDataSource(secondList, shuffle, batchSize);

        return new Dataset(trainTestSplit, firstSource, secondSource);
    }

    /**
     * Load a dataset by id using the default loader configuration.
     *
     * @param datasetId the dataset identifier (for example "mnist")
     * @return a {@link HFDataset} handle
     * @throws Exception if loading fails
     */

    public static HFDataset loadDataset(String datasetId) throws Exception {
        try (DatasetLoader loader = new DatasetLoader()) {
            return loader.loadDataset(datasetId);
        } catch (Exception e) {
            throw new Exception("Failed to load dataset: " + datasetId, e);
        }
    }

    public static HFDataset loadDataset(String datasetId, LoadConfig config) throws Exception {
        try (DatasetLoader loader = new DatasetLoader()) {
            return loader.loadDataset(datasetId, config);
        } catch (Exception e) {
            throw new Exception("Failed to load dataset: " + datasetId, e);
        }
    }

    public static HFDataset loadDataset(String datasetId, ProgressCallback progressCallback) throws Exception {
        try (DatasetLoader loader = new DatasetLoader(progressCallback)) {
            return loader.loadDataset(datasetId);
        } catch (Exception e) {
            throw new Exception("Failed to load dataset: " + datasetId, e);
        }
    }

    public static CompletableFuture<HFDataset> loadDatasetAsync(String datasetId) {
        try (DatasetLoader loader = new DatasetLoader()) {
            return loader.loadDatasetAsync(datasetId);
        } catch (Exception e) {
            return CompletableFuture.failedFuture(e);
        }
    }

    public static CompletableFuture<HFDataset> loadDatasetAsync(String datasetId, LoadConfig config) {
        try (DatasetLoader loader = new DatasetLoader()) {
            return loader.loadDatasetAsync(datasetId, config);
        } catch (Exception e) {
            return CompletableFuture.failedFuture(e);
        }
    }
}
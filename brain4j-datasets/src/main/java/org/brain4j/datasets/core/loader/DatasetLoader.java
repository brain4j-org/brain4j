package org.brain4j.datasets.core.loader;

import org.brain4j.datasets.api.DatasetInfo;
import org.brain4j.datasets.api.HuggingFaceClient;
import org.brain4j.datasets.cache.manager.CacheManager;
import org.brain4j.datasets.core.dataset.Dataset;
import org.brain4j.datasets.core.dataset.DatasetFile;
import org.brain4j.datasets.core.loader.config.LoadConfig;
import org.brain4j.datasets.download.callback.ProgressCallback;
import org.brain4j.datasets.download.manager.DownloadManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ForkJoinPool;

public class DatasetLoader implements AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(DatasetLoader.class);

    private final HuggingFaceClient client;
    private final CacheManager cacheManager;
    private final DownloadManager downloadManager;
    
    public DatasetLoader() {
        this.client = new HuggingFaceClient();
        this.cacheManager = new CacheManager();
        this.downloadManager = new DownloadManager(client, cacheManager);
    }

    public DatasetLoader(ProgressCallback progressCallback) {
        this.client = new HuggingFaceClient();
        this.cacheManager = new CacheManager();
        this.downloadManager = new DownloadManager(client, cacheManager,
                ForkJoinPool.commonPool(),
                progressCallback);
    }

    public Dataset loadDataset(String datasetId) throws Exception {
        return loadDataset(datasetId, LoadConfig.defaultConfig());
    }

    public Dataset loadDataset(String datasetId, LoadConfig config) throws Exception {
        logger.info("Loading dataset: {}", datasetId);

        Optional<DatasetInfo> infoOpt = client.getDatasetInfo(datasetId);

        if (infoOpt.isEmpty()) {
            throw new Exception(
                        """
                        Dataset not found: %s
                        This error doesn't necessarily mean that the dataset doesn't exist, it could imply that the requested dataset is private or requires authentication.
                        Unfortunately, this library doesn't support authentication yet.
                        """.formatted(datasetId)
            );
        }

        DatasetInfo info = infoOpt.get();
        logger.debug("Dataset info retrieved for: {} (resolved to: {})", datasetId, info.id());

        String resolvedDatasetId = info.id();

        List<DatasetFile> files = new ArrayList<>();
        List<String> filesToDownload = determineFilesToDownload(info, config);

        logger.debug("Files to download: {}", filesToDownload);

        for (String filename : filesToDownload) {
            try {
                Path filePath = downloadManager.downloadFile(resolvedDatasetId, filename, config.forceDownload());
                long size = Files.size(filePath);
                String format = determineFileFormat(filename);

                files.add(new DatasetFile(filename, filePath, size, format));
                logger.debug("Added file: {} ({} bytes, {})", filename, size, format);
            } catch (IOException e) {
                logger.warn("Failed to process file {}: {}", filename, e.getMessage());
            }
        }

        Map<String, Object> datasetConfig = new HashMap<>();
        datasetConfig.put("split", config.split());
        datasetConfig.put("streaming", config.streaming());

        Dataset dataset = new Dataset(resolvedDatasetId, info, files, datasetConfig);
        logger.info("Successfully loaded dataset: {} with {} files", resolvedDatasetId, files.size());

        return dataset;
    }

    public CompletableFuture<Dataset> loadDatasetAsync(String datasetId) {
        return loadDatasetAsync(datasetId, LoadConfig.defaultConfig());
    }

    public CompletableFuture<Dataset> loadDatasetAsync(String datasetId, LoadConfig config) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return loadDataset(datasetId, config);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
    }

    private List<String> determineFilesToDownload(DatasetInfo info, LoadConfig config) {
        List<String> files = new ArrayList<>();

        String split = config.split();
        if (split == null || "all".equals(split)) {
            info.siblings().stream()
                    .map(DatasetInfo.DatasetFile::filename)
                    .filter(this::isMetaFile)
                    .forEach(files::add);
        } else {
            info.siblings().stream()
                    .map(DatasetInfo.DatasetFile::filename)
                    .filter(filename -> filename.contains(split) && isMetaFile(filename))
                    .forEach(files::add);
        }

        if (config.maxFiles() < files.size()) {
            files = files.subList(0, config.maxFiles());
        }

        return files;
    }

    private boolean isMetaFile(String filename) {
        return !filename.equals(".gitattributes") &&
                !filename.equals("README.md") &&
                !filename.startsWith(".");
    }

    private String determineFileFormat(String filename) {
        String extension = filename.substring(filename.lastIndexOf('.') + 1).toLowerCase();
        return switch (extension) {
            case "json", "jsonl" -> "json"; // TODO: fix for jsonl
            case "csv" -> "csv";
            case "parquet" -> "parquet";
            case "txt" -> "text";
            case "arrow" -> "arrow";
            default -> "unknown";
        };
    }

    @Override
    public void close() throws IOException {
        if (client != null) {
            client.close();
        }
    }
}
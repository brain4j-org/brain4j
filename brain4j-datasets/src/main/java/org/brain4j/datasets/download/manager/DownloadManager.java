package org.brain4j.datasets.download.manager;

import org.brain4j.datasets.api.FileDownloadResponse;
import org.brain4j.datasets.api.HuggingFaceClient;
import org.brain4j.datasets.api.exception.DatasetException;
import org.brain4j.datasets.cache.manager.CacheManager;
import org.brain4j.datasets.download.callback.ProgressCallback;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.ForkJoinPool;

public class DownloadManager {
    private static final Logger logger = LoggerFactory.getLogger(DownloadManager.class);

    private final HuggingFaceClient client;
    private final CacheManager cacheManager;
    private final Executor executor;
    private final ProgressCallback progressCallback;

    public DownloadManager(
            HuggingFaceClient client,
            CacheManager cacheManager
    ) {
        this(client, cacheManager, ForkJoinPool.commonPool(), ProgressCallback.NOOP);
    }

    public DownloadManager(
            HuggingFaceClient client,
            CacheManager cacheManager,
            Executor executor,
            ProgressCallback progressCallback
    ) {
        this.client = client;
        this.cacheManager = cacheManager;
        this.executor = executor;
        this.progressCallback = progressCallback;
    }

    public Path downloadFile(
            String datasetId,
            String filename
    ) throws DatasetException {
        return downloadFile(datasetId, filename, false);
    }

    public Path downloadFile(
            String datasetId,
            String filename,
            boolean force
    ) throws DatasetException {

        Path cachedPath = cacheManager.getCachedFilePath(datasetId, filename);

        if (!force && cacheManager.isCached(datasetId, filename)) {
            logger.debug("Using cached file: {}", cachedPath);
            progressCallback.onProgress(filename, 100.0, "Using cached file");
            return cachedPath;
        }

        logger.info("Downloading {} from dataset {}", filename, datasetId);
        progressCallback.onProgress(filename, 0.0, "Starting download");

        cacheManager.ensureDatasetCacheDirectory(datasetId);
        cacheManager.ensureFileDirectories(cachedPath);

        try (FileDownloadResponse response = client.downloadFile(datasetId, filename)) {
            logger.debug("Saving file to: {}", cachedPath);
            Files.copy(response.inputStream(), cachedPath, StandardCopyOption.REPLACE_EXISTING);

            if (!Files.exists(cachedPath)) {
                throw new DatasetException("File was not created successfully: " + cachedPath);
            }

            long fileSize = Files.size(cachedPath);
            logger.debug("File saved successfully, size: {} bytes", fileSize);

            progressCallback.onProgress(filename, 100.0, "Download complete");
            logger.info("Successfully downloaded: {}", cachedPath);
            return cachedPath;
        } catch (IOException e) {
            try { // this means that if a partially downloaded file exists, there's an attempt to clean it
                Files.deleteIfExists(cachedPath);
            } catch (IOException cleanupException) {
                logger.warn("Failed to clean up partial file: {}", cachedPath, cleanupException);
            }
            throw new DatasetException("Failed to save downloaded file: " + filename, e);
        }
    }

    public CompletableFuture<Path> downloadFileAsync(
            String datasetId,
            String filename
    ) {
        return downloadFileAsync(datasetId, filename, false);
    }

    public CompletableFuture<Path> downloadFileAsync(
            String datasetId,
            String filename,
            boolean force
    ) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return downloadFile(datasetId, filename, force);
            } catch (DatasetException e) {
                throw new RuntimeException(e);
            }
        }, executor);
    }
}
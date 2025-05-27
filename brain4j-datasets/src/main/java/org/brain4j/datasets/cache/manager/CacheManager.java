package org.brain4j.datasets.cache.manager;

import org.brain4j.datasets.cache.strategy.CacheStrategy;
import org.brain4j.datasets.cache.strategy.impl.DefaultCacheStrategy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

public class CacheManager {
    private static final Logger logger = LoggerFactory.getLogger(CacheManager.class);
    private static final String DEFAULT_CACHE_DIR = System.getProperty("user.home") + "/.cache/brain4j/datasets";

    private final Path cacheDirectory;
    private final CacheStrategy cacheStrategy;

    public CacheManager() {
        this(Paths.get(DEFAULT_CACHE_DIR), new DefaultCacheStrategy());
    }

    public CacheManager(Path cacheDirectory, CacheStrategy cacheStrategy) {
        this.cacheDirectory = cacheDirectory;
        this.cacheStrategy = cacheStrategy;
        initializeCacheDirectory();
    }

    public CacheManager(Path cacheDirectory) {
        this(cacheDirectory, new DefaultCacheStrategy());
    }

    public Path getCachedFilePath(String datasetId, String filename) {
        String normalizedFilename = filename.replace('/', java.io.File.separatorChar);

        String hashedPrefix = hashFilename(datasetId + "/" + filename);

        return cacheDirectory.resolve(datasetId).resolve(hashedPrefix + "_" + normalizedFilename);
    }

    public boolean isCached(String datasetId, String filename) {
        Path cachedPath = getCachedFilePath(datasetId, filename);
        return cacheStrategy.isValid(cachedPath);
    }

    // TODO migrate this to void-returning type?
    public Path ensureDatasetCacheDirectory(String datasetId) {
        Path datasetDir = cacheDirectory.resolve(datasetId);
        try {
            Files.createDirectories(datasetDir);
            return datasetDir;
        } catch (IOException e) {
            throw new RuntimeException("Failed to create cache directory: " + datasetDir, e);
        }
    }

    public void ensureFileDirectories(Path filePath) {
        Path parentDir = filePath.getParent();
        if (parentDir != null && !Files.exists(parentDir)) {
            try {
                Files.createDirectories(parentDir);
                logger.debug("Created directories for: {}", parentDir);
            } catch (IOException e) {
                throw new RuntimeException("Failed to create directories for: " + parentDir, e);
            }
        }
    }

    public void clearDatasetCache(String datasetId) {
        Path datasetDir = cacheDirectory.resolve(datasetId);
        cacheStrategy.clear(datasetDir);
    }

    public void clearAllCache() {
        cacheStrategy.clear(cacheDirectory);
    }

    private void initializeCacheDirectory() {
        try {
            Files.createDirectories(cacheDirectory);
            logger.debug("Cache directory initialized: {}", cacheDirectory);
        } catch (IOException e) {
            throw new RuntimeException("Failed to initialize cache directory: " + cacheDirectory, e);
        }
    }

    private String hashFilename(String input) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] hash = md.digest(input.getBytes());
            StringBuilder hexString = new StringBuilder();
            for (byte b : hash) {
                String hex = Integer.toHexString(0xff & b);
                if (hex.length() == 1) {
                    hexString.append('0');
                }
                hexString.append(hex);
            }
            return hexString.substring(0, 8); // only take the first 8 characters, yes, I'm lazy.
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("SHA-256 algorithm not available", e);
        }
    }
}
package org.brain4j.transformers.cache.strategy.impl;

import org.brain4j.transformers.cache.strategy.CacheStrategy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.stream.Stream;

public class DefaultCacheStrategy implements CacheStrategy {

    private static final Logger logger = LoggerFactory.getLogger(DefaultCacheStrategy.class);

    @Override
    public boolean isValid(Path filePath) {
        return Files.exists(filePath) && Files.isRegularFile(filePath);
    }

    @Override
    public void clear(Path path) {
        if (!Files.exists(path)) {
            return;
        }

        try {
            if (!Files.isDirectory(path)) {
                deleteQuietly(path);
                return;
            }

            try (Stream<Path> paths = Files.walk(path)) {
                paths.sorted(Comparator.reverseOrder()).forEach(this::deleteQuietly);
            }
        } catch (IOException e) {
            logger.error("Failed to clear cache at: {}", path, e);
        }
    }

    private void deleteQuietly(Path path) {
        try {
            Files.deleteIfExists(path);
        } catch (IOException e) {
            logger.warn("Failed to delete: {}", path, e);
        }
    }
}
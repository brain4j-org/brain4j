package org.brain4j.datasets.core.dataset;

import org.brain4j.datasets.api.DatasetInfo;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;

public record Dataset(
        String id,
        DatasetInfo info,
        List<DatasetFile> files,
        Map<String, Object> config
) {

    public record DatasetFile(
            String name,
            Path path,
            long size,
            String format
    ) {}


    public Optional<DatasetFile> getFile(String filename) {
        return files.stream()
                .filter(file -> file.name().equals(filename))
                .findFirst();
    }


    public List<DatasetFile> getFilesByFormat(String format) {
        return files.stream()
                .filter(file -> file.format().equalsIgnoreCase(format))
                .toList();
    }

    public long getTotalSize() {
        return files.stream()
                .mapToLong(DatasetFile::size)
                .sum();
    }
}
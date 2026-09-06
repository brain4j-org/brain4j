package org.brain4j.datasets.core.dataset;

import org.brain4j.datasets.api.DatasetInfo;

import java.util.List;
import java.util.Map;
import java.util.Optional;

public record HFDataset(
    String id,
    DatasetInfo info,
    List<DatasetFile> files,
    Map<String, Object> config
) {
    public Optional<DatasetFile> find(String filename) {
        return files.stream().filter(file -> file.name().equals(filename)).findFirst();
    }

    public List<DatasetFile> filesByFormat(String format) {
        return files.stream().filter(file -> file.format().equalsIgnoreCase(format)).toList();
    }

    public long totalSize() {
        return files.stream().mapToLong(DatasetFile::size).sum();
    }
}
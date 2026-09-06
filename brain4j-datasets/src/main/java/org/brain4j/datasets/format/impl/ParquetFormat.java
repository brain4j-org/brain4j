package org.brain4j.datasets.format.impl;

import dev.hardwood.InputFile;
import dev.hardwood.reader.ParquetFileReader;
import dev.hardwood.reader.RowReader;
import org.brain4j.datasets.format.FileFormat;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class ParquetFormat implements FileFormat<RowReader> {

    @Override
    public String format() {
        return "parquet";
    }
    
    @Override
    public Iterable<RowReader> read(File file) throws IOException {
        Path path = file.toPath();
        List<RowReader> result = new ArrayList<>();

        try (ParquetFileReader fileReader = ParquetFileReader.open(InputFile.of(path));
             RowReader rowReader = fileReader.rowReader()) {

            while (rowReader.hasNext()) {
                rowReader.next();
                result.add(rowReader);
            }
        }

        return result;
    }
}

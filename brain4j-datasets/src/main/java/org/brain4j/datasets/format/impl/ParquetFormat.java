package org.brain4j.datasets.format.impl;

import dev.hardwood.InputFile;
import dev.hardwood.reader.ParquetFileReader;
import dev.hardwood.reader.RowReader;
import org.brain4j.datasets.format.FileFormat;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.NoSuchElementException;

public class ParquetFormat implements FileFormat<RowReader> {

    @Override
    public String format() {
        return "parquet";
    }

    @Override
    public Iterable<RowReader> read(File file) throws IOException {
        Path path = file.toPath();

        ParquetFileReader fileReader = ParquetFileReader.open(InputFile.of(path));
        RowReader rowReader = fileReader.rowReader();

        return () -> new Iterator<>() {
            @Override
            public boolean hasNext() {
                boolean hasNext = rowReader.hasNext();

                if (!hasNext) {
                    closeQuietly();
                }

                return hasNext;
            }

            @Override
            public RowReader next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }

                rowReader.next();
                return rowReader;
            }

            private void closeQuietly() {
                try {
                    rowReader.close();
                    fileReader.close();
                } catch (Exception ignored) {
                    // best effort cleanup at exhaustion
                }
            }
        };
    }
}

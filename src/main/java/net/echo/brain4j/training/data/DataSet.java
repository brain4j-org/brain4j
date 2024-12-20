package net.echo.brain4j.training.data;

import net.echo.brain4j.utils.Vector;

import java.util.*;

public class DataSet implements Iterable<DataRow> {

    private final List<DataRow> dataRows;
    private final List<List<DataRow>> partitions = new ArrayList<>();
    private int batches;

    public DataSet(List<DataRow> dataRows) {
        this.dataRows = dataRows;
    }

    public DataSet(DataRow... rows) {
        this.dataRows = new ArrayList<>(Arrays.asList(rows));
    }

    public List<DataRow> getDataRows() {
        return dataRows;
    }

    public List<List<DataRow>> getPartitions() {
        return partitions;
    }

    public void add(DataRow row) {
        dataRows.add(row);
    }

    public void add(Vector input, Vector output) {
        dataRows.add(new DataRow(input, output));
    }

    public int getBatches() {
        return batches;
    }

    public boolean isPartitioned() {
        return !partitions.isEmpty();
    }

    private List<DataRow> subdivide(List<DataRow> rows, double batches, int offset) {
        int start = (int) Math.min(offset * batches, rows.size());
        int stop = (int) Math.min((offset + 1) * batches, rows.size());

        return rows.subList(start, stop);
    }

    public void partition(int batches) {
        this.batches = batches;
        this.partitions.clear();

        int rowsPerBatch = dataRows.size() / batches;

        for (int i = 0; i < batches; i++) {
            this.partitions.add(subdivide(dataRows, rowsPerBatch, i));
        }
    }

    public void partitionWithSize(int batchSize) {
        this.batches = dataRows.size() / batchSize;
        this.partitions.clear();

        for (int i = 0; i < batches; i++) {
            this.partitions.add(subdivide(dataRows, batchSize, i));
        }
    }

    /**
     * Randomly shuffles the dataset, making the training more efficient.
     */
    public void shuffle() {
        Collections.shuffle(dataRows);
    }

    @Override
    public Iterator<DataRow> iterator() {
        return dataRows.iterator();
    }
}

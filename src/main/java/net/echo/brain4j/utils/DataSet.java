package net.echo.brain4j.utils;

import java.util.*;

public class DataSet<T> implements Iterable<T> {

    protected final List<List<T>> partitions;
    protected final List<T> data;
    protected int batches;

    public DataSet(List<T> data) {
        this.partitions = new ArrayList<>();
        this.data = data;
    }

    @SafeVarargs
    public DataSet(T... rows) {
        this.partitions = new ArrayList<>();
        this.data = new ArrayList<>(Arrays.asList(rows));
    }

    public int size() {
        return data.size();
    }

    public int getBatches() {
        return batches;
    }

    public boolean isPartitioned() {
        return !partitions.isEmpty();
    }

    public List<T> getData() {
        return data;
    }

    public List<List<T>> getPartitions() {
        return partitions;
    }

    public void add(T row) {
        data.add(row);
    }

    private List<T> subdivide(List<T> rows, double batches, int offset) {
        int start = (int) Math.min(offset * batches, rows.size());
        int stop = (int) Math.min((offset + 1) * batches, rows.size());

        return rows.subList(start, stop);
    }

    public void partition(int batches) {
        this.batches = batches;
        this.partitions.clear();

        int rowsPerBatch = data.size() / batches;

        for (int i = 0; i < batches; i++) {
            this.partitions.add(subdivide(data, rowsPerBatch, i));
        }
    }

    public void partitionWithSize(int batchSize) {
        this.batches = data.size() / batchSize;
        this.partitions.clear();

        for (int i = 0; i < batches; i++) {
            this.partitions.add(subdivide(data, batchSize, i));
        }
    }

    /**
     * Randomly shuffles the data set, making the training more efficient.
     */
    public void shuffle() {
        Collections.shuffle(data);
    }

    @Override
    public Iterator<T> iterator() {
        return data.iterator();
    }
}

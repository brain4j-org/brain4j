package net.echo.math;

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
        int remainder = data.size() % batches;

        int start = 0;

        for (int i = 0; i < batches; i++) {
            int end = start + rowsPerBatch + (i < remainder ? 1 : 0);
            partitions.add(data.subList(start, end));
            start = end;
        }
    }

    public void partitionWithSize(int batchSize) {
        this.partitions.clear();

        int batches = (int) Math.ceil((double) data.size() / batchSize);
        this.batches = batches;

        int start = 0;

        for (int i = 0; i < batches; i++) {
            int end = start + batchSize;

            if (end > data.size()) {
                end = data.size();
            }

            partitions.add(data.subList(start, end));
            start = end;
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

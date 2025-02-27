package net.echo.brain4j.clustering;

import net.echo.brain4j.utils.DataSet;
import net.echo.brain4j.utils.Vector;

import java.util.List;

public class ClusterData extends DataSet<Vector> {

    public ClusterData(List<Vector> data) {
        super(data);
    }

    public ClusterData(Vector... data) {
        super(data);
    }

    @Override
    public void partition(int batches) {
        throw new UnsupportedOperationException("This method cannot be used on ClusterData");
    }

    @Override
    public void partitionWithSize(int batchSize) {
        throw new UnsupportedOperationException("This method cannot be used on ClusterData");
    }
}

package net.echo.brain4j.clustering;

import net.echo.math4j.DataSet;
import net.echo.math4j.math.tensor.Tensor;

import java.util.List;

public class ClusterData extends DataSet<Tensor> {

    public ClusterData(List<Tensor> data) {
        super(data);
    }

    public ClusterData(Tensor... data) {
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

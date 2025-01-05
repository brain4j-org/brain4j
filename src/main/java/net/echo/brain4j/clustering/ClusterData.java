package net.echo.brain4j.clustering;

import net.echo.brain4j.utils.GenericDataSet;
import net.echo.brain4j.utils.Vector;

import java.util.List;

public class ClusterData extends GenericDataSet<Vector> {

    public ClusterData(List<Vector> data) {
        super(data);
    }

    public ClusterData(Vector... data) {
        super(data);
    }
}

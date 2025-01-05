package net.echo.brain4j.training.data;

import net.echo.brain4j.utils.GenericDataSet;
import net.echo.brain4j.utils.Vector;

import java.util.List;

public class DataSet extends GenericDataSet<DataRow> {

    public DataSet(List<DataRow> data) {
        super(data);
    }

    public DataSet(DataRow... rows) {
        super(rows);
    }

    public void add(Vector input, Vector output) {
        getData().add(new DataRow(input, output));
    }
}

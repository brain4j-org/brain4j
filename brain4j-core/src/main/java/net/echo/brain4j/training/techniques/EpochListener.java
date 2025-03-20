package net.echo.brain4j.training.techniques;

import net.echo.math4j.DataSet;

public class EpochListener<R> extends TrainListener<R> {

    @Override
    public void onEvaluated(DataSet<R> dataSet, int epoch, double loss, long took) {
        System.out.println("Evaluated at epoch #" + epoch + " with loss: " + loss);
    }
}

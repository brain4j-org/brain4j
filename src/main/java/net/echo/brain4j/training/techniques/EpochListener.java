package net.echo.brain4j.training.techniques;

import net.echo.brain4j.utils.DataSet;

public class EpochListener<R> extends TrainListener<R> {

    @Override
    public void onEvaluated(DataSet<R> dataSet, int epoch, double loss, long took) {
        System.out.println("Evaluated at epoch #" + epoch + " with loss: " + loss);
    }

    @Override
    public void onLossIncreased(double loss, double previousLoss) {
        super.onLossIncreased(loss, previousLoss);
        trainer.abort();
    }
}

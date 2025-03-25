package net.echo.brain4j.training.techniques;

import net.echo.brain4j.training.data.DataRow;
import net.echo.math4j.DataSet;

public class EpochListener extends TrainListener {

    @Override
    public void onEvaluated(DataSet<DataRow> dataSet, int epoch, double loss, long took) {
        System.out.println("Evaluated at epoch #" + epoch + " with loss: " + loss);

        if (loss < 0.01) {
            trainer.abort();
        }
    }
}

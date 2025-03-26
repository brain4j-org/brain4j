package cnn;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.conv.ConvLayer;
import net.echo.brain4j.layer.impl.conv.FlattenLayer;
import net.echo.brain4j.layer.impl.conv.InputLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.training.optimizer.impl.Adam;

public class ConvTest {

    public static void main(String[] args) {
        new ConvTest().start();
    }

    private void start() {
        var model = new Sequential(
                new InputLayer(28, 28),
                new ConvLayer(16, 7, 7),
                new ConvLayer(24, 7, 7),
                new FlattenLayer(),
                new DenseLayer(10, Activations.LINEAR)
        );

        model.compile(LossFunctions.CROSS_ENTROPY, new Adam(0.001));

        System.out.println(model.summary());
    }
}

package cnn;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.conv.ConvLayer;
import net.echo.brain4j.layer.impl.conv.FlattenLayer;
import net.echo.brain4j.layer.impl.conv.InputLayer;
import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.training.optimizer.impl.Adam;

public class CNNExample {

    public static void main(String[] args) {
        new CNNExample().start();
    }

    private void start() {
        Sequential model = new Sequential(
                new InputLayer(28, 28),
                new ConvLayer(16, 3, 3),
                new ConvLayer(24, 5, 5),
                new ConvLayer(32, 7, 7),
                new FlattenLayer(),
                new DenseLayer(10, Activations.LINEAR)
        );

        model.compile(Loss.CROSS_ENTROPY, new Adam(0.001));

        System.out.println(model.summary());
    }
}

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.convolution.impl.ConvLayer;
import net.echo.brain4j.convolution.impl.FlattenLayer;
import net.echo.brain4j.convolution.impl.InputLayer;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.training.updater.impl.StochasticUpdater;
import net.echo.brain4j.utils.Vector;

public class ConvExample {

    public static void main(String[] args) {
        ConvExample example = new ConvExample();
        example.start();
    }

    private void start() {
        Model model = getModel();
        Vector input = Vector.random(28 * 28);

        Vector prediction = model.predict(input);

        System.out.println("Input: " + input);
        System.out.println("Prediction: " + prediction);
    }

    private Model getModel() {
        Model model = new Model(
                // Input layer, necessary when using CNNs
                new InputLayer(28, 28),

                // #1 convolutional block
                new ConvLayer(32, 1, 1, Activations.RELU),

                // #2 convolutional block
                new ConvLayer(64, 1, 1, Activations.RELU),

                // #3 convolutional block
                new ConvLayer(128, 1, 1, Activations.RELU),

                // Flattens the feature map to a 1D vector
                new FlattenLayer(784), // You must find the right size by trial and error

                // Classifiers
                new DenseLayer(32, Activations.RELU),
                new DenseLayer(10, Activations.SOFTMAX)
        );

        return model.compile(WeightInit.HE, LossFunctions.CROSS_ENTROPY, new Adam(0.1), new StochasticUpdater());
    }
}

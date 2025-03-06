import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.impl.Transformer;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.transformers.TransformerEncoder;

import java.text.DecimalFormat;

public class TransformerExample {

    public static void main(String[] args) throws InterruptedException {
        TransformerExample example = new TransformerExample();
        example.start();
    }

    public void start() throws InterruptedException {
        Transformer transformer = new Transformer(
                new TransformerEncoder(4, 784, 1.0),
                new TransformerEncoder(4, 784, 1.0),
                new TransformerEncoder(4, 784, 1.0),
                new TransformerEncoder(4, 784, 1.0)
        );

        transformer.compile(LossFunctions.BINARY_CROSS_ENTROPY, new Adam(0.001));

        System.out.println(transformer.getStats());

        Thread.sleep(Integer.MAX_VALUE);
    }
}

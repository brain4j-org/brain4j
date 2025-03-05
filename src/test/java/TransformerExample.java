import net.echo.brain4j.model.impl.Transformer;
import net.echo.brain4j.transformers.TransformerEncoder;

public class TransformerExample {

    public static void main(String[] args) {

    }

    public void start() {
        Transformer transformer = new Transformer(
                new TransformerEncoder(4, 16, 1.0)
        );


    }
}

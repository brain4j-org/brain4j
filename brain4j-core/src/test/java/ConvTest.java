import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.DropoutLayer;
import org.brain4j.core.layer.impl.conv.ConvLayer;
import org.brain4j.core.layer.impl.conv.FlattenLayer;
import org.brain4j.core.layer.impl.conv.InputLayer;
import org.brain4j.core.layer.impl.conv.pooling.MaxPooling;
import org.brain4j.core.loss.Loss;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.core.training.optimizer.impl.AdamW;
import org.brain4j.math.activation.Activations;

public class ConvTest {

    public static void main(String[] args) {
        String skibidi = """
                \033[94m[30/50]\033[0m \033[92m━━━━━━━━━━━━━━━━━━━━\033[0m \033[93m60,00%\033[0m \033[96m[0,71ms/epoch | 14,27ms remaining]\033[0m
                \033[94m[30/50]\033[0m \033[35mLoss:\033[0m \033[37m0,0028\033[0m \033[92m| Accuracy:\033[0m \033[97m100,00%\033[0m \033[92m| F1-Score:\033[0m \033[97m100,00%\033[0m
                """;
        System.out.println(skibidi);
        Model model = new Sequential(
                new InputLayer(28, 28, 1),
                new ConvLayer(Activations.RELU, 32, 3,3),
                new MaxPooling(2, 2),
                new ConvLayer(Activations.RELU, 64, 3, 3),
                new MaxPooling(2, 2),
                new FlattenLayer(),
                new DenseLayer(128, Activations.RELU),
                new DropoutLayer(0.5),
                new DenseLayer(10, Activations.SOFTMAX)
        );
        model.compile(Loss.CROSS_ENTROPY, new AdamW(0.01));
    }
}

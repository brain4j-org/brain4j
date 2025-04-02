package matrices;

import net.echo.math4j.BrainUtils;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

public class MatrixDerivationTest {

    public static void main(String[] args) {
        new MatrixDerivationTest().start();
    }

    public void start() {
        Tensor A = TensorFactory.matrix(2, 3,
            1, 2, 3,
            4, 5, 6);
        Tensor B = TensorFactory.matrix(3, 2,
                1, 2,
                3, 4,
                5, 6);
        Tensor D = TensorFactory.matrix(2, 2,
                1, 2,
                3, 4);

        double learningRate = BrainUtils.estimateMaxLearningRate(A) * 0.9;

        System.out.println("Suggested learning rate: " + learningRate);

        for (int i = 0; i < 1000; i++) {
            Tensor C = A.matmul(B);
            Tensor delta = C.clone().sub(D);

            Tensor gradient = A.transpose().matmul(delta).mul(learningRate);
            B.sub(gradient);

            if (i % 100 == 0) {
                System.out.println("Iteration " + i + ", Loss: " + delta.norm());
            }
        }
    }
}

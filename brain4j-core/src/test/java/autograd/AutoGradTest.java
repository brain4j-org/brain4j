package autograd;

import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.TensorFactory;

public class AutoGradTest {

    public static void main(String[] args) {
        new AutoGradTest().start();
    }

    public void start() {
        Tensor a = TensorFactory.vector(1, 2, 3).requiresGrad(true);
        Tensor b = TensorFactory.vector(4, 5, 6).requiresGrad(true);

        Tensor c = a.addWithGrad(b);        // c = a + b
        Tensor z = c.mulWithGrad(b);        // z = c * b (element-wise)

        z.backward();  // lancia la backward

        System.out.println("dz/da: " + a.grad());  // atteso: [4, 5, 6]
        System.out.println("dz/db: " + b.grad());  // atteso: [9, 12, 15]

    }
}

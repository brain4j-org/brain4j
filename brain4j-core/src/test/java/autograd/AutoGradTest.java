package autograd;

import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.Tensors;

public class AutoGradTest {

    public static void main(String[] args) {
        new AutoGradTest().start();
    }

    public void start() {
        Tensor a = Tensors.matrix(2, 3,
                1, 2, 3,
                4, 5, 6).withGrad();

        Tensor b = Tensors.matrix(3, 1,
                1,
                2,
                3).withGrad();

        Tensor c = a.matmulWithGrad(b);
        c.backward();

        System.out.println("c: " + c);
        System.out.println("dz/da: " + a.grad());
        System.out.println("dz/db: " + b.grad());
    }
}

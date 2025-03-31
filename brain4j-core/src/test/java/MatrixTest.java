import org.ejml.simple.SimpleMatrix;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

import java.util.Random;

public class MatrixTest {

    public static SimpleMatrix matmul(SimpleMatrix A, SimpleMatrix B) {
        return A.mult(B); // EJML usa questa funzione per la moltiplicazione matriciale
    }

    public static void main(String[] args) {
        int n = 10;
        int m = 1;
        int p = 10;

        SimpleMatrix A = SimpleMatrix.random_FDRM(n, m, 1, 1, new Random());
        SimpleMatrix B = SimpleMatrix.random_FDRM(m, p, 1, 1, new Random());

        long start = System.nanoTime();
        SimpleMatrix result = matmul(A, B);
        long end = System.nanoTime();

        System.out.println("EJML Time: " + (end - start) / 1e6 + " ms");

//        Tensor tensorA = TensorFactory.matrix(m, n).fill(1);
//        Tensor tensorB = TensorFactory.matrix(n, p).fill(1);

        Tensor tensorA = TensorFactory.matrix(n, m).fill(1);
        Tensor tensorB = TensorFactory.matrix(m, p).fill(1);

        long b4jStartFast = System.nanoTime();
        Tensor tensorResult = tensorA.matmulFast(tensorB);
        long b4jEndFast = System.nanoTime();

        long b4jStartNormal = System.nanoTime();
        Tensor tensorResult2 = tensorA.matmul(tensorB);
        long b4jEndNormal = System.nanoTime();

        System.out.println(tensorResult2.equals(tensorResult));

        System.out.println("Fast time: " + (b4jEndFast - b4jStartFast) / 1e6 + " ms");
        System.out.println("Normal time: " + (b4jEndNormal - b4jStartNormal) / 1e6 + " ms");
    }
}

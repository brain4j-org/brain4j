package matrices;

import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;
import org.ejml.simple.SimpleMatrix;

import java.util.Random;

public class MatrixMultiplicationTest {

    public static void main(String[] args) {
        int n = 10;
        int m = 1;
        int p = 10;

        SimpleMatrix A = SimpleMatrix.random_FDRM(n, m, 1, 1, new Random());
        SimpleMatrix B = SimpleMatrix.random_FDRM(m, p, 1, 1, new Random());

        long start = System.nanoTime();
        A.mult(B);
        long end = System.nanoTime();

        System.out.println("EJML Time: " + (end - start) / 1e6 + " ms");

        Tensor tensorA = TensorFactory.matrix(n, m).fill(1);
        Tensor tensorB = TensorFactory.matrix(m, p).fill(1);

        long b4jStartFast = System.nanoTime();
        tensorA.oldMatmul(tensorB);
        long b4jEndFast = System.nanoTime();

        long b4jStartNormal = System.nanoTime();
        tensorA.matmul(tensorB);
        long b4jEndNormal = System.nanoTime();

        double msTookOld = (b4jEndFast - b4jStartFast) / 1e6;
        double msTookNew = (b4jEndNormal - b4jStartNormal) / 1e6;
        double speedup = msTookOld / msTookNew;

        System.out.println("Old method: " + msTookOld + " ms");
        System.out.println("New method: " + msTookNew + " ms");
        System.out.printf("Speed up: %.3fx", speedup);
    }
}

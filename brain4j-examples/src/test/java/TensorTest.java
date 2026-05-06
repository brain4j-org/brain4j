import org.brain4j.core.Brain4J;
import org.brain4j.math.Tensors;
import org.brain4j.math.gpu.silicon.SiliconDevice;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.commons.Range;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class TensorTest {
    
    private final SiliconDevice device;

    public TensorTest() {
        this.device = Brain4J.firstDevice();
    }

    @Test
    public void orthogonalTest() {
        Tensor A = Tensors.orthogonal(3, 3);
        Tensor T = Tensors.matrix(3, 3,
                1, 0, 0,
                0, 1, 0,
                0, 0, 1
        );

        Tensor R = A.transpose().matmul(A);
        assertArrayEquals(R.data(), T.data(), 0.0001f);
    }

    @Test
    public void activationTest() {
        Tensor a = Tensors.random(10, 10);
        Tensor gpuA = a.to(device);

        Tensor b = a.relu();
        Tensor gpuB = gpuA.relu();

        assertArrayEquals(b.data(), gpuB.data(), 0.0001f);
    }

    @Test
    public void convTest() {
        Tensor a = Tensors.random(512, 512);
        Tensor b = Tensors.random(3, 3);
        
        Tensor gpuA = a.to(device);
        Tensor gpuB = b.to(device);
        
        Tensor c = a.convolve(b);
        Tensor gpuC = gpuA.convolve(gpuB);
        
        assertArrayEquals(c.data(), gpuC.data(), 0.0001f);
    }
    
    @Test
    public void normTest() {
        double epsilon = 1e-5;

        Tensor A = Tensors.random(32, 16, 64);
        Tensor B = A.layerNorm(epsilon);

        Tensor gpuA = A.to(device);
        Tensor gpuB = gpuA.layerNorm(epsilon);

        assertArrayEquals(B.data(), gpuB.data(), 0.001f);
    }

    @Test
    public void concatTest() {
        Tensor A = Tensors.random(2, 3);
        Tensor B = Tensors.random(2, 2);
        Tensor C = A.concat(B);

        Tensor gpuA = A.to(device);
        Tensor gpuB = B.to(device);
        Tensor gpuC = gpuA.concat(gpuB);

        assertArrayEquals(C.shape(), gpuC.shape());
        assertArrayEquals(C.data(), gpuC.data(), 0.001f);
    }

    @Test
    public void matmulTest() {
        Tensor A = Tensors.random(4, 8);
        Tensor B = Tensors.random(8, 3);
        Tensor C = A.matmul(B);

        Tensor gpuA = A.to(device);
        Tensor gpuB = B.to(device);
        Tensor gpuC = gpuA.matmul(gpuB);

        assertArrayEquals(C.shape(), gpuC.shape());
        assertArrayEquals(C.data(), gpuC.data(), 0.001f);
    }

    @Test
    public void sliceTest() {
        Range[] ranges = { Range.all(), Range.interval(10, 20) };

        Tensor A = Tensors.random(64, 32);
        Tensor B = A.slice(ranges);

        Tensor gpuA = A.to(device);
        Tensor gpuB = gpuA.slice(ranges);

        assertArrayEquals(B.shape(), gpuB.shape());
        assertArrayEquals(B.data(), gpuB.data(), 0.001f);
    }

    @Test
    public void addTest() {
        Tensor A = Tensors.random(32, 32);
        Tensor B = Tensors.random(32, 32);
        Tensor C = A.plus(B);

        Tensor gpuA = A.to(device);
        Tensor gpuB = B.to(device);
        Tensor gpuC = gpuA.plus(gpuB);

        assertArrayEquals(C.data(), gpuC.data(), 0.001f);
    }

    @Test
    public void subTest() {
        Tensor A = Tensors.random(32, 32);
        Tensor B = Tensors.random(32, 32);
        Tensor C = A.minus(B);

        Tensor gpuA = A.to(device);
        Tensor gpuB = B.to(device);
        Tensor gpuC = gpuA.minus(gpuB);

        assertArrayEquals(C.data(), gpuC.data(), 0.001f);
    }

    @Test
    public void mulTest() {
        Tensor A = Tensors.random(32, 32);
        Tensor B = Tensors.random(32, 32);
        Tensor C = A.times(B);

        Tensor gpuA = A.to(device);
        Tensor gpuB = B.to(device);
        Tensor gpuC = gpuA.times(gpuB);

        assertArrayEquals(C.data(), gpuC.data(), 0.001f);
    }

    @Test
    public void divTest() {
        Tensor A = Tensors.random(32, 32);
        Tensor B = Tensors.random(32, 32);
        Tensor C = A.divide(B);

        Tensor gpuA = A.to(device);
        Tensor gpuB = B.to(device);
        Tensor gpuC = gpuA.divide(gpuB);

        assertArrayEquals(C.data(), gpuC.data(), 0.001f);
    }

    @Test
    public void chainTest() {
        Tensor A = Tensors.random(16, 16);
        Tensor B = A.matmul(A.transpose()).relu().sum(0, false);

        Tensor gpuA = A.to(device);
        Tensor gpuB = gpuA.matmul(gpuA.transpose()).relu().sum(0, false);

        assertEquals(B.get(0), gpuB.get(0), 0.001f);
    }

    @Test
    public void transposeTest() {
        Tensor A = Tensors.random(4, 8);
        Tensor B = Tensors.random(3, 8);
        Tensor C = A.matmul(B.transpose());

        Tensor gpuA = A.to(device);
        Tensor gpuB = B.to(device);
        Tensor gpuC = gpuA.matmul(gpuB.transpose());

        assertArrayEquals(C.shape(), gpuC.shape());
        assertArrayEquals(C.data(), gpuC.data(), 0.001f);
    }

    @Test
    void testTranspose3D() {
        Tensor t = Tensors.create(Shape.of(2, 2, 2),
            1, 2,
                3, 4,
                5, 6,
                7, 8);

        Tensor transposed = t.transpose(0, 1);

        assertArrayEquals(new int[]{2, 2, 2}, transposed.shape());

        assertEquals(1, transposed.get(0, 0, 0));
        assertEquals(5, transposed.get(0, 1, 0));
        assertEquals(3, transposed.get(1, 0, 0));
        assertEquals(7, transposed.get(1, 1, 0));
    }

    @Test
    public void testCloneTransposedTensor() {
        Tensor A = Tensors.matrix(2, 3,
                1, 2, 3,
                4, 5, 6
        );

        Tensor A_T = A.transpose();
        Tensor A_T_clone = A_T.copy();

        int[] expectedStrides = Tensors.computeStrides(A_T_clone.shape());
        assertArrayEquals(expectedStrides, A_T_clone.strides(),
                "Cloned transposed tensor should have contiguous strides");

        assertEquals(1, A_T_clone.get(0, 0), 0.001f);
        assertEquals(4, A_T_clone.get(0, 1), 0.001f);
        assertEquals(2, A_T_clone.get(1, 0), 0.001f);
        assertEquals(5, A_T_clone.get(1, 1), 0.001f);
        assertEquals(3, A_T_clone.get(2, 0), 0.001f);
        assertEquals(6, A_T_clone.get(2, 1), 0.001f);

        float[] expectedData = {1, 4, 2, 5, 3, 6};
        assertArrayEquals(expectedData, A_T_clone.data(), 0.001f,
                "Cloned data should be in contiguous row-major order");
    }

    @Test
    public void testTimesWithTransposedTensor() {
        Tensor P = Tensors.matrix(2, 2,
                0.7f, 0.3f,
                0.4f, 0.6f
        );

        Tensor V = Tensors.matrix(2, 2,
                1, 2,
                3, 4
        );
        Tensor V_T = V.transpose(); // V_T is a view with non-standard strides

        // dO @ V_T
        Tensor dO = Tensors.matrix(2, 2,
                1, 1,
                1, 1
        );

        Tensor dP = dO.matmul(V_T);
        Tensor PdP = P.times(dP);

        assertArrayEquals(new int[]{2, 2}, PdP.shape());
        assertEquals(2.1f, PdP.get(0, 0), 0.01f);
        assertEquals(2.1f, PdP.get(0, 1), 0.01f);
        assertEquals(1.2f, PdP.get(1, 0), 0.01f);
        assertEquals(4.2f, PdP.get(1, 1), 0.01f);
    }

    @Test
    public void testMatmulWithTransposedResult() {
        Tensor A = Tensors.matrix(2, 3,
                1, 2, 3,
                4, 5, 6
        );
        Tensor B = Tensors.matrix(3, 2,
                1, 2,
                3, 4,
                5, 6
        );

        // C = A @ B -> [2, 2]
        Tensor C = A.matmul(B);

        // C_T = C^T -> [2, 2] (transposed view)
        Tensor C_T = C.transpose();

        // D should be able to use C_T in further operations
        Tensor D = Tensors.matrix(2, 2,
                1, 0,
                0, 1
        );

        Tensor result = C_T.matmul(D);

        // check that the result is equal to C_T (since D is identity)
        assertArrayEquals(C_T.shape(), result.shape());

        assertEquals(22, result.get(0, 0), 0.01f);
        assertEquals(49, result.get(0, 1), 0.01f);
        assertEquals(28, result.get(1, 0), 0.01f);
        assertEquals(64, result.get(1, 1), 0.01f);
    }
}

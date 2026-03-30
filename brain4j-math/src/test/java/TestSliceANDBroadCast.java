import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.device.DeviceUtils;
import org.brain4j.math.tensor.impl.GpuTensor;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.brain4j.math.tensor.index.Range;// ← aggiunto

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;  // ← era org.junit.Assert
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class TestSliceANDBroadCast {
    private static Device device;

    @BeforeAll
    static void setup() {
        device = DeviceUtils.findDevice(null);
        assertNotNull(device, "No GPU found!");
        GpuTensor.initKernels(device);
    }

    @Test
    public void testBroadcastRows() {
        GpuTensor input = new GpuTensor(device, new int[]{1, 3}, 10f, 20f, 30f);
        int[] in = new int[]{4,3};
        float[] result = input.broadcast(in).data();
        float[] expected = {10,20,30, 10,20,30, 10,20,30, 10,20,30};
        System.out.println("testBroadcastRows: " + Arrays.toString(result));
        assertArrayEquals(expected, result, 1e-4f);
    }

    @Test
    public void testBroadcastCols() {
        GpuTensor input = new GpuTensor(device, new int[]{4, 1}, 10f, 20f, 30f, 40f);
        int[] in = new int[]{4,3};
        float[] result = input.broadcast(in).data();
        float[] expected = {10,10,10, 20,20,20, 30,30,30, 40,40,40};
        System.out.println("testBroadcastCols: " + Arrays.toString(result));
        assertArrayEquals(expected, result, 1e-4f);
    }

    @Test
    public void testBroadcastScalar() {
        GpuTensor input = new GpuTensor(device, new int[]{1, 1}, 42f);
        int[] in = new int[]{4,3};
        float[] result = input.broadcast(in).data();
        float[] expected = new float[12];
        Arrays.fill(expected, 42f);
        System.out.println("testBroadcastScalar: " + Arrays.toString(result));
        assertArrayEquals(expected, result, 1e-4f);
    }

    @Test
    public void testBroadcastAlreadyCorrectShape() {
        GpuTensor input = new GpuTensor(device, new int[]{4, 3},
                1f,2f,3f, 4f,5f,6f, 7f,8f,9f, 10f,11f,12f);
        int[] in = new int[]{4,3};
        float[] result = input.broadcast(in).data();
        float[] expected = {1,2,3, 4,5,6, 7,8,9, 10,11,12};
        System.out.println("testBroadcastAlreadyCorrectShape: " + Arrays.toString(result));
        assertArrayEquals(expected, result, 1e-4f);
    }

    @Test
    public void testSlicePartialRanges() {
        GpuTensor input = new GpuTensor(device, new int[]{4, 3},
                1f,2f,3f, 4f,5f,6f, 7f,8f,9f, 10f,11f,12f);

        float[] result = input.slice(Range.interval(0, 2)).data();
        float[] expected = {1f, 2f, 3f, 4f, 5f, 6f};

        System.out.println("testSlicePartialRanges: 1 " + Arrays.toString(result));
        assertArrayEquals(expected, result, 1e-4f);


         result = input.slice(new Range(1,4,2)).data();
         expected = new float[]{4f, 5f, 6f, 10f, 11f, 12f};

        System.out.println("testSlicePartialRanges: 2" + Arrays.toString(result));
        assertArrayEquals(expected, result, 1e-4f);
    }

    @Test
    public void testBroadcast() {
        GpuTensor input = new GpuTensor(device, new int[]{2, 3},
                1f,2f,3f, 4f,5f,6f);


        float[] result = input.broadcast(new int[]{4,2,3}).data();
        float[] expected = {1f, 2f, 3f, 4f, 5f, 6f, 1f, 2f, 3f, 4f, 5f, 6f
        , 1f, 2f, 3f, 4f, 5f, 6f, 1f, 2f, 3f, 4f, 5f, 6f};

        System.out.println("testBroadcast: 1 " + Arrays.toString(result));
        assertArrayEquals(expected, result, 1e-4f);


    }
}
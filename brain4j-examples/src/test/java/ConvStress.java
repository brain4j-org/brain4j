import org.brain4j.core.Brain4J;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.convolutional.ConvLayer;
import org.brain4j.core.layer.impl.utility.InputLayer;
import org.brain4j.core.layer.impl.utility.ReshapeLayer;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.ModelSpecs;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.gpu.silicon.SiliconDevice;
import org.brain4j.math.tensor.Tensor;

import java.util.concurrent.Callable;

public class ConvStress {
    
    public static void main(String[] args) {
        new ConvStress().start();
    }
    
    private void start() {
        Model model = getSpecs().compile(42);
        model.summary();
        
        SiliconDevice device = Brain4J.firstDevice();
        Model gpuModel = model.fork(device);
        
        Tensor x = Tensors.random(32, 3, 224, 224);
        
        double tookCpu = benchmark(() -> model.predict(x), 5, 10);
        Tensor gpuX = x.to(device);
        double tookGpu = benchmark(() -> gpuModel.predict(gpuX), 5, 10);
        
        System.out.printf("took cpu = %.2f %n", tookCpu);
        System.out.printf("took gpu = %.2f %n", tookGpu);
    }
    
    public double benchmark(Callable<?> op, int warmup, int runs) {
        try {
            // warmup
            for (int i = 0; i < warmup; i++) {
                op.call();
            }
            
            long start = System.nanoTime();
            for (int i = 0; i < runs; i++) {
                op.call();
            }
            long end = System.nanoTime();
            
            return (end - start) / (runs * 1e6);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
    
    private ModelSpecs getSpecs() {
        return ModelSpecs.of(
            new InputLayer(3, 224, 224),
            
            new ConvLayer(3, 64, 7, 7, 2, Activations.LEAKY_RELU),     // 64x109x109
            new ConvLayer(64, 128, 3, 3, 2, Activations.LEAKY_RELU),   // 128x54x54
            new ConvLayer(128, 256, 3, 3, 2, Activations.LEAKY_RELU),  // 256x26x26
            new ConvLayer(256, 256, 3, 3, 1, Activations.LEAKY_RELU),  // 256x24x24
            new ConvLayer(256, 512, 3, 3, 2, Activations.LEAKY_RELU),  // 512x11x11
            
            new ReshapeLayer(512 * 11 * 11),
            new DenseLayer(1024, Activations.LEAKY_RELU),
            new DenseLayer(1000, Activations.SOFTMAX)
        );
    }
}

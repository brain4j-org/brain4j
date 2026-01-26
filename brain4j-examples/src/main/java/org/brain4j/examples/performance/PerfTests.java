package org.brain4j.examples.performance;

import org.brain4j.core.Brain4J;
import org.brain4j.math.Tensors;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.device.DeviceUtils;
import org.brain4j.math.gpu.silicon.SiliconDevice;
import org.brain4j.math.gpu.silicon.SiliconDeviceUtils;
import org.brain4j.math.tensor.Tensor;
import org.silicon.Silicon;
import org.silicon.backend.BackendType;

import java.util.List;

public class PerfTests {
    
    public static final int WARMUP = 5;
    public static final int ROUNDS = 20;
    
    public static void main(String[] args) {
        PerfTests instance = new PerfTests();
        instance.matmul();
    }
    
    private double evaluate(Runnable task) {
        for (int i = 0; i < WARMUP; i++) {
            task.run();
        }
        
        long start = System.nanoTime();
        for (int i = 0; i < ROUNDS; i++) {
            task.run();
        }
        long end = System.nanoTime();
        long tookNanos = end - start;
        
        double tookMs = tookNanos / 1e6;
        return tookMs / ROUNDS;
    }
    
    public void matmul() {
        SiliconDevice cudaDevice = Brain4J.firstDevice();
        Device openCLDevice = Brain4J.firstLegacyDevice();
        
        int N = 16;
        
        Tensor a = Tensors.random(N, N);
        Tensor b = Tensors.random(N, N);
        
        double cpuTime = evaluate(() -> a.matmul(b));
        
        Tensor cudaA = a.to(cudaDevice);
        Tensor cudaB = b.to(cudaDevice);
        
        double cudaTime = evaluate(() -> cudaA.matmul(cudaB));
        
        Tensor openClA = a.to(openCLDevice);
        Tensor openClB = b.to(openCLDevice);
        
        double openClTime = evaluate(() -> openClA.matmul(openClB));
        
        System.out.printf("cpu time = %.2f, cuda time = %.2f, opencl time = %.2f %n", cpuTime, cudaTime, openClTime);
    }
    
    private Device firstDevice() {
        List<String> devices = DeviceUtils.allDeviceNames();
        
        if (devices.isEmpty()) {
            throw new IllegalStateException("No GPU-acceleration device has been found!");
        }
        
        Device device = DeviceUtils.findDevice(devices.getFirst());
        
        if (device != null) Brain4J.initKernels(device);
        
        return device;
    }
}

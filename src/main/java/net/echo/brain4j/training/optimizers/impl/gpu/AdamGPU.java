package net.echo.brain4j.training.optimizers.impl.gpu;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.structure.cache.Parameters;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.utils.opencl.DeviceUtils;
import org.jocl.*;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import static org.jocl.CL.*;

public class AdamGPU extends Optimizer {

    protected Synapse[] synapses;

    protected float beta1Timestep;
    protected float beta2Timestep;

    protected float beta1;
    protected float beta2;
    protected float epsilon;
    protected int timestep = 0;

    // OpenCL-related fields
    private cl_context context;
    private cl_command_queue commandQueue;
    private cl_kernel kernel;
    private cl_mem updatesMemory;
    private cl_mem gradientsMemory;
    private long size;
    private long localWorkSize;

    public AdamGPU(double learningRate) {
        this(learningRate, 0.9f, 0.999f, 0.00001f); // Anything below 1e-6 is 0
    }

    public AdamGPU(double learningRate, float beta1, float beta2, float epsilon) {
        super(learningRate);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;

        initialize();
    }

    private void initialize() {
        cl_device_id[] devices = { DeviceUtils.findDevice(DeviceUtils.DeviceType.GPU) };

        this.localWorkSize = DeviceUtils.getMaxLocalWorkSize();
        this.context = clCreateContext(null, 1, devices, null, null, null);
        this.commandQueue = clCreateCommandQueue(context, devices[0], 0, null);

        String[] kernelSource = { loadKernelSource() };

        cl_program program = clCreateProgramWithSource(context, 1, kernelSource, null, null);
        clBuildProgram(program, 0, null, null, null, null);

        this.kernel = clCreateKernel(program, "adam_update", null);
    }

    private String loadKernelSource() {
        try (InputStream inputStream = getClass().getClassLoader().getResourceAsStream("kernels/adam_kernel.cl")) {
            if (inputStream == null) {
                throw new RuntimeException("Kernel file not found: kernels/adam_update_kernel.cl");
            }

            return new String(inputStream.readAllBytes());
        } catch (IOException e) {
            throw new RuntimeException("Failed to load kernel file", e);
        }
    }

    @Override
    public void postInitialize(Sequential model) {
        this.synapses = new Synapse[Parameters.TOTAL_SYNAPSES];

        for (Layer<?, ?> layer : model.getLayers()) {
            for (Synapse synapse : layer.getSynapses()) {
                synapses[synapse.getSynapseId()] = synapse;
            }
        }

        this.size = (long) Parameters.TOTAL_SYNAPSES * Sizeof.cl_float;

        cl_mem firstMomentum = DeviceUtils.createBuffer(context, CL_MEM_READ_WRITE, size);
        cl_mem secondMomentum = DeviceUtils.createBuffer(context, CL_MEM_READ_WRITE, size);

        this.updatesMemory = DeviceUtils.createBuffer(context, CL_MEM_READ_ONLY, size);
        this.gradientsMemory = DeviceUtils.createBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size);

        DeviceUtils.writeBuffer(commandQueue, firstMomentum, size, new float[Parameters.TOTAL_SYNAPSES]);
        DeviceUtils.writeBuffer(commandQueue, secondMomentum, size, new float[Parameters.TOTAL_SYNAPSES]);

        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(firstMomentum));
        clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(secondMomentum));
        clSetKernelArg(kernel, 4, Sizeof.cl_float, DeviceUtils.to(beta1));
        clSetKernelArg(kernel, 5, Sizeof.cl_float, DeviceUtils.to(beta2));
        clSetKernelArg(kernel, 6, Sizeof.cl_float, DeviceUtils.to(1f - beta1));
        clSetKernelArg(kernel, 7, Sizeof.cl_float, DeviceUtils.to(1f - beta2));
        clSetKernelArg(kernel, 10, Sizeof.cl_float, DeviceUtils.to(epsilon));
        clSetKernelArg(kernel, 11, Sizeof.cl_float, DeviceUtils.to((float) learningRate));
        clSetKernelArg(kernel, 12, Sizeof.cl_int, DeviceUtils.to(Parameters.TOTAL_SYNAPSES));
    }

    @Override
    public void postIteration(StatesCache cacheHolder, Updater updater, List<Layer<?, ?>> layers) {
        this.timestep++;

        this.beta1Timestep = (float) (1.0 - Math.pow(beta1, timestep));
        this.beta2Timestep = (float) (1.0 - Math.pow(beta2, timestep));

        float[] gradients = new float[Parameters.TOTAL_SYNAPSES];

        for (Layer<?, ?> layer : layers) {
            for (Synapse synapse : layer.getSynapses()) {
                int synapseId = synapse.getSynapseId();

                double delta = synapse.getOutputNeuron().getDelta(cacheHolder);
                double value = synapse.getInputNeuron().getValue(cacheHolder);

                gradients[synapseId] = (float) (delta * value);
            }
        }

        executeKernel(cacheHolder, updater, gradients);
    }
    
    private void executeKernel(StatesCache cacheHolder, Updater updater, float[] gradients) {
        float[] updates = new float[Parameters.TOTAL_SYNAPSES];
        cl_event kernelEvent = new cl_event();

        clEnqueueWriteBuffer(commandQueue, gradientsMemory, CL_TRUE, 0, size, Pointer.to(gradients), 0, null, null);

        clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(gradientsMemory));
        clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(updatesMemory));
        clSetKernelArg(kernel, 8, Sizeof.cl_float, DeviceUtils.to(beta1Timestep));
        clSetKernelArg(kernel, 9, Sizeof.cl_float, DeviceUtils.to(beta2Timestep));

        // Launch kernel async
        DeviceUtils.runKernel(commandQueue, kernel, localWorkSize, Parameters.TOTAL_SYNAPSES, kernelEvent);

        clSetEventCallback(kernelEvent, CL_SUBMITTED, (event1, status1, userData1) -> {
            DeviceUtils.readBuffer(commandQueue, updatesMemory, size, updates);

            applyChanges(cacheHolder, updater, updates);
        }, updates);
    }

    @Override
    public double update(StatesCache cache, Synapse synapse) {
        // CPU-based update fallback
        return 0; // Not used when GPU is enabled
    }

    @Override
    public double update(StatesCache cache, int id, float gradient, float weight) {
        // CPU-based update fallback
        return 0;
    }

    private void applyChanges(StatesCache cache, Updater updater, float[] updates) {
        for (int i = 0; i < updates.length; i++) {
            Synapse synapse = synapses[i];
            float update = updates[i];

            updater.acknowledgeChange(synapse, update);
        }
    }

    @Override
    public void setLearningRate(double learningRate) {
        super.setLearningRate(learningRate);
        clSetKernelArg(kernel, 11, Sizeof.cl_float, DeviceUtils.to((float) learningRate));
    }
}
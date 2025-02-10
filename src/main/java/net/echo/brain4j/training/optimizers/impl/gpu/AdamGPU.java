package net.echo.brain4j.training.optimizers.impl.gpu;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.utils.opencl.DeviceUtils;
import org.jocl.*;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.structure.StatesCache;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import org.jocl.*;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import static org.jocl.CL.*;

public class AdamGPU extends Optimizer {

    protected Synapse[] synapses;

    protected float beta1Timestep;
    protected float beta2Timestep;

    private long size;

    protected float beta1;
    protected float beta2;
    protected float epsilon;
    protected int timestep = 0;

    // OpenCL-related fields
    private cl_context context;
    private cl_command_queue commandQueue;
    private cl_kernel kernel;

    private cl_mem dUpdates;
    private cl_mem dGradients;

    public AdamGPU(double learningRate) {
        this(learningRate, 0.9f, 0.999f, 0.000001f); // Anything below 1e-6 is 0
    }

    public AdamGPU(double learningRate, float beta1, float beta2, float epsilon) {
        super(learningRate);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;

        initialize();
    }

    private void initialize() {
        cl_device_id device = DeviceUtils.findDevice(DeviceUtils.DeviceType.GPU);

        System.out.println("Using Device: " + DeviceUtils.getDeviceName());

        cl_platform_id[] platforms = new cl_platform_id[1];
        CL.clGetPlatformIDs(1, platforms, null);

        System.out.println("OpenCL Version: " + DeviceUtils.getOpenCLVersion());

        context = clCreateContext(null, 1, new cl_device_id[]{device}, null, null, null);
        commandQueue = clCreateCommandQueue(context, device, 0, null);

        String kernelSource = loadKernelSource();

        cl_program program = clCreateProgramWithSource(context, 1, new String[]{kernelSource}, null, null);
        clBuildProgram(program, 0, null, null, null, null);

        kernel = clCreateKernel(program, "adam_update", null);
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
    public void postInitialize(Model model) {
        this.synapses = new Synapse[Synapse.SYNAPSE_COUNTER];

        for (Layer layer : model.getLayers()) {
            for (Synapse synapse : layer.getSynapses()) {
                synapses[synapse.getSynapseId()] = synapse;
            }
        }

        this.size = (long) Synapse.SYNAPSE_COUNTER * Sizeof.cl_float;

        cl_mem dFirstMomentum = DeviceUtils.createBuffer(context, CL_MEM_READ_WRITE, size);
        cl_mem dSecondMomentum = DeviceUtils.createBuffer(context, CL_MEM_READ_WRITE, size);

        this.dUpdates = DeviceUtils.createBuffer(context, CL_MEM_READ_WRITE, size);
        this.dGradients = DeviceUtils.createBuffer(context, CL_MEM_READ_ONLY, size);

        DeviceUtils.writeBuffer(commandQueue, dFirstMomentum, size, new float[Synapse.SYNAPSE_COUNTER]);
        DeviceUtils.writeBuffer(commandQueue, dSecondMomentum, size, new float[Synapse.SYNAPSE_COUNTER]);

        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(dFirstMomentum));
        clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(dSecondMomentum));
        clSetKernelArg(kernel, 4, Sizeof.cl_float, DeviceUtils.to(beta1));
        clSetKernelArg(kernel, 5, Sizeof.cl_float, DeviceUtils.to(beta2));
        clSetKernelArg(kernel, 6, Sizeof.cl_float, DeviceUtils.to(1.0 - beta1));
        clSetKernelArg(kernel, 7, Sizeof.cl_float, DeviceUtils.to(1.0 - beta2));
        clSetKernelArg(kernel, 10, Sizeof.cl_float, DeviceUtils.to(epsilon));
        clSetKernelArg(kernel, 11, Sizeof.cl_float, DeviceUtils.to((float) learningRate));
        clSetKernelArg(kernel, 12, Sizeof.cl_int, DeviceUtils.to(Synapse.SYNAPSE_COUNTER));
    }

    @Override
    public void postIteration(StatesCache cacheHolder, Updater updater, List<Layer> layers) {
        this.timestep++;

        this.beta1Timestep = (float) (1.0 - Math.pow(beta1, timestep));
        this.beta2Timestep = (float) (1.0 - Math.pow(beta2, timestep));

        float[] gradients = new float[Synapse.SYNAPSE_COUNTER];

        for (Layer layer : layers) {
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
        float[] updates = new float[Synapse.SYNAPSE_COUNTER];
        cl_event kernelEvent = new cl_event();

        clEnqueueWriteBuffer(commandQueue, dGradients, CL_TRUE, 0, size, Pointer.to(gradients), 0, null, null);

        clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(dGradients));
        clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(dUpdates));
        clSetKernelArg(kernel, 8, Sizeof.cl_float, DeviceUtils.to(beta1Timestep));
        clSetKernelArg(kernel, 9, Sizeof.cl_float, DeviceUtils.to(beta2Timestep));

        long[] globalWorkSize = new long[]{(long) Synapse.SYNAPSE_COUNTER};

        // Launch kernel async
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, globalWorkSize, null, 0, null, kernelEvent);

        clSetEventCallback(kernelEvent, CL_SUBMITTED, (event1, status1, userData1) -> {
            float[] updates1 = (float[]) userData1;

            DeviceUtils.readBuffer(commandQueue, dUpdates, size, updates1);

            applyChanges(cacheHolder, updater, updates1);
        }, updates);
    }

    @Override
    public double update(StatesCache cacheHolder, Synapse synapse) {
        // CPU-based update fallback
        return 0; // Not used when GPU is enabled
    }

    private void applyChanges(StatesCache cacheHolder, Updater updater, float[] updates) {
        for (int i = 0; i < updates.length; i++) {
            Synapse synapse = synapses[i];
            float update = updates[i];

            updater.acknowledgeChange(cacheHolder, synapse, update);
        }
    }

    @Override
    public void setLearningRate(double learningRate) {
        super.setLearningRate(learningRate);
        clSetKernelArg(kernel, 11, Sizeof.cl_float, DeviceUtils.to((float) learningRate));
    }
}
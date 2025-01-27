package net.echo.brain4j.training.optimizers.impl.gpu;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.opencl.DeviceUtils;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.threading.NeuronCacheHolder;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import org.jocl.*;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import static org.jocl.CL.*;

public class AdamWGPU extends Optimizer {

    protected Synapse[] synapses;

    protected double beta1Timestep;
    protected double beta2Timestep;

    private long size;
    protected double beta1;
    protected double beta2;
    protected double epsilon;
    protected double weightDecay;
    protected int timestep = 0;

    // OpenCL-related fields
    private cl_context context;
    private cl_command_queue commandQueue;
    private cl_kernel kernel;

    private cl_mem dFirstMomentum;
    private cl_mem dSecondMomentum;
    private cl_mem dUpdates;
    private cl_mem dGradients;
    private cl_mem dWeights;

    public AdamWGPU(double learningRate) {
        this(learningRate, 0.001);
    }

    public AdamWGPU(double learningRate, double weightDecay) {
        this(learningRate, 0.9, 0.999, 1e-8, weightDecay);
    }

    public AdamWGPU(double learningRate, double beta1, double beta2, double epsilon, double weightDecay) {
        super(learningRate);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;

        initialize();
    }

    private void initialize() {
        cl_device_id device = DeviceUtils.findDevice(DeviceUtils.DeviceType.GPU);

        System.out.println("Using " + DeviceUtils.getDeviceName());

        this.context = clCreateContext(null, 1, new cl_device_id[]{device}, null, null, null);
        this.commandQueue = clCreateCommandQueueWithProperties(context, device, null, null);

package org.brain4j.common.device;

import org.jocl.*;

import static org.jocl.CL.*;

public class Device {

    static {
        CL.setExceptionsEnabled(true);
    }

    private final cl_platform_id platform;
    private final cl_device_id device;
    private final cl_context context;

    public Device(cl_platform_id platform, cl_device_id device) {
        this.platform = platform;
        this.device = device;
        this.context = newContext();
    }

    public cl_context newContext() {
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

        cl_device_id[] devices = new cl_device_id[] { device };
        return clCreateContext(contextProperties, 1, devices, null, null, null);
    }

    public cl_command_queue newCommandQueue() {
        cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, null);

        if (commandQueue == null) {
            throw new RuntimeException("Failed to create command queue");
        }

        return commandQueue;
    }

    public String name() {
        return DeviceUtils.deviceName(device);
    }

    public cl_platform_id platform() {
        return platform;
    }

    public cl_device_id device() {
        return device;
    }

    public cl_context context() {
        return context;
    }
}

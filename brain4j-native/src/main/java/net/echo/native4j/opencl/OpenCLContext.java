package net.echo.native4j.opencl;

import org.jocl.*;

import static org.jocl.CL.*;

public class OpenCLContext {

    private static OpenCLContext instance;
    
    private boolean initialized;
    private cl_device_id deviceId;
    private cl_context context;
    private cl_command_queue commandQueue;
    private KernelManager kernelManager;
    
    private OpenCLContext() {
        try {
            initialized = initialize();
        } catch (Exception e) {
            System.err.println("GPU acceleration not available: " + e.getMessage());
            initialized = false;
        }
    }
    
    public static synchronized OpenCLContext getInstance() {
        if (instance == null) {
            instance = new OpenCLContext();
        }
        return instance;
    }
    
    private boolean initialize() {
        try {
            CL.setExceptionsEnabled(true);
            
            deviceId = DeviceManager.findDevice(DeviceManager.DeviceType.GPU);
            
            if (deviceId == null) {
                System.out.println("No GPU device found, falling back to CPU");
                deviceId = DeviceManager.findDevice(DeviceManager.DeviceType.CPU);
                
                if (deviceId == null) {
                    System.err.println("No OpenCL device available");
                    return false;
                }
            }
            
            cl_device_id[] devices = {deviceId};
            context = clCreateContext(null, 1, devices, null, null, null);
            commandQueue = clCreateCommandQueue(context, deviceId, 0, null);
            
            kernelManager = new KernelManager(context, deviceId);
            boolean kernelsInitialized = kernelManager.initializeKernels();
            
            if (kernelsInitialized) {
                System.out.println("GPU acceleration enabled using device: " + DeviceManager.getDeviceName());
            } else {
                System.err.println("Failed to initialize OpenCL kernels");
                return false;
            }
            
            return kernelsInitialized;
        } catch (Exception e) {
            System.err.println("Failed to initialize OpenCL: " + e.getMessage());
            return false;
        }
    }
    
    public boolean isInitialized() {
        return initialized;
    }
    
    public cl_device_id getDeviceId() {
        return deviceId;
    }
    
    public cl_context getContext() {
        return context;
    }
    
    public cl_command_queue getCommandQueue() {
        return commandQueue;
    }
    
    public KernelManager getKernelManager() {
        return kernelManager;
    }
    
    public cl_kernel getKernel(String kernelName) {
        if (kernelManager != null) {
            return kernelManager.getKernel(kernelName);
        }
        return null;
    }
    
    public synchronized void releaseResources() {
        if (initialized) {
            try {
                if (kernelManager != null) {
                    kernelManager.releaseResources();
                }
                
                if (commandQueue != null) {
                    clReleaseCommandQueue(commandQueue);
                    commandQueue = null;
                }
                
                if (context != null) {
                    clReleaseContext(context);
                    context = null;
                }
                
                initialized = false;
            } catch (Exception e) {
                System.err.println("Error releasing OpenCL resources: " + e.getMessage());
            }
        }
    }
    
    public synchronized boolean reinitialize() {
        if (!initialized) {
            try {
                initialized = initialize();
                return initialized;
            } catch (Exception e) {
                System.err.println("Failed to reinitialize OpenCL: " + e.getMessage());
                return false;
            }
        }
        return true;
    }
} 
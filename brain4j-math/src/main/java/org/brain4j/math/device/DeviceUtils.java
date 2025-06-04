package org.brain4j.math.device;

import org.brain4j.math.tensor.impl.TensorGPU;
import org.jocl.*;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import static org.jocl.CL.*;

public class DeviceUtils {

    private static Device device;

    static {
        CL.setExceptionsEnabled(true);
    }

    public static Device findDevice(DeviceType deviceType, String name) {
        int[] numPlatformsArray = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];

        cl_platform_id[] platforms = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);

        for (cl_platform_id platform : platforms) {
            int[] numDevicesArray = new int[1];
            int result = clGetDeviceIDs(platform, deviceType.getMask(), 0, null, numDevicesArray);

            if (result != CL_SUCCESS) return null;

            int numDevices = numDevicesArray[0];

            cl_device_id[] devices = new cl_device_id[numDevices];
            clGetDeviceIDs(platform, deviceType.getMask(), numDevices, devices, null);

            if (name == null) {
                return new Device(platform, devices[0]);
            }

            for (cl_device_id dev : devices) {
                if (getDeviceName(dev).contains(name)) {
                    return new Device(platform, dev);
                }
            }
        }

        return null;
    }

    public static void setDevice(Device value) {
        device = value;
    }

    public static String getDeviceName(cl_device_id d) {
        long[] size = new long[1];
        clGetDeviceInfo(d, CL_DEVICE_NAME, 0, null, size);

        byte[] buffer = new byte[(int) size[0]];
        clGetDeviceInfo(d, CL_DEVICE_NAME, buffer.length, Pointer.to(buffer), null);

        return new String(buffer, 0, buffer.length - 1).trim();
    }

    public static List<String> getAllDeviceNames() {
        List<String> deviceNames = new ArrayList<>();

        int[] numPlatformsArray = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];

        cl_platform_id[] platforms = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);

        for (cl_platform_id platform : platforms) {
            int[] numDevicesArray = new int[1];
            int result = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, null, numDevicesArray);

            if (result != CL_SUCCESS) continue;

            int numDevices = numDevicesArray[0];

            cl_device_id[] devices = new cl_device_id[numDevices];
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices, null);

            for (cl_device_id dev : devices) {
                deviceNames.add(getDeviceName(dev));
            }
        }

        return deviceNames;
    }

    public static Device device() {
        return device;
    }

    public static String getDeviceName() {
        return getDeviceName(device.device());
    }

    public static cl_command_queue newCommandQueue(Device device) {
        cl_queue_properties properties = new cl_queue_properties();
        cl_command_queue commandQueue = clCreateCommandQueueWithProperties(
                device.context(),
                device.device(),
                properties,
                null
        );

        if (commandQueue == null) {
            throw new RuntimeException("Failed to create command queue");
        }

        return commandQueue;
    }

    public static String readKernelSource(String resourcePath) {
        try (InputStream input = TensorGPU.class.getResourceAsStream(resourcePath)) {
            if (input == null) {
                throw new IllegalArgumentException("Resource not found: " + resourcePath);
            }
            return new String(input.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to read kernel source from: " + resourcePath, e);
        }
    }
}

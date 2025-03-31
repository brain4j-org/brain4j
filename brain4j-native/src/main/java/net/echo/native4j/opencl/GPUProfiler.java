package net.echo.native4j.opencl;

import org.jocl.*;
import static org.jocl.CL.*;

public class GPUProfiler {

    public static void printDefaultDeviceInfo() {
        cl_device_id device = DeviceManager.getDevice();
        
        if (device == null) {
            System.out.println("No OpenCL-compatible GPU available");
            return;
        }
        
        System.out.println("\n==== OpenCL Device Info ====");
        System.out.println("Device: " + DeviceManager.getDeviceName());
        System.out.println("OpenCL Version: " + DeviceManager.getOpenCLVersion());
        
        long globalMemSize = getDeviceInfoLong(device, CL_DEVICE_GLOBAL_MEM_SIZE);
        long localMemSize = getDeviceInfoLong(device, CL_DEVICE_LOCAL_MEM_SIZE);
        long maxAllocSize = getDeviceInfoLong(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
        
        System.out.println("Global Memory: " + formatMemSize(globalMemSize));
        System.out.println("Local Memory: " + formatMemSize(localMemSize));
        System.out.println("Max Allocation Size: " + formatMemSize(maxAllocSize));
        
        long maxComputeUnits = getDeviceInfoLong(device, CL_DEVICE_MAX_COMPUTE_UNITS);
        long maxWorkGroupSize = getDeviceInfoLong(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
        
        System.out.println("Compute Units: " + maxComputeUnits);
        System.out.println("Max Work Group Size: " + maxWorkGroupSize);
        
        long[] maxWorkItemSizes = getDeviceInfoLongArray(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3);
        System.out.print("Max Work Item Sizes: [");
        for (int i = 0; i < maxWorkItemSizes.length; i++) {
            System.out.print(maxWorkItemSizes[i]);
            if (i < maxWorkItemSizes.length - 1) {
                System.out.print(", ");
            }
        }
        System.out.println("]");
        
        String driverVersion = getDeviceInfoString(device, CL_DRIVER_VERSION);
        System.out.println("Driver Version: " + driverVersion);
        
        System.out.println("==============================\n");
    }
    
    public static String getHardwareInfo() {
        StringBuilder info = new StringBuilder();
        cl_device_id device = DeviceManager.getDevice();
        
        if (device == null) {
            return "No OpenCL-compatible GPU available";
        }
        
        info.append("Device: ").append(DeviceManager.getDeviceName()).append("\n");
        info.append("Compute Units: ").append(getDeviceInfoLong(device, CL_DEVICE_MAX_COMPUTE_UNITS)).append("\n");
        info.append("Global Memory: ").append(formatMemSize(getDeviceInfoLong(device, CL_DEVICE_GLOBAL_MEM_SIZE))).append("\n");
        
        return info.toString();
    }
    
    private static String formatMemSize(long sizeInBytes) {
        if (sizeInBytes < 1024) {
            return sizeInBytes + " B";
        } else if (sizeInBytes < 1024 * 1024) {
            return String.format("%.2f KB", sizeInBytes / 1024.0);
        } else if (sizeInBytes < 1024 * 1024 * 1024) {
            return String.format("%.2f MB", sizeInBytes / (1024.0 * 1024.0));
        } else {
            return String.format("%.2f GB", sizeInBytes / (1024.0 * 1024.0 * 1024.0));
        }
    }
    
    private static long getDeviceInfoLong(cl_device_id device, int paramName) {
        long[] result = new long[1];
        clGetDeviceInfo(device, paramName, Sizeof.cl_long, Pointer.to(result), null);
        return result[0];
    }
    
    private static long[] getDeviceInfoLongArray(cl_device_id device, int paramName, int size) {
        long[] result = new long[size];
        clGetDeviceInfo(device, paramName, Sizeof.cl_long * size, Pointer.to(result), null);
        return result;
    }
    
    private static String getDeviceInfoString(cl_device_id device, int paramName) {
        long[] size = new long[1];
        clGetDeviceInfo(device, paramName, 0, null, size);
        byte[] buffer = new byte[(int)size[0]];
        clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);
        return new String(buffer, 0, buffer.length - 1).trim();
    }
} 
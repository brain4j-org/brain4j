package net.echo.brain4j.opencl;

import org.jocl.*;

import static org.jocl.CL.*;
import static org.jocl.CL.clGetDeviceInfo;

public class DeviceUtils {

    private static cl_device_id device;

    static {
        CL.setExceptionsEnabled(true);
    }

    public static cl_platform_id getPlatform() {
        cl_platform_id[] platforms = new cl_platform_id[1];
        clGetPlatformIDs(1, platforms, null);

        return platforms[0];
    }

    public static cl_device_id findDevice(DeviceType deviceType) {
        cl_platform_id platform = getPlatform();

        cl_device_id[] devices = new cl_device_id[1];
        clGetDeviceIDs(platform, deviceType.mask, 2, devices, null);

        return device = devices[0];
    }

    public static String getDeviceName() {
        long[] size = new long[1];
        clGetDeviceInfo(device, CL_DEVICE_NAME, 0, null, size);

        byte[] buffer = new byte[(int) size[0]];

        clGetDeviceInfo(device, CL.CL_DEVICE_NAME, buffer.length, Pointer.to(buffer), null);

        return new String(buffer, 0, buffer.length - 1).trim();
    }

    public static cl_device_id getDevice() {
        return device;
    }

    public static cl_mem createBuffer(cl_context context, long flags, long size) {
        return clCreateBuffer(context, flags, size, null, null);
    }

    public static void writeBuffer(cl_command_queue commandQueue, cl_mem memory, long size, double[] target) {
        clEnqueueWriteBuffer(commandQueue, memory, CL_TRUE, 0, size, Pointer.to(target), 0, null, null);
    }

    public static Pointer to(double... values) {
        return Pointer.to(values);
    }

    public static Pointer to(int... values) {
        return Pointer.to(values);
    }

    public static Pointer to(byte... values) {
        return Pointer.to(values);
    }

    public static Pointer to(long... values) {
        return Pointer.to(values);
    }

    public static void readBuffer(cl_command_queue commandQueue, cl_mem memory, long size, double[] target) {
        clEnqueueReadBuffer(commandQueue, memory, CL_TRUE, 0, size, Pointer.to(target), 0,
                null, null);
    }

    public static long getInfoLong(int flag, int arraySize, int size) {
        long[] memory = new long[arraySize];

        clGetDeviceInfo(device, flag, size, Pointer.to(memory), null);

        return memory[0];
    }

    public static String getOpenCLVersion(){
        cl_platform_id[] platforms = new cl_platform_id[1];

        CL.clGetPlatformIDs(1, platforms, null);
        cl_platform_id platform = platforms[0];

        byte[] version = new byte[1024];
        CL.clGetPlatformInfo(platform, CL.CL_PLATFORM_VERSION, 1024, Pointer.to(version), null);

        return new String(version).trim();
    }

    public static void awaitAndRunKernel(cl_command_queue commandQueue, cl_kernel kernel, int workDimension, long[] globalWorkSize) {
        clEnqueueNDRangeKernel(
                commandQueue,
                kernel,
                1,
                null,
                globalWorkSize,
                null,
                0,
                null,
                null
        );
        clFinish(commandQueue);
    }

    public enum DeviceType {
        DEFAULT(1),
        CPU(1 << 1),
        GPU(1 << 2),
        ACCELERATOR(1 << 3),
        CUSTOM(1 << 4);

        private final long mask;

        DeviceType(long mask) {
            this.mask = mask;
        }
    }
}

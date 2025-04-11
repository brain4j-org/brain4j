package net.echo.math4j.device;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.cl_device_id;
import org.jocl.cl_platform_id;

import static org.jocl.CL.*;

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
        clGetDeviceIDs(platform, deviceType.getMask(), 2, devices, null);

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
}

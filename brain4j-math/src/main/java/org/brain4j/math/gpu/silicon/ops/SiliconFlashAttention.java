package org.brain4j.math.gpu.silicon.ops;

import org.brain4j.math.gpu.silicon.SiliconContext;
import org.brain4j.math.gpu.silicon.SiliconDevice;
import org.brain4j.math.gpu.silicon.SiliconKernel;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.SiliconGpuTensor;
import org.silicon.api.kernel.ComputeSize;

public class SiliconFlashAttention {

    private static final int FA_TILE_SIZE = 16;
    private static final int FA_HEAD_DIM = 64;

    private SiliconFlashAttention() { }

    public static Tensor forward(Tensor q, Tensor k, Tensor v, double scale, boolean causal) {
        if (!(q instanceof SiliconGpuTensor Q) ||
            !(k instanceof SiliconGpuTensor K) ||
            !(v instanceof SiliconGpuTensor V)) {
            return null;
        }
        if (!Q.device().equals(K.device()) || !Q.device().equals(V.device())) {
            return null;
        }

        int[] shape = Q.shape();
        int B = shape[0];
        int H = shape[1];
        int L = shape[2];
        int D = shape[3];

        SiliconDevice device = Q.device();
        SiliconGpuTensor O = new SiliconGpuTensor(device, shape);
        O.setAutogradContext(Q.getAutogradContext());

        try (SiliconContext.QueueHandle qh = SiliconContext.getOrCreateQueue(device)) {
            ComputeSize globalSize = new ComputeSize(L, B * H, 1);

            SiliconKernel.create(device, "flash_attention_forward")
                .buffer(Q.getDataBuffer())
                .buffer(K.getDataBuffer())
                .buffer(V.getDataBuffer())
                .buffer(O.getDataBuffer())
                .buffer(Q.getStridesBuffer())
                .buffer(K.getStridesBuffer())
                .buffer(V.getStridesBuffer())
                .buffer(O.getStridesBuffer())
                .intVal(B)
                .intVal(H)
                .intVal(L)
                .intVal(D)
                .floatVal((float) scale)
                .intVal(causal ? 1 : 0)
                .launch(qh.queue(), globalSize);
            qh.queue().await();
        } catch (Throwable e) {
            throw new RuntimeException("FlashAttention forward failed", e);
        }

        return O;
    }

    public static Tensor[] forwardWithLse(Tensor q, Tensor k, Tensor v, double scale, boolean causal) {
        if (!(q instanceof SiliconGpuTensor Q) ||
            !(k instanceof SiliconGpuTensor K) ||
            !(v instanceof SiliconGpuTensor V)) {
            return null;
        }
        if (!Q.device().equals(K.device()) || !Q.device().equals(V.device())) {
            return null;
        }

        int[] shape = Q.shape();
        int B = shape[0];
        int H = shape[1];
        int L = shape[2];
        int D = shape[3];

        SiliconDevice device = Q.device();
        SiliconGpuTensor O = new SiliconGpuTensor(device, shape);
        O.setAutogradContext(Q.getAutogradContext());

        // LSE has shape [B, H, L]
        SiliconGpuTensor LSE = new SiliconGpuTensor(device, new int[]{B, H, L});

        try (SiliconContext.QueueHandle qh = SiliconContext.getOrCreateQueue(device)) {
            ComputeSize globalSize = new ComputeSize(L, B * H, 1);

            SiliconKernel.create(device, "flash_attention_forward_with_lse")
                .buffer(Q.getDataBuffer())
                .buffer(K.getDataBuffer())
                .buffer(V.getDataBuffer())
                .buffer(O.getDataBuffer())
                .buffer(LSE.getDataBuffer())
                .buffer(Q.getStridesBuffer())
                .buffer(K.getStridesBuffer())
                .buffer(V.getStridesBuffer())
                .buffer(O.getStridesBuffer())
                .intVal(B)
                .intVal(H)
                .intVal(L)
                .intVal(D)
                .floatVal((float) scale)
                .intVal(causal ? 1 : 0)
                .launch(qh.queue(), globalSize);
            qh.queue().await();
        } catch (Throwable e) {
            throw new RuntimeException("FlashAttention forward with LSE failed", e);
        }

        return new Tensor[] { O, LSE };
    }

    public static Tensor[] backward(
            Tensor q, Tensor k, Tensor v, Tensor o, Tensor dO, Tensor lse,
            double scale, boolean causal
    ) {
        if (!(q instanceof SiliconGpuTensor Q) ||
            !(k instanceof SiliconGpuTensor K) ||
            !(v instanceof SiliconGpuTensor V) ||
            !(o instanceof SiliconGpuTensor O) ||
            !(dO instanceof SiliconGpuTensor DO) ||
            !(lse instanceof SiliconGpuTensor LSE)) {
            return null;
        }

        SiliconDevice device = Q.device();
        if (!device.equals(K.device()) || !device.equals(V.device()) ||
            !device.equals(O.device()) || !device.equals(DO.device()) ||
            !device.equals(LSE.device())) {
            return null;
        }

        int[] shape = Q.shape();
        int B = shape[0];
        int H = shape[1];
        int L = shape[2];
        int D = shape[3];

        SiliconGpuTensor dQ = new SiliconGpuTensor(device, shape);
        SiliconGpuTensor dK = new SiliconGpuTensor(device, shape);
        SiliconGpuTensor dV = new SiliconGpuTensor(device, shape);

        try (SiliconContext.QueueHandle qh = SiliconContext.getOrCreateQueue(device)) {
            ComputeSize globalSize = new ComputeSize(L, B * H, 1);

            SiliconKernel.create(device, "flash_attention_backward")
                .buffer(Q.getDataBuffer())
                .buffer(K.getDataBuffer())
                .buffer(V.getDataBuffer())
                .buffer(O.getDataBuffer())
                .buffer(DO.getDataBuffer())
                .buffer(LSE.getDataBuffer())
                .buffer(dQ.getDataBuffer())
                .buffer(dK.getDataBuffer())
                .buffer(dV.getDataBuffer())
                .buffer(Q.getStridesBuffer())
                .buffer(K.getStridesBuffer())
                .buffer(V.getStridesBuffer())
                .buffer(O.getStridesBuffer())
                .buffer(DO.getStridesBuffer())
                .buffer(dK.getStridesBuffer())
                .buffer(dV.getStridesBuffer())
                .intVal(B)
                .intVal(H)
                .intVal(L)
                .intVal(D)
                .floatVal((float) scale)
                .intVal(causal ? 1 : 0)
                .launch(qh.queue(), globalSize);

            SiliconKernel.create(device, "flash_attention_backward_dq")
                .buffer(Q.getDataBuffer())
                .buffer(K.getDataBuffer())
                .buffer(V.getDataBuffer())
                .buffer(O.getDataBuffer())
                .buffer(DO.getDataBuffer())
                .buffer(LSE.getDataBuffer())
                .buffer(dQ.getDataBuffer())
                .buffer(Q.getStridesBuffer())
                .buffer(K.getStridesBuffer())
                .buffer(V.getStridesBuffer())
                .buffer(O.getStridesBuffer())
                .buffer(DO.getStridesBuffer())
                .buffer(dQ.getStridesBuffer())
                .intVal(B)
                .intVal(H)
                .intVal(L)
                .intVal(D)
                .floatVal((float) scale)
                .intVal(causal ? 1 : 0)
                .launch(qh.queue(), globalSize);

            qh.queue().await();
        } catch (Throwable e) {
            throw new RuntimeException("FlashAttention backward failed", e);
        }

        return new Tensor[] { dQ, dK, dV };
    }

    public static Tensor[] forwardTiled(Tensor q, Tensor k, Tensor v, double scale, boolean causal) {
        if (!(q instanceof SiliconGpuTensor Q) ||
            !(k instanceof SiliconGpuTensor K) ||
            !(v instanceof SiliconGpuTensor V)) {
            return null;
        }
        if (!Q.device().equals(K.device()) || !Q.device().equals(V.device())) {
            return null;
        }

        int[] shape = Q.shape();
        int B = shape[0];
        int H = shape[1];
        int L = shape[2];
        int D = shape[3];

        SiliconDevice device = Q.device();
        SiliconGpuTensor O = new SiliconGpuTensor(device, shape);
        O.setAutogradContext(Q.getAutogradContext());
        SiliconGpuTensor LSE = new SiliconGpuTensor(device, new int[]{B, H, L});

        try (SiliconContext.QueueHandle qh = SiliconContext.getOrCreateQueue(device)) {
            ComputeSize globalSize = new ComputeSize(L, B * H, 1);
            ComputeSize localSize = new ComputeSize(FA_TILE_SIZE, 1, 1);

            SiliconKernel.create(device, "flash_attention_forward_tiled")
                .buffer(Q.getDataBuffer())
                .buffer(K.getDataBuffer())
                .buffer(V.getDataBuffer())
                .buffer(O.getDataBuffer())
                .buffer(LSE.getDataBuffer())
                .buffer(Q.getStridesBuffer())
                .buffer(K.getStridesBuffer())
                .buffer(V.getStridesBuffer())
                .buffer(O.getStridesBuffer())
                .intVal(B)
                .intVal(H)
                .intVal(L)
                .intVal(D)
                .floatVal((float) scale)
                .intVal(causal ? 1 : 0)
                .launch(qh.queue(), globalSize, localSize);
            qh.queue().await();
        } catch (Throwable e) {
            throw new RuntimeException("FlashAttention forward tiled failed", e);
        }

        return new Tensor[] { O, LSE };
    }

    public static Tensor[] backwardTiled(
            Tensor q, Tensor k, Tensor v, Tensor o, Tensor dO, Tensor lse,
            double scale, boolean causal
    ) {
        if (!(q instanceof SiliconGpuTensor Q) ||
            !(k instanceof SiliconGpuTensor K) ||
            !(v instanceof SiliconGpuTensor V) ||
            !(o instanceof SiliconGpuTensor O) ||
            !(dO instanceof SiliconGpuTensor DO) ||
            !(lse instanceof SiliconGpuTensor LSE)) {
            return null;
        }

        SiliconDevice device = Q.device();
        if (!device.equals(K.device()) || !device.equals(V.device()) ||
            !device.equals(O.device()) || !device.equals(DO.device()) ||
            !device.equals(LSE.device())) {
            return null;
        }

        int[] shape = Q.shape();
        int B = shape[0];
        int H = shape[1];
        int L = shape[2];
        int D = shape[3];

        SiliconGpuTensor dQ = new SiliconGpuTensor(device, shape);
        SiliconGpuTensor dK = new SiliconGpuTensor(device, shape);
        SiliconGpuTensor dV = new SiliconGpuTensor(device, shape);

        try (SiliconContext.QueueHandle qh = SiliconContext.getOrCreateQueue(device)) {
            ComputeSize globalSize = new ComputeSize(L, B * H, 1);
            ComputeSize localSize = new ComputeSize(FA_TILE_SIZE, 1, 1);

            SiliconKernel.create(device, "flash_attention_backward_tiled")
                .buffer(Q.getDataBuffer())
                .buffer(K.getDataBuffer())
                .buffer(V.getDataBuffer())
                .buffer(O.getDataBuffer())
                .buffer(DO.getDataBuffer())
                .buffer(LSE.getDataBuffer())
                .buffer(dQ.getDataBuffer())
                .buffer(dK.getDataBuffer())
                .buffer(dV.getDataBuffer())
                .buffer(Q.getStridesBuffer())
                .buffer(K.getStridesBuffer())
                .buffer(V.getStridesBuffer())
                .buffer(O.getStridesBuffer())
                .buffer(DO.getStridesBuffer())
                .buffer(dK.getStridesBuffer())
                .buffer(dV.getStridesBuffer())
                .intVal(B)
                .intVal(H)
                .intVal(L)
                .intVal(D)
                .floatVal((float) scale)
                .intVal(causal ? 1 : 0)
                .launch(qh.queue(), globalSize, localSize);

            SiliconKernel.create(device, "flash_attention_backward_dq")
                .buffer(Q.getDataBuffer())
                .buffer(K.getDataBuffer())
                .buffer(V.getDataBuffer())
                .buffer(O.getDataBuffer())
                .buffer(DO.getDataBuffer())
                .buffer(LSE.getDataBuffer())
                .buffer(dQ.getDataBuffer())
                .buffer(Q.getStridesBuffer())
                .buffer(K.getStridesBuffer())
                .buffer(V.getStridesBuffer())
                .buffer(O.getStridesBuffer())
                .buffer(DO.getStridesBuffer())
                .buffer(dQ.getStridesBuffer())
                .intVal(B)
                .intVal(H)
                .intVal(L)
                .intVal(D)
                .floatVal((float) scale)
                .intVal(causal ? 1 : 0)
                .launch(qh.queue(), globalSize);

            qh.queue().await();
        } catch (Throwable e) {
            throw new RuntimeException("FlashAttention backward tiled failed", e);
        }

        return new Tensor[] { dQ, dK, dV };
    }
}


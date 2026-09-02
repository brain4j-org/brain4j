package org.brain4j.core.importing.io;

import org.brain4j.core.codec.OnnxCodec;
import org.brain4j.core.codec.onnx.*;
import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.impl.*;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.tensor.autograd.Operation;
import org.brain4j.math.tensor.autograd.impl.ActivationOperation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Central registry for ONNX operation codecs.
 * JSON-related codecs stay in {@link LayerIO}, while all operation-specific
 * ONNX codecs live here. This keeps the {@code Codec} hierarchy clean:
 * base {@code Codec} -> {@code JsonCodec} (layers/activations) and {@code OnnxCodec} (operations).
 */
public final class OnnxIO {

    private static final Map<String, OnnxCodec<? extends Operation>> BY_TYPE = new HashMap<>();
    private static final List<OnnxCodec<? extends Operation>> ALL = new ArrayList<>();

    private static final Map<Class<? extends Activation>, String> ACTIVATION_TO_ONNX = Map.of(
        ReLU.class, "Relu",
        GELU.class, "Gelu",
        Softmax.class, "Softmax",
        Sigmoid.class, "Sigmoid",
        Tanh.class, "Tanh",
        LeakyReLU.class, "LeakyRelu"
    );

    static {
        // stateless
        register(new AddOnnxCodec());
        register(new SubOnnxCodec());
        register(new MulOnnxCodec());
        register(new DivOnnxCodec());
        register(new MatMulOnnxCodec());
        register(new GemmOnnxCodec());
        // stateful
        register(new ConcatOnnxCodec());
        register(new SqueezeOnnxCodec());
        register(new LayerNormOnnxCodec());
        register(new ReshapeOnnxCodec());
        register(new TransposeOnnxCodec());
        // activations (one codec per ONNX type, same targetClass)
        register(new ReluOnnxCodec());
        register(new SigmoidOnnxCodec());
        register(new TanhOnnxCodec());
        register(new LeakyReluOnnxCodec());
        register(new GeluOnnxCodec());
        register(new SoftmaxOnnxCodec());
    }

    private OnnxIO() {}

    private static void register(OnnxCodec<? extends Operation> codec) {
        if (BY_TYPE.put(codec.type(), codec) != null) {
            throw Commons.illegalState("Duplicate ONNX codec type: %s", codec.type());
        }
        ALL.add(codec);
    }

    public static OnnxCodec<? extends Operation> get(String type) {
        return BY_TYPE.get(type);
    }

    @SuppressWarnings("unchecked")
    public static Operation decode(NodeProto node) {
        OnnxCodec<? extends Operation> codec = BY_TYPE.get(node.getOpType());

        if (codec == null) return null;

        OnnxCodec<Operation> c = (OnnxCodec<Operation>) codec;
        return c.decode(node);
    }

    public static String encodeType(Operation op) {
        if (op instanceof ActivationOperation(Activation activation)) {
            String t = ACTIVATION_TO_ONNX.get(activation.getClass());
            if (t != null) return t;
        }

        for (OnnxCodec<? extends Operation> codec : ALL) {
            if (codec.targetClass().isInstance(op)) {
                // For ActivationOperation, ensure the codec's type matches the activation
                if (op instanceof ActivationOperation) {
                    continue;
                }

                return codec.type();
            }
        }

        // Fallback for ActivationOperation if not found via map (should not happen)
        if (op instanceof ActivationOperation(Activation activation)) {
            return ACTIVATION_TO_ONNX.getOrDefault(activation.getClass(), null);
        }
        return null;
    }

    @SuppressWarnings("unchecked")
    public static void encode(Operation op, NodeProto.Builder builder) {
        OnnxCodec<? extends Operation> codec = findCodecForOp(op);

        if (codec == null) return;

        OnnxCodec<Operation> c = (OnnxCodec<Operation>) codec;
        c.encode(op, builder);
    }

    private static OnnxCodec<? extends Operation> findCodecForOp(Operation op) {
        if (op instanceof ActivationOperation(Activation activation)) {
            String t = ACTIVATION_TO_ONNX.get(activation.getClass());
            if (t != null) return BY_TYPE.get(t);
        }

        for (OnnxCodec<? extends Operation> codec : ALL) {
            if (codec.targetClass().equals(op.getClass()))
                return codec;

            if (codec.targetClass().isInstance(op) && !(op instanceof ActivationOperation))
                return codec;
        }
        return null;
    }
}

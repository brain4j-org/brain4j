package org.brain4j.core.importing.io;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.brain4j.core.codec.OnnxCodec;
import org.brain4j.core.codec.onnx.AddOnnxCodec;
import org.brain4j.core.codec.onnx.ConcatOnnxCodec;
import org.brain4j.core.codec.onnx.DivOnnxCodec;
import org.brain4j.core.codec.onnx.EluOnnxCodec;
import org.brain4j.core.codec.onnx.GeluOnnxCodec;
import org.brain4j.core.codec.onnx.GemmOnnxCodec;
import org.brain4j.core.codec.onnx.LayerNormOnnxCodec;
import org.brain4j.core.codec.onnx.LeakyReluOnnxCodec;
import org.brain4j.core.codec.onnx.LinearOnnxCodec;
import org.brain4j.core.codec.onnx.MatMulOnnxCodec;
import org.brain4j.core.codec.onnx.MishOnnxCodec;
import org.brain4j.core.codec.onnx.MulOnnxCodec;
import org.brain4j.core.codec.onnx.ReluOnnxCodec;
import org.brain4j.core.codec.onnx.ReshapeOnnxCodec;
import org.brain4j.core.codec.onnx.SigmoidOnnxCodec;
import org.brain4j.core.codec.onnx.SliceOnnxCodec;
import org.brain4j.core.codec.onnx.SoftPlusOnnxCodec;
import org.brain4j.core.codec.onnx.SoftmaxOnnxCodec;
import org.brain4j.core.codec.onnx.SqueezeOnnxCodec;
import org.brain4j.core.codec.onnx.SubOnnxCodec;
import org.brain4j.core.codec.onnx.SwishOnnxCodec;
import org.brain4j.core.codec.onnx.TanhOnnxCodec;
import org.brain4j.core.codec.onnx.TransposeOnnxCodec;
import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.impl.ELU;
import org.brain4j.math.activation.impl.GELU;
import org.brain4j.math.activation.impl.LeakyReLU;
import org.brain4j.math.activation.impl.Linear;
import org.brain4j.math.activation.impl.Mish;
import org.brain4j.math.activation.impl.ReLU;
import org.brain4j.math.activation.impl.Sigmoid;
import org.brain4j.math.activation.impl.SoftPlus;
import org.brain4j.math.activation.impl.Softmax;
import org.brain4j.math.activation.impl.Swish;
import org.brain4j.math.activation.impl.Tanh;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.operations.Operation;
import org.brain4j.math.operations.impl.ActivationOperation;

/**
 * Maps ONNX op types to codecs for import and export.
 * Use it to decode nodes and encode operations.
 */
public final class OnnxIO {

    private static final Map<String, OnnxCodec<? extends Operation>> BY_TYPE = new HashMap<>();
    private static final List<OnnxCodec<? extends Operation>> ALL = new ArrayList<>();

    private static final Map<Class<? extends Activation>, String> ACTIVATION_TO_ONNX = Map.ofEntries(
        Map.entry(ReLU.class, "Relu"),
        Map.entry(GELU.class, "Gelu"),
        Map.entry(Softmax.class, "Softmax"),
        Map.entry(Sigmoid.class, "Sigmoid"),
        Map.entry(Tanh.class, "Tanh"),
        Map.entry(LeakyReLU.class, "LeakyRelu"),
        Map.entry(ELU.class, "Elu"),
        Map.entry(SoftPlus.class, "Softplus"),
        Map.entry(Mish.class, "Mish"),
        Map.entry(Swish.class, "SiLU"),
        Map.entry(Linear.class, "Identity")
    );

    static {
        register(new AddOnnxCodec());
        register(new SubOnnxCodec());
        register(new MulOnnxCodec());
        register(new DivOnnxCodec());
        register(new MatMulOnnxCodec());
        register(new GemmOnnxCodec());
        register(new ConcatOnnxCodec());
        register(new SqueezeOnnxCodec());
        register(new LayerNormOnnxCodec());
        register(new ReshapeOnnxCodec());
        register(new TransposeOnnxCodec());
        register(new SliceOnnxCodec());
        register(new ReluOnnxCodec());
        register(new SigmoidOnnxCodec());
        register(new TanhOnnxCodec());
        register(new LeakyReluOnnxCodec());
        register(new GeluOnnxCodec());
        register(new SoftmaxOnnxCodec());
        register(new EluOnnxCodec());
        register(new SoftPlusOnnxCodec());
        register(new MishOnnxCodec());
        register(new SwishOnnxCodec());
        register(new LinearOnnxCodec());
    }

    private OnnxIO() {
    }

    private static void register(OnnxCodec<? extends Operation> codec) {
        if (BY_TYPE.put(codec.type(), codec) != null) {
            throw Commons.illegalState("Duplicate ONNX codec type: %s", codec.type());
        }

        ALL.add(codec);
    }

    /**
     * Find the codec for the given ONNX op type.
     *
     * @return the codec, or {@code null} if missing
     */
    public static OnnxCodec<? extends Operation> get(String type) {
        return BY_TYPE.get(type);
    }

    /**
     * Decode a node into its operation.
     *
     * @param node the source node
     * @return the operation, or {@code null} if unknown
     */
    @SuppressWarnings("unchecked")
    public static Operation decode(NodeProto node) {
        OnnxCodec<? extends Operation> codec = BY_TYPE.get(node.getOpType());

        if (codec == null) {
            return null;
        }

        OnnxCodec<Operation> c = (OnnxCodec<Operation>) codec;

        return c.decode(node);
    }

    /**
     * Find the ONNX op type for the given operation.
     *
     * @return the op type, or {@code null} if missing
     */
    public static String encodeType(Operation op) {
        if (op instanceof ActivationOperation(Activation activation)) {
            String t = ACTIVATION_TO_ONNX.get(activation.getClass());

            if (t != null) {
                return t;
            }
        }

        for (OnnxCodec<? extends Operation> codec : ALL) {
            if (codec.targetClass().isInstance(op)) {
                if (op instanceof ActivationOperation) {
                    continue;
                }

                return codec.type();
            }
        }

        if (op instanceof ActivationOperation(Activation activation)) {
            return ACTIVATION_TO_ONNX.getOrDefault(activation.getClass(), null);
        }

        return null;
    }

    /**
     * Encode an operation into the given node builder.
     *
     * @param builder the target builder
     */
    @SuppressWarnings("unchecked")
    public static void encode(Operation op, NodeProto.Builder builder) {
        OnnxCodec<? extends Operation> codec = findCodecForOp(op);

        if (codec == null) {
            return;
        }

        OnnxCodec<Operation> c = (OnnxCodec<Operation>) codec;
        c.encode(op, builder);
    }

    private static OnnxCodec<? extends Operation> findCodecForOp(Operation op) {
        if (op instanceof ActivationOperation(Activation activation)) {
            String t = ACTIVATION_TO_ONNX.get(activation.getClass());

            if (t != null) {
                return BY_TYPE.get(t);
            }
        }

        for (OnnxCodec<? extends Operation> codec : ALL) {
            if (codec.targetClass().equals(op.getClass())) {
                return codec;
            }

            if (codec.targetClass().isInstance(op) && !(op instanceof ActivationOperation)) {
                return codec;
            }
        }

        return null;
    }
}

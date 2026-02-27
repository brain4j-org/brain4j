package org.brain4j.core.importing;

import org.brain4j.core.importing.format.GeneralRegistry;
import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.newimpl.ConvLayer;
import org.brain4j.core.layer.newimpl.DenseLayer;
import org.brain4j.core.layer.newimpl.InputLayer;
import org.brain4j.core.layer.newimpl.NormLayer;
import org.brain4j.math.loss.LossFunction;
import org.brain4j.math.loss.impl.BinaryCrossEntropy;
import org.brain4j.math.loss.impl.CrossEntropy;
import org.brain4j.math.loss.impl.MeanAbsoluteError;
import org.brain4j.math.loss.impl.MeanSquaredError;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.optimizer.impl.Adam;
import org.brain4j.core.training.optimizer.impl.AdamW;
import org.brain4j.core.training.optimizer.impl.GradientDescent;
import org.brain4j.core.training.optimizer.impl.Lion;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.training.updater.impl.NormalUpdater;
import org.brain4j.core.training.updater.impl.StochasticUpdater;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.impl.*;
import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.clipper.impl.HardClipper;
import org.brain4j.math.clipper.impl.L2Clipper;
import org.brain4j.math.clipper.impl.NoClipper;
import org.brain4j.math.scaler.FeatureScaler;
import org.brain4j.math.scaler.impl.MinMaxScaler;
import org.brain4j.math.scaler.impl.ZScoreScaler;
import org.brain4j.math.tensor.autograd.Operation;
import org.brain4j.math.tensor.autograd.impl.*;
import org.brain4j.math.weightsinit.WeightInit;
import org.brain4j.math.weightsinit.impl.*;

public class Registries {
    
    public static final GeneralRegistry<Operation, NodeProto> ONNX_OPERATIONS_REGISTRY = new GeneralRegistry<>();
    public static final GeneralRegistry<Optimizer, Object> OPTIMIZERS_REGISTRY = new GeneralRegistry<>();
    public static final GeneralRegistry<LossFunction, Object> LOSS_FUNCTION_REGISTRY = new GeneralRegistry<>();
    public static final GeneralRegistry<Updater, Object> UPDATERS_REGISTRY = new GeneralRegistry<>();
    public static final GeneralRegistry<GradientClipper, Object> CLIPPERS_REGISTRY = new GeneralRegistry<>();
    public static final GeneralRegistry<Activation, Object> ACTIVATION_REGISTRY = new GeneralRegistry<>();
    public static final GeneralRegistry<Layer, Object> LAYER_REGISTRY = new GeneralRegistry<>();
    public static final GeneralRegistry<FeatureScaler, Object> SCALER_REGISTRY = new GeneralRegistry<>();
    public static final GeneralRegistry<WeightInit, Object> WEIGHT_INIT_REGISTRY = new GeneralRegistry<>();

    static {
        ONNX_OPERATIONS_REGISTRY.register("Add", AddOperation.class);
        ONNX_OPERATIONS_REGISTRY.register("Add", AddOperation.class);
        ONNX_OPERATIONS_REGISTRY.register("Sub", SubOperation.class);
        ONNX_OPERATIONS_REGISTRY.register("Mul", MulOperation.class);
        ONNX_OPERATIONS_REGISTRY.register("Div", DivOperation.class);
        ONNX_OPERATIONS_REGISTRY.register("Gemm", GemmOperation.class);
        ONNX_OPERATIONS_REGISTRY.register("MatMul", MatMulOperation.class);

        ONNX_OPERATIONS_REGISTRY.register("Concat", (node) -> {
            int axis = (int) node.getAttribute(0).getI();
            return new ConcatOperation(axis);
        });
        ONNX_OPERATIONS_REGISTRY.register("Squeeze", (node) -> {
            int dimension = (int) node.getAttribute(0).getI();
            return new SqueezeOperation(dimension);
        });
        ONNX_OPERATIONS_REGISTRY.register("Concat", ConcatOperation.class);
        ONNX_OPERATIONS_REGISTRY.register("Relu", (x) -> new ActivationOperation(new ReLU()));
        ONNX_OPERATIONS_REGISTRY.register("Relu", ActivationOperation.class);
        ONNX_OPERATIONS_REGISTRY.register("Sigmoid", (x) -> new ActivationOperation(new Sigmoid()));
        ONNX_OPERATIONS_REGISTRY.register("Sigmoid", ActivationOperation.class);
        ONNX_OPERATIONS_REGISTRY.register("Tanh", (x) -> new ActivationOperation(new Tanh()));
        ONNX_OPERATIONS_REGISTRY.register("Tanh", ActivationOperation.class);
        ONNX_OPERATIONS_REGISTRY.register("LeakyRelu", (x) -> new ActivationOperation(new LeakyReLU()));
        ONNX_OPERATIONS_REGISTRY.register("LeakyRelu", ActivationOperation.class);
        ONNX_OPERATIONS_REGISTRY.register("Gelu", (x) -> new ActivationOperation(new GELU()));
        ONNX_OPERATIONS_REGISTRY.register("Gelu", ActivationOperation.class);
        ONNX_OPERATIONS_REGISTRY.register("Softmax", (x) -> new ActivationOperation(new Softmax()));
        ONNX_OPERATIONS_REGISTRY.register("Softmax", ActivationOperation.class);
        ONNX_OPERATIONS_REGISTRY.register("LayerNormalization", (node) -> {
            float epsilon = node.getAttribute(0).getF();
            return new LayerNormOperation( epsilon);
        });
        
        OPTIMIZERS_REGISTRY.register("adam", Adam.class);
        OPTIMIZERS_REGISTRY.register("adamw", AdamW.class);
        OPTIMIZERS_REGISTRY.register("gradient_descent", GradientDescent.class);
        OPTIMIZERS_REGISTRY.register("lion", Lion.class);
        
        UPDATERS_REGISTRY.register("stochastic", StochasticUpdater.class);
        UPDATERS_REGISTRY.register("normal", NormalUpdater.class);
        
        LOSS_FUNCTION_REGISTRY.register("binary_cross_entropy", BinaryCrossEntropy.class);
        LOSS_FUNCTION_REGISTRY.register("cross_entropy", CrossEntropy.class);
        LOSS_FUNCTION_REGISTRY.register("mean_absolute_error", MeanAbsoluteError.class);
        LOSS_FUNCTION_REGISTRY.register("mean_squared_error", MeanSquaredError.class);
        
        CLIPPERS_REGISTRY.register("none", NoClipper.class);
        CLIPPERS_REGISTRY.register("clamp", HardClipper.class);
        CLIPPERS_REGISTRY.register("l2", L2Clipper.class);
        
        ACTIVATION_REGISTRY.register("elu", ELU.class);
        ACTIVATION_REGISTRY.register("gelu", GELU.class);
        ACTIVATION_REGISTRY.register("leaky_relu", LeakyReLU.class);
        ACTIVATION_REGISTRY.register("linear", Linear.class);
        ACTIVATION_REGISTRY.register("mish", Mish.class);
        ACTIVATION_REGISTRY.register("relu", ReLU.class);
        ACTIVATION_REGISTRY.register("sigmoid", Sigmoid.class);
        ACTIVATION_REGISTRY.register("softmax", Softmax.class);
        ACTIVATION_REGISTRY.register("softplus", SoftPlus.class);
        ACTIVATION_REGISTRY.register("swish", Swish.class);
        ACTIVATION_REGISTRY.register("tanh", Tanh.class);

        LAYER_REGISTRY.register("input", InputLayer.class);
        LAYER_REGISTRY.register("dense", DenseLayer.class);
//        LAYER_REGISTRY.register("dropout", DropoutLayer.class);
//        LAYER_REGISTRY.register("lstm", LSTMLayer.class);
        LAYER_REGISTRY.register("norm", NormLayer.class);
//        LAYER_REGISTRY.register("recurrent", RecurrentLayer.class);
        LAYER_REGISTRY.register("conv_2d", ConvLayer.class);
//
//        LAYER_REGISTRY.register("embedding", EmbeddingLayer.class);
//        LAYER_REGISTRY.register("positional_encode", PosEncodeLayer.class);
//        LAYER_REGISTRY.register("transformer_decoder", TransformerDecoder.class);
//        LAYER_REGISTRY.register("transformer_encoder", TransformerEncoder.class);
//
//        LAYER_REGISTRY.register("activation", ActivationLayer.class);
//        LAYER_REGISTRY.register("reshape", ReshapeLayer.class);
//        LAYER_REGISTRY.register("slice", SliceLayer.class);
//        LAYER_REGISTRY.register("squeeze", SqueezeLayer.class);

        SCALER_REGISTRY.register("z_score", ZScoreScaler.class);
        SCALER_REGISTRY.register("min_max", MinMaxScaler.class);

        WEIGHT_INIT_REGISTRY.register("normal", NormalInit.class);
        WEIGHT_INIT_REGISTRY.register("normal_he", NormalHeInit.class);
        WEIGHT_INIT_REGISTRY.register("normal_xavier", NormalXavierInit.class);
        WEIGHT_INIT_REGISTRY.register("uniform_he", UniformHeInit.class);
        WEIGHT_INIT_REGISTRY.register("uniform_xavier", UniformXavierInit.class);
        WEIGHT_INIT_REGISTRY.register("lecun", LeCunInit.class);

    }
}

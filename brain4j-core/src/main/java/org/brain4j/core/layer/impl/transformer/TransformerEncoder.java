package org.brain4j.core.layer.impl.transformer;

import com.google.gson.JsonObject;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.DropoutLayer;
import org.brain4j.core.layer.impl.NormLayer;
import org.brain4j.core.layer.impl.RMSNormLayer;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.silicon.SiliconDevice;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.weightsinit.impl.UniformXavierInit;

import java.util.Arrays;
import java.util.Map;
import java.util.random.RandomGenerator;

/**
 * Implements a single encoder block of the Transformer architecture,
 * as introduced in the paper "Attention is All You Need".
 *
 * <p>It includes a multi-head self-attention layer, along with
 * residual connections, layer normalization, feed-forward projection
 * and dropout for regularization.
 *
 * <p>The expected input shape is a 3D tensor of shape {@code [batch, seq_len, embedding_dim]}.
 * The output has the same shape.
 *
 * @see TransformerDecoder
 * @see MultiHeadAttention
 * @see DenseLayer
 * @see DropoutLayer
 * @see NormLayer
 * @author xEcho1337
 */
public class TransformerEncoder extends Layer {

    protected DenseLayer upProjection;
    protected DenseLayer gateProjection;
    protected DenseLayer downProjection;
    protected Layer normalizer1;
    protected Layer normalizer2;
    protected DropoutLayer dropout;
    protected MultiHeadAttention attention;
    protected NormType normType;

    protected int numHeads;
    protected int embeddingDim;
    protected double dropoutRate;
    protected boolean useGating;
    protected boolean attnQkvHasBias;
    protected boolean attnOutHasBias;

    protected TransformerEncoder() {
    }

    /**
     * Constructs a new encoder block with the specified parameters.
     * @param numHeads the amount of heads in the attention block
     * @param embeddingDim the embedding dimension of the input
     * @param dropout the dropout used when training
     */
    public TransformerEncoder(int numHeads, int embeddingDim, double dropout) {
        this(numHeads, embeddingDim, dropout, Activations.GELU);
    }

    /**
     * Constructs a new encoder block with the specified parameters.
     * @param numHeads the amount of heads in the attention block
     * @param embeddingDim the embedding dimension of the input
     * @param dropout the dropout used when training
     * @param activation the activation used in the projection
     */
    public TransformerEncoder(int numHeads, int embeddingDim, double dropout, Activations activation) {
        this(numHeads, embeddingDim, 4 * embeddingDim, dropout, false, false, false, activation.function(), NormType.LAYER_NORM);
    }

    /**
     * Constructs a new encoder block with the specified parameters.
     * @param numHeads the amount of heads in the attention block
     * @param embeddingDim the embedding dimension of the input
     * @param projDim the dimension of the projected embedding
     * @param dropout the dropout used when training
     * @param activation the activation used in the projection
     * @param attnQkvHasBias whether the QKV projection matrix should have a bias
     * @param attnOutHasBias whether the out projection matrix should have a bias
     */
    public TransformerEncoder(int numHeads, int embeddingDim, int projDim, double dropout, boolean useGating,
                              boolean attnQkvHasBias, boolean attnOutHasBias, Activation activation, NormType normType) {
        this.numHeads = numHeads;
        this.embeddingDim = embeddingDim;
        this.dropoutRate = dropout;
        this.activation = activation;
        this.normType = normType;
        this.useGating = useGating;
        this.attnQkvHasBias = attnQkvHasBias;
        this.attnOutHasBias = attnOutHasBias;

        this.dropout = new DropoutLayer(dropout);
        this.weightInit = new UniformXavierInit();
        this.normalizer1 = createNormLayer();
        this.normalizer2 = createNormLayer();
        this.upProjection = new DenseLayer(projDim);
        this.downProjection = new DenseLayer(embeddingDim);
        this.attention = createAttention(numHeads, embeddingDim);

        if (useGating) this.gateProjection = new DenseLayer(projDim);
        
        attention.setAttnQkvBias(attnQkvHasBias);
        attention.setAttnOutBias(attnOutHasBias);
    }

    public Layer createNormLayer() {
        return switch (normType) {
            case LAYER_NORM -> new NormLayer();
            case RMS_NORM -> new RMSNormLayer();
        };
    }

    public MultiHeadAttention createAttention(int heads, int embeddingDim) {
        return new MultiHeadAttention(clipper, heads, embeddingDim);
    }
    
    @Override
    public void resetGrad() {
        normalizer1.resetGrad();
        normalizer2.resetGrad();
        upProjection.resetGrad();
        downProjection.resetGrad();
        attention.resetGrad();

        if (useGating) {
            gateProjection.resetGrad();
        }
    }
    
    @Override
    public void connect(Layer previous) {
        normalizer1.connect(this);
        normalizer2.connect(this);
        upProjection.connect(this);
        downProjection.connect(upProjection);
        attention.connect(previous);

        if (useGating) gateProjection.connect(this);
        
        if (frozen) {
            normalizer1.freeze();
            normalizer2.freeze();
            upProjection.freeze();
            downProjection.freeze();
            attention.freeze();
            
            if (useGating) gateProjection.freeze();
        }

    }

    @Override
    public void initWeights(RandomGenerator generator, int input, int output) {
        normalizer1.initWeights(generator, embeddingDim, embeddingDim);
        normalizer2.initWeights(generator, embeddingDim, embeddingDim);
        upProjection.initWeights(generator, embeddingDim, embeddingDim * 4);
        downProjection.initWeights(generator, embeddingDim * 4, embeddingDim);
        attention.initWeights(generator, embeddingDim, embeddingDim);

        if (useGating) {
            gateProjection.initWeights(generator, embeddingDim, embeddingDim * 4);
        }
    }

    @Override
    public void toDevice(SiliconDevice device) {
        normalizer1.toDevice(device);
        normalizer2.toDevice(device);
        upProjection.toDevice(device);
        downProjection.toDevice(device);
        attention.toDevice(device);

        if (useGating) {
            gateProjection.toDevice(device);
        }
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        checkInputLength(1, inputs);
        Tensor input = inputs[0];

        if (input.rank() != 3) {
            throw Commons.illegalArgument("Input must have shape [batch, seq_length, dimension]! Got: ",
                Arrays.toString(input.shape()));
        }

        Tensor attended = attention.forward(cache, input);

        if (cache.isTraining()) {
            attended = dropout.forward(cache, attended);
        }

        Tensor added = attended.addGrad(input);
        Tensor normalized = normalizer1.forward(cache, added);

        Tensor upProjected, downProjected;
        if (gateProjection != null) {
            Tensor gate = gateProjection.forward(cache, normalized).activateGrad(activation);
            Tensor up = upProjection.forward(cache, normalized);
            Tensor prod = gate.mul(up);
            downProjected = downProjection.forward(cache, prod);
        } else {
            upProjected = upProjection.forward(cache, normalized).activateGrad(activation);
            downProjected = downProjection.forward(cache, upProjected);
        }

        if (cache.isTraining()) {
            downProjected = dropout.forward(cache, downProjected);
        }

        Tensor added2 = downProjected.addGrad(normalized);
        normalized = normalizer2.forward(cache, added2);

        cache.rememberOutput(this, normalized);
        
        return new Tensor[] { normalized };
    }
    
    @Override
    public void backward(StatesCache cache, Updater updater, Optimizer optimizer) {
        normalizer2.backward(cache, updater, optimizer);
        downProjection.backward(cache, updater, optimizer);
        upProjection.backward(cache, updater, optimizer);
        normalizer1.backward(cache, updater, optimizer);
        attention.backward(cache, updater, optimizer);

        if (useGating) {
            gateProjection.backward(cache, updater, optimizer);
        }
    }
    
    @Override
    public void loadWeights(Map<String, Tensor> mappedWeights) {
        this.upProjection = new DenseLayer(0);
        this.downProjection = new DenseLayer(0);
        this.dropout = new DropoutLayer(dropoutRate);
        this.normalizer1 = createNormLayer();
        this.normalizer2 = createNormLayer();
        this.attention = createAttention(numHeads, embeddingDim);

        upProjection.setWeights(mappedWeights.get("up_proj.weights"));
        upProjection.setBias(mappedWeights.get("up_proj.bias"));
        downProjection.setWeights(mappedWeights.get("down_proj.weights"));
        downProjection.setBias(mappedWeights.get("down_proj.bias"));

        normalizer1.setWeights(mappedWeights.get("norm_1.weights"));
        normalizer2.setWeights(mappedWeights.get("norm_2.weights"));

        if (normType == NormType.LAYER_NORM) {
            normalizer1.setBias(mappedWeights.get("norm_1.bias"));
            normalizer2.setBias(mappedWeights.get("norm_2.bias"));
        }

        if (useGating) {
            gateProjection = new DenseLayer(0);
            gateProjection.setWeights(mappedWeights.get("gate_proj.weights"));
            gateProjection.setBias(mappedWeights.get("gate_proj.bias"));
        }
        
        attention.setWeights(mappedWeights.get("attention.weights"));
        attention.setOutProj(mappedWeights.get("attention.out_proj"));
        
        if (attnQkvHasBias) attention.setBias(mappedWeights.get("attention.bias"));
        if (attnOutHasBias) attention.setOutBias(mappedWeights.get("attention.out_bias"));
        
        attention.setAttnOutBias(attnOutHasBias);
        attention.setAttnQkvBias(attnQkvHasBias);
    }
    
    @Override
    public Layer freeze() {
        upProjection.freeze();
        downProjection.freeze();
        normalizer1.freeze();
        normalizer2.freeze();
        attention.freeze();
        
        if (gateProjection != null) gateProjection.freeze();
        
        return super.freeze();
    }
    
    @Override
    public Layer unfreeze() {
        upProjection.unfreeze();
        downProjection.unfreeze();
        normalizer1.unfreeze();
        normalizer2.unfreeze();
        attention.unfreeze();
        
        if (gateProjection != null) gateProjection.unfreeze();
        
        return super.unfreeze();
    }
    
    @Override
    public void serialize(JsonObject object) {
        object.addProperty("norm_type", normType.name().toLowerCase());
        object.addProperty("dropout", dropoutRate);
        object.addProperty("heads", numHeads);
        object.addProperty("embedding_dim", embeddingDim);
        object.addProperty("use_gating", useGating);
        object.addProperty("attn_qkv_has_bias", attnQkvHasBias);
        object.addProperty("attn_out_has_bias", attnOutHasBias);
    }
    
    @Override
    public void deserialize(JsonObject object) {
        this.useGating = object.get("use_gating").getAsBoolean();
        this.attnQkvHasBias = object.get("attn_qkv_has_bias").getAsBoolean();
        this.attnQkvHasBias = object.get("attn_out_has_bias").getAsBoolean();
        this.normType = NormType.valueOf(object.get("norm_type").getAsString().toUpperCase());
        this.dropoutRate = object.get("dropout").getAsDouble();
        this.numHeads = object.get("heads").getAsInt();
        this.embeddingDim = object.get("embedding_dim").getAsInt();
    }
    
    @Override
    public int size() {
        return embeddingDim;
    }
    
    @Override
    public int totalBiases() {
        int total = upProjection.totalBiases()
            + downProjection.totalBiases()
            + normalizer1.totalBiases()
            + normalizer2.totalBiases();

        if (useGating) total += gateProjection.totalBiases();
        return total;
    }
    
    @Override
    public int totalWeights() {
        int total = upProjection.totalWeights()
            + downProjection.totalWeights()
            + normalizer1.totalWeights()
            + normalizer2.totalWeights()
            + attention.totalWeights();

        if (useGating) total += gateProjection.totalWeights();
        return total;
    }
    
    @Override
    public Map<String, Tensor> weightsMap() {
        var result = super.weightsMap();
        
        result.put("up_proj.weights", upProjection.getWeights());
        result.put("up_proj.bias", upProjection.getBias());
        result.put("down_proj.weights", downProjection.getWeights());
        result.put("down_proj.bias", downProjection.getBias());

        result.put("norm_1.weights", normalizer1.getWeights());
        result.put("norm_2.weights", normalizer2.getWeights());

        if (normType == NormType.LAYER_NORM) {
            result.put("norm_1.bias", normalizer1.getBias());
            result.put("norm_2.bias", normalizer2.getBias());
        }

        if (useGating) {
            result.put("gate_proj.weights", gateProjection.getWeights());
            result.put("gate_proj.bias", gateProjection.getBias());
        }

        result.put("attention.weights", attention.getWeights());
        result.put("attention.out_proj", attention.getOutProj());
        
        if (attention.hasAttnQkvBias()) result.put("attention.bias", attention.getBias());
        if (attention.hasAttnOutBias()) result.put("attention.out_bias", attention.getOutBias());

        return result;
    }
    
    public DenseLayer getUpProjection() {
        return upProjection;
    }
    
    public TransformerEncoder setUpProjection(DenseLayer upProjection) {
        this.upProjection = upProjection;
        return this;
    }
    
    public DenseLayer getGateProjection() {
        return gateProjection;
    }
    
    public TransformerEncoder setGateProjection(DenseLayer gateProjection) {
        this.gateProjection = gateProjection;
        return this;
    }
    
    public DenseLayer getDownProjection() {
        return downProjection;
    }
    
    public TransformerEncoder setDownProjection(DenseLayer downProjection) {
        this.downProjection = downProjection;
        return this;
    }
    
    public Layer getNormalizer1() {
        return normalizer1;
    }
    
    public TransformerEncoder setNormalizer1(Layer normalizer1) {
        this.normalizer1 = normalizer1;
        return this;
    }
    
    public Layer getNormalizer2() {
        return normalizer2;
    }
    
    public TransformerEncoder setNormalizer2(Layer normalizer2) {
        this.normalizer2 = normalizer2;
        return this;
    }
    
    public DropoutLayer getDropout() {
        return dropout;
    }
    
    public TransformerEncoder setDropout(DropoutLayer dropout) {
        this.dropout = dropout;
        return this;
    }
    
    public MultiHeadAttention getAttention() {
        return attention;
    }
    
    public TransformerEncoder setAttention(MultiHeadAttention attention) {
        this.attention = attention;
        return this;
    }
    
    public NormType getNormType() {
        return normType;
    }
    
    public TransformerEncoder setNormType(NormType normType) {
        this.normType = normType;
        return this;
    }
    
    public int getNumHeads() {
        return numHeads;
    }
    
    public TransformerEncoder setNumHeads(int numHeads) {
        this.numHeads = numHeads;
        return this;
    }
    
    public int getEmbeddingDim() {
        return embeddingDim;
    }
    
    public TransformerEncoder setEmbeddingDim(int embeddingDim) {
        this.embeddingDim = embeddingDim;
        return this;
    }
    
    public double getDropoutRate() {
        return dropoutRate;
    }
    
    public TransformerEncoder setDropoutRate(double dropoutRate) {
        this.dropoutRate = dropoutRate;
        return this;
    }
    
    public boolean isUseGating() {
        return useGating;
    }
    
    public TransformerEncoder setUseGating(boolean useGating) {
        this.useGating = useGating;
        return this;
    }
    
    public boolean isAttnQkvHasBias() {
        return attnQkvHasBias;
    }
    
    public TransformerEncoder setAttnQkvHasBias(boolean attnQkvHasBias) {
        this.attnQkvHasBias = attnQkvHasBias;
        return this;
    }
    
    public boolean isAttnOutHasBias() {
        return attnOutHasBias;
    }
    
    public TransformerEncoder setAttnOutHasBias(boolean attnOutHasBias) {
        this.attnOutHasBias = attnOutHasBias;
        return this;
    }
}

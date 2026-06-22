package org.brain4j.transformers.core.architecture.impl;

import com.google.gson.JsonObject;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.NormLayer;
import org.brain4j.core.layer.impl.transformer.EmbeddingLayer;
import org.brain4j.core.layer.impl.transformer.Transformer;
import org.brain4j.core.layer.impl.transformer.attention.MaskedMultiHeadAttention;
import org.brain4j.core.layer.impl.transformer.attention.MultiHeadAttention;
import org.brain4j.core.layer.impl.transformer.PosEncodeLayer;
import org.brain4j.core.layer.impl.InputLayer;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.ModelSpecs;
import org.brain4j.transformers.core.architecture.ArchitectureAdapter;
import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.commons.Range;
import org.brain4j.math.activation.impl.GELU;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.random.RandomGenerator;

public class GPT2Adapter implements ArchitectureAdapter {
    @Override
    public boolean supports(String modelType) {
        return modelType.equals("gpt2");
    }

    private static Tensor findContaining(String text, Map<String, Tensor> weights) {
        Map.Entry<String, Tensor> entry = weights.entrySet()
            .stream()
            .filter(x -> x.getKey().contains(text))
            .findFirst()
            .orElse(null);

        if (entry == null) return null;

        return entry.getValue();
    }
    
    @Override
    public Model buildModel(JsonObject config, Map<String, Tensor> weights) {
        int layers = config.get("n_layer").getAsInt();
        int heads = config.get("n_head").getAsInt();
        int embeddingDim = config.get("n_embd").getAsInt();
        int context = config.get("n_ctx").getAsInt();
        int vocabSize = config.get("vocab_size").getAsInt();
        
        ModelSpecs specs = ModelSpecs.of();

        Tensor embedding = findContaining("wte.weight", weights); // embedding  -> [vocab, dim]
        Tensor posEncode = findContaining("wpe.weight", weights); // pos encode -> [length, dim]

        if (embedding == null) throw Commons.illegalState("Unable to find embeddings!");
        if (posEncode == null) throw Commons.illegalState("Unable to find positional encoding!");

        EmbeddingLayer embeddingLayer = new EmbeddingLayer(vocabSize, embeddingDim);
        DenseLayer vocabLayer = new DenseLayer(0);
        PosEncodeLayer posEncodeLayer = new PosEncodeLayer(context, embeddingDim);

        embeddingLayer.registerParam("weights", embedding);
        embeddingLayer.registerParam("bias", Tensors.zeros(embedding.elements()));

        vocabLayer.registerParam("weights", embedding.transpose());
        vocabLayer.registerParam("bias", Tensors.zeros(embedding.elements()));

        posEncodeLayer.setWeights(posEncode);

        specs.add(new InputLayer(Shape.of(-1)).freeze());
        specs.add(embeddingLayer.freeze());
        specs.add(posEncodeLayer.freeze());
        
        for (int i = 0; i < layers; i++) {
            String prefix = String.format("h.%s.", i);
            GPT2Decoder decoder = new GPT2Decoder(embeddingDim, heads, 0.0);
            decoder.loadWeights(prefix, weights);

            specs.add(decoder.freeze());
        }
        
        TokenSelectionLayer selectionLayer = new TokenSelectionLayer();
        
        NormLayer normLayer = new NormLayer();
        Tensor lnGamma = findContaining("ln_f.weight", weights);
        Tensor lnBeta = findContaining("ln_f.bias", weights);
        
        normLayer.registerParam("weights", lnGamma);
        normLayer.registerParam("bias", lnBeta);
        
        specs.add(normLayer.freeze());
        specs.add(vocabLayer.freeze());
        specs.add(selectionLayer.freeze());

        return specs.compile();
    }

    private static class GPT2Decoder extends Transformer.Decoder {

        private GPT2Decoder(int embeddingDim, int heads, double dropout) {
            super(new Transformer.Config(
                embeddingDim,
                4 * embeddingDim,
                heads,
                dropout,
                false,
                new GELU(),
                NormLayer::new
            ));
        }

        @Override
        protected Layer getAttention() {
            var attentionConfig = new MultiHeadAttention.Config(
                config.embedDim(),
                config.heads(),
                true,
                true
            );

            return new MaskedMultiHeadAttention(attentionConfig);
        }

        private void loadWeights(String prefix, Map<String, Tensor> weights) {
            NormLayer firstNorm = (NormLayer) norm1;
            NormLayer secondNorm = (NormLayer) norm2;

            firstNorm.registerParam("weights", findContaining(prefix + "ln_1.weight", weights));
            firstNorm.registerParam("bias", findContaining(prefix + "ln_1.bias", weights));
            secondNorm.registerParam("weights", findContaining(prefix + "ln_2.weight", weights));
            secondNorm.registerParam("bias", findContaining(prefix + "ln_2.bias", weights));

            upProj.registerParam("weights", findContaining(prefix + "mlp.c_fc.weight", weights));
            upProj.registerParam("bias", findContaining(prefix + "mlp.c_fc.bias", weights));
            downProj.registerParam("weights", findContaining(prefix + "mlp.c_proj.weight", weights));
            downProj.registerParam("bias", findContaining(prefix + "mlp.c_proj.bias", weights));

            MultiHeadAttention attentionLayer = (MultiHeadAttention) attention;

            attentionLayer.registerParam("weights", findContaining(prefix + "attn.c_attn.weight", weights));
            attentionLayer.registerParam("bias", findContaining(prefix + "attn.c_attn.bias", weights));
            attentionLayer.registerParam("out_proj", findContaining(prefix + "attn.c_proj.weight", weights));
            attentionLayer.registerParam("out_bias", findContaining(prefix + "attn.c_proj.bias", weights));
        }
    }
    
    private static class TokenSelectionLayer extends Layer {

        public record Config() {}

        protected Config config = new Config();

        @Override
        public void build(List<Shape> inputShapes) {
        }

        @Override
        public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
        }

        @Override
        public List<Shape> inferOutputShapes(List<Shape> inputShapes) {
            return new ArrayList<>(inputShapes);
        }

        @Override
        public Tensor[] forward(StatesCache cache, Tensor... inputs) {
            if (cache.isTraining()) return inputs;

            Tensor input = inputs[0]; // [batch, seq_len, dim]
            int seqLength = input.shapeAt(1);

            Range[] ranges = {
                Range.all(),
                Range.point(seqLength - 1),
                Range.all()
            };

            return new Tensor[] { input.slice(ranges).squeezeGrad(1) };
        }

        @Override
        public Layer copy() {
            return new TokenSelectionLayer();
        }

        public Config config() {
            return config;
        }
    }
}

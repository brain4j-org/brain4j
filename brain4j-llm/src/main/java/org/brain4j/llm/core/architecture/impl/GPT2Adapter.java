package org.brain4j.llm.core.architecture.impl;

import com.google.gson.JsonObject;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.NormLayer;
import org.brain4j.core.layer.impl.transformer.EmbeddingLayer;
import org.brain4j.core.layer.impl.transformer.MultiHeadAttention;
import org.brain4j.core.layer.impl.transformer.PosEncodeLayer;
import org.brain4j.core.layer.impl.transformer.TransformerDecoder;
import org.brain4j.core.layer.impl.InputLayer;
import org.brain4j.core.layer.old.OldLayer;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.ModelSpecs;
import org.brain4j.llm.core.architecture.ArchitectureAdapter;
import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.commons.Range;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.random.RandomGenerator;

public class GPT2Adapter implements ArchitectureAdapter {
    @Override
    public boolean supports(String modelType) {
        return modelType.equals("gpt2");
    }

    private Tensor findContaining(String text, Map<String, Tensor> weights) {
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
            TransformerDecoder decoder = new TransformerDecoder(heads, embeddingDim, 0.0);
            
            NormLayer norm1 = (NormLayer) decoder.normalizer1();
            NormLayer norm2 = (NormLayer) decoder.normalizer2();
            DenseLayer upProj = decoder.upProjection();
            DenseLayer downProj = decoder.downProjection();
            
            Tensor ln1Gamma = findContaining(prefix + "ln_1.weight", weights);
            Tensor ln1Beta = findContaining(prefix + "ln_1.bias", weights);
            Tensor ln2Gamma = findContaining(prefix + "ln_2.weight", weights);
            Tensor ln2Beta = findContaining(prefix + "ln_2.bias", weights);
            
            norm1.registerParam("weights", ln1Gamma);
            norm1.registerParam("bias", ln1Beta);
            norm2.registerParam("weights", ln2Gamma);
            norm2.registerParam("bias", ln2Beta);
            
            Tensor upProjWeight = findContaining(prefix + "mlp.c_fc.weight", weights);
            Tensor upProjBias = findContaining(prefix + "mlp.c_fc.bias", weights);
            Tensor downProjWeight = findContaining(prefix + "mlp.c_proj.weight", weights);
            Tensor downProjBias = findContaining(prefix + "mlp.c_proj.bias", weights);
            
            upProj.registerParam("weights", upProjWeight);
            upProj.registerParam("bias", upProjBias);
            downProj.registerParam("weights", downProjWeight);
            downProj.registerParam("bias", downProjBias);
            
            Tensor attnWeight = findContaining(prefix + "attn.c_attn.weight", weights);
            Tensor attnBias = findContaining(prefix + "attn.c_attn.bias", weights);
            Tensor attnOutWeight = findContaining(prefix + "attn.c_proj.weight", weights);
            Tensor attnOutBias = findContaining(prefix + "attn.c_proj.bias", weights);
            
            MultiHeadAttention attention = decoder.attention();
            attention.attnQkvHasBias(true);
            attention.attnOutHasBias(true);
            
            attention.registerParam("weights", attnWeight);
            attention.registerParam("bias", attnBias);
            attention.registerParam("out_proj", attnOutWeight);
            attention.registerParam("out_bias", attnOutBias);

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
    
    private static class TokenSelectionLayer extends Layer {
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
    }
}

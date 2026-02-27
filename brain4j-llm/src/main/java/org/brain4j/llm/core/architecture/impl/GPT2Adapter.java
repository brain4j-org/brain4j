package org.brain4j.llm.core.architecture.impl;

import com.google.gson.JsonObject;
import org.brain4j.core.layer.OldLayer;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.NormLayer;
import org.brain4j.core.layer.impl.transformer.EmbeddingLayer;
import org.brain4j.core.layer.impl.transformer.MultiHeadAttention;
import org.brain4j.core.layer.impl.transformer.PosEncodeLayer;
import org.brain4j.core.layer.impl.transformer.TransformerDecoder;
import org.brain4j.core.layer.impl.utility.InputLayer;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.ModelSpecs;
import org.brain4j.llm.core.architecture.ArchitectureAdapter;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.commons.Range;

import java.util.Map;

public class GPT2Adapter implements ArchitectureAdapter {
    @Override
    public boolean supports(String modelType) {
        return modelType.equals("gpt2");
    }
    
    @Override
    public Model buildModel(JsonObject config, Map<String, Tensor> weights) {
        int layers = config.get("n_layer").getAsInt();
        int heads = config.get("n_head").getAsInt();
        int embeddingDim = config.get("n_embd").getAsInt();
        int context = config.get("n_ctx").getAsInt();
        int vocabSize = config.get("vocab_size").getAsInt();
        
        ModelSpecs specs = ModelSpecs.of();
        
        Tensor embedding = weights.get("wte.weight"); // embedding  -> [vocab, dim]
        Tensor posEncode = weights.get("wpe.weight"); // pos encode -> [length, dim]
        
        EmbeddingLayer embeddingLayer = new EmbeddingLayer(vocabSize, embeddingDim);
        DenseLayer vocabLayer = new DenseLayer(0);
        PosEncodeLayer posEncodeLayer = new PosEncodeLayer(context, embeddingDim);

        embeddingLayer.setWeights(embedding);
        vocabLayer.setWeights(embedding.transpose());
        posEncodeLayer.setWeights(posEncode);
        
        specs.add(new InputLayer(-1).freeze());
        specs.add(embeddingLayer.freeze());
        specs.add(posEncodeLayer.freeze());
        
        for (int i = 0; i < layers; i++) {
            String prefix = String.format("h.%s.", i);
            TransformerDecoder decoder = new TransformerDecoder(heads, embeddingDim, 0.0);
            
            NormLayer norm1 = (NormLayer) decoder.getNormalizer1();
            NormLayer norm2 = (NormLayer) decoder.getNormalizer2();
            DenseLayer upProj = decoder.getUpProjection();
            DenseLayer downProj = decoder.getDownProjection();
            
            Tensor ln1Gamma = weights.get(prefix + "ln_1.weight");
            Tensor ln1Beta = weights.get(prefix + "ln_1.bias");
            Tensor ln2Gamma = weights.get(prefix + "ln_2.weight");
            Tensor ln2Beta = weights.get(prefix + "ln_2.bias");
            
            norm1.setWeights(ln1Gamma);
            norm1.setBias(ln1Beta);
            norm2.setWeights(ln2Gamma);
            norm2.setBias(ln2Beta);
            
            Tensor upProjWeight = weights.get(prefix + "mlp.c_fc.weight");
            Tensor upProjBias = weights.get(prefix + "mlp.c_fc.bias");
            Tensor downProjWeight = weights.get(prefix + "mlp.c_proj.weight");
            Tensor downProjBias = weights.get(prefix + "mlp.c_proj.bias");
            
            upProj.setWeights(upProjWeight);
            upProj.setBias(upProjBias);
            downProj.setWeights(downProjWeight);
            downProj.setBias(downProjBias);
            
            Tensor attnWeight = weights.get(prefix + "attn.c_attn.weight");
            Tensor attnBias = weights.get(prefix + "attn.c_attn.bias");
            Tensor attnOutWeight = weights.get(prefix + "attn.c_proj.weight");
            Tensor attnOutBias = weights.get(prefix + "attn.c_proj.bias");
            
            MultiHeadAttention attention = decoder.getAttention();
            attention.setAttnQkvBias(true);
            attention.setAttnOutBias(true);
            
            attention.setWeights(attnWeight);
            attention.setBias(attnBias);
            attention.setOutProj(attnOutWeight);
            attention.setOutBias(attnOutBias);
            
            specs.add(decoder.freeze());
        }
        
        TokenSelectionLayer selectionLayer = new TokenSelectionLayer();
        
        NormLayer normLayer = new NormLayer();
        Tensor lnGamma = weights.get("ln_f.weight");
        Tensor lnBeta = weights.get("ln_f.bias");
        
        normLayer.setWeights(lnGamma);
        normLayer.setBias(lnBeta);
        
        specs.add(normLayer.freeze());
        specs.add(selectionLayer.freeze());
        specs.add(vocabLayer.freeze());
        
        return specs.compile();
    }
    
    static class TokenSelectionLayer extends OldLayer {
        @Override
        public Tensor[] forward(StatesCache cache, Tensor... inputs) {
            if (cache.isTraining()) return inputs;
            
            Tensor input = inputs[0]; // [batch, seq_len, dim]
            int seqLength = input.shapeAt(1);
            
            return new Tensor[] { input.slice(Range.all(), Range.point(seqLength - 1), Range.all()).squeezeGrad(1) };
        }
        
        @Override
        public int size() {
            return 0;
        }
    }
}

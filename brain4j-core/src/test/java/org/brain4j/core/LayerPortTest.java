package org.brain4j.core;

import org.brain4j.core.layer.newimpl.InputLayer;
import org.brain4j.core.layer.newimpl.MaxPoolLayer;
import org.brain4j.core.layer.newimpl.transformer.EmbeddingLayer;
import org.brain4j.core.layer.newimpl.transformer.MultiHeadAttention;
import org.brain4j.core.layer.newimpl.transformer.MaskedMultiHeadAttention;
import org.brain4j.core.layer.newimpl.transformer.PosEncodeLayer;
import org.brain4j.core.layer.newimpl.transformer.TransformerDecoder;
import org.brain4j.core.layer.newimpl.transformer.TransformerEncoder;
import org.brain4j.core.layer.newimpl.utility.ActivationLayer;
import org.brain4j.core.layer.newimpl.utility.SelectLayer;
import org.brain4j.core.layer.newimpl.utility.SliceLayer;
import org.brain4j.core.layer.newimpl.utility.SqueezeLayer;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.clipper.impl.HardClipper;
import org.brain4j.math.commons.Range;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class LayerPortTest {
    
    @Test
    void activationLayerMatchesTensorActivation() {
        ActivationLayer layer = new ActivationLayer(Activations.TANH);
        Tensor input = Tensors.random(2, 3);
        
        Tensor out = layer.forward(new StatesCache(), input)[0];
        Tensor expected = input.activate(Activations.TANH.function());
        
        assertArrayEquals(expected.data(), out.data(), 1e-6f);
    }
    
    @Test
    void selectLayerReturnsIndexedInput() {
        SelectLayer layer = new SelectLayer(1);
        Tensor a = Tensors.random(2, 2);
        Tensor b = Tensors.random(2, 2);
        
        Tensor out = layer.forward(new StatesCache(), a, b)[0];
        
        assertSame(b, out);
    }
    
    @Test
    void sliceLayerMatchesTensorSlice() {
        Range[] ranges = { Range.all(), Range.interval(1, 3) };
        SliceLayer layer = new SliceLayer(ranges);
        Tensor input = Tensors.random(4, 5);
        
        Tensor out = layer.forward(new StatesCache(), input)[0];
        Tensor expected = input.sliceGrad(ranges);
        
        assertArrayEquals(expected.shape(), out.shape());
        assertArrayEquals(expected.data(), out.data(), 1e-6f);
    }
    
    @Test
    void squeezeLayerMatchesTensorSqueeze() {
        SqueezeLayer layer = new SqueezeLayer(-1);
        Tensor input = Tensors.random(1, 3, 1, 2);
        
        Tensor out = layer.forward(new StatesCache(), input)[0];
        Tensor expected = input.reshapeGrad(1, 3, 2);
        
        assertArrayEquals(expected.shape(), out.shape());
        assertArrayEquals(expected.data(), out.data(), 1e-6f);
    }
    
    @Test
    void maxPoolLayerMatchesTensorMaxPool() {
        MaxPoolLayer layer = new MaxPoolLayer(2, 2, 2);
        Tensor input = Tensors.random(1, 1, 4, 4);
        
        Tensor out = layer.forward(new StatesCache(), input)[0];
        Tensor expected = input.maxPoolGrad(2, 2, 2);
        
        assertArrayEquals(expected.shape(), out.shape());
        assertArrayEquals(expected.data(), out.data(), 1e-6f);
    }
    
    @Test
    void inputLayerValidatesShape() {
        InputLayer layer = new InputLayer(Shape.of(3, 4));
        Tensor ok = Tensors.random(2, 3, 4);
        Tensor bad = Tensors.random(2, 3, 5);
        
        assertDoesNotThrow(() -> layer.forward(new StatesCache(), ok));
        assertThrows(IllegalArgumentException.class, () -> layer.forward(new StatesCache(), bad));
    }
    
    @Test
    void embeddingLayerRuns() {
        int vocab = 6;
        int dim = 4;
        int seq = 3;
        int batch = 2;
        
        EmbeddingLayer layer = new EmbeddingLayer(vocab, dim);
        layer.build(List.of(Shape.of(seq)));
        
        Tensor weights = Tensors.random(vocab, dim);
        layer.registerParam("weights", weights);
        
        Tensor input = Tensors.create(Shape.of(batch, seq),
            0, 1, 2,
            3, 4, 5
        );
        
        StatesCache cache = new StatesCache();
        
        Tensor outNew = layer.forward(cache, input)[0];
        assertEquals(input.shapeAt(0), outNew.shapeAt(0));
    }
    
    @Test
    void posEncodeLayerApplies() {
        int length = 16;
        int dim = 6;
        int seq = 4;
        int batch = 2;
        
        PosEncodeLayer layer = new PosEncodeLayer(length, dim);
        
        Tensor input = Tensors.random(batch, seq, dim);
        StatesCache cache = new StatesCache();
        
        Tensor outNew = layer.forward(cache, input)[0];
        assertArrayEquals(input.shape(), outNew.shape());
    }
    
    @Test
    void multiHeadAttentionRuns() {
        int heads = 2;
        int dim = 4;
        int seq = 3;
        int batch = 2;
        
        MultiHeadAttention layer =
            new MultiHeadAttention(new HardClipper(5), heads, dim)
                .attnQkvHasBias(false)
                .attnOutHasBias(false);
        layer.build(List.of(Shape.of(seq, dim)));
        
        Tensor input = Tensors.random(batch, seq, dim);
        StatesCache cache = new StatesCache();
        
        Tensor outNew = layer.forward(cache, input)[0];
        assertArrayEquals(input.shape(), outNew.shape());
    }
    
    @Test
    void transformerEncoderForwardShape() {
        int heads = 2;
        int dim = 4;
        int seq = 3;
        int batch = 2;
        
        TransformerEncoder encoder = new TransformerEncoder(heads, dim, 0.0);
        encoder.build(List.of(Shape.of(seq, dim)));
        
        Tensor input = Tensors.random(batch, seq, dim);
        Tensor out = encoder.forward(new StatesCache(), input)[0];
        
        assertArrayEquals(input.shape(), out.shape());
    }
    
    @Test
    void transformerDecoderForwardShape() {
        int heads = 2;
        int dim = 4;
        int seq = 3;
        int batch = 2;
        
        TransformerDecoder decoder = new TransformerDecoder(heads, dim, 0.0);
        decoder.build(List.of(Shape.of(seq, dim)));
        
        Tensor input = Tensors.random(batch, seq, dim);
        Tensor out = decoder.forward(new StatesCache(), input)[0];
        
        assertArrayEquals(input.shape(), out.shape());
    }
    
    @Test
    void maskedMultiHeadAttentionRuns() {
        int heads = 2;
        int dim = 4;
        int seq = 3;
        int batch = 2;
        
        MaskedMultiHeadAttention layer = new MaskedMultiHeadAttention(new HardClipper(5), heads, dim);
        layer.attnQkvHasBias(false).attnOutHasBias(false);
        layer.build(List.of(Shape.of(seq, dim)));
        
        Tensor input = Tensors.random(batch, seq, dim);
        Tensor out = layer.forward(new StatesCache(), input)[0];
        
        assertArrayEquals(input.shape(), out.shape());
    }
}

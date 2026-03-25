package org.brain4j.core;

import org.brain4j.core.layer.newimpl.DenseLayer;
import org.brain4j.core.layer.newimpl.DropoutLayer;
import org.brain4j.core.layer.newimpl.InputLayer;
import org.brain4j.core.layer.newimpl.LSTMLayer;
import org.brain4j.core.layer.newimpl.MaxPoolLayer;
import org.brain4j.core.layer.newimpl.NormLayer;
import org.brain4j.core.layer.newimpl.RMSNormLayer;
import org.brain4j.core.layer.newimpl.ReshapeLayer;
import org.brain4j.core.layer.newimpl.ConvLayer;
import org.brain4j.core.layer.newimpl.transformer.MaskedMultiHeadAttention;
import org.brain4j.core.layer.newimpl.transformer.EmbeddingLayer;
import org.brain4j.core.layer.newimpl.transformer.PosEncodeLayer;
import org.brain4j.core.layer.newimpl.transformer.TransformerDecoder;
import org.brain4j.core.layer.newimpl.transformer.TransformerEncoder;
import org.brain4j.core.layer.newimpl.transformer.MultiHeadAttention;
import org.brain4j.core.layer.newimpl.utility.ActivationLayer;
import org.brain4j.core.layer.newimpl.utility.SelectLayer;
import org.brain4j.core.layer.newimpl.utility.SliceLayer;
import org.brain4j.core.layer.newimpl.utility.SqueezeLayer;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.ModelSpecs;
import org.brain4j.core.training.Trainer;
import org.brain4j.core.training.TrainingConfig;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.clipper.impl.HardClipper;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.Sample;
import org.brain4j.math.commons.Range;
import org.brain4j.math.loss.impl.MeanSquaredError;
import org.brain4j.core.training.optimizer.impl.Adam;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;

public class TrainingSmokeTest {
    
    @Test
    void trainingRunsOnSimpleRegression() {
        ModelSpecs specs = ModelSpecs.of(
            new InputLayer(Shape.of(2)),
            new DenseLayer(1)
        );
        
        Model model = specs.compile(42);
        
        List<Sample> samples = new ArrayList<>();
        
        for (int i = 0; i < 8; i++) {
            Tensor input = Tensors.vector(i, i + 1);
            Tensor label = Tensors.vector((i + i + 1));
            samples.add(new Sample(input, label));
        }
        
        ListDataSource data = new ListDataSource(samples, false, 4);
        
        TrainingConfig config = TrainingConfig.of(
            new MeanSquaredError(),
            new Adam(0.01)
        );
        
        Trainer trainer = Trainer.of(model, config);
        
        trainer.fit(data, 1);
        
        Tensor prediction = model.predict(Tensors.vector(1, 2).reshape(1, 2));
        assertArrayEquals(new int[] { 1, 1 }, prediction.shape());
    }
    
    @Test
    void trainingRunsOnConvStack() {
        ModelSpecs specs = ModelSpecs.of(
            new InputLayer(Shape.of(1, 6, 6)),
            new ConvLayer(2, 3, 3, 1, Activations.RELU.function()),
            new MaxPoolLayer(2, 2, 2),
            new ReshapeLayer(Shape.of(8)),
            new DenseLayer(3, Activations.LINEAR.function())
        );
        
        Model model = specs.compile(7);
        
        List<Sample> samples = new ArrayList<>();
        
        for (int i = 0; i < 6; i++) {
            Tensor input = Tensors.random(1, 6, 6);
            Tensor label = Tensors.vector(i % 3, (i + 1) % 3, (i + 2) % 3);
            samples.add(new Sample(input, label));
        }
        
        ListDataSource data = new ListDataSource(samples, false, 2);
        
        TrainingConfig config = TrainingConfig.of(
            new MeanSquaredError(),
            new Adam(0.005)
        );
        
        Trainer trainer = Trainer.of(model, config);
        
        trainer.fit(data, 1);
    }
    
    @Test
    void trainingRunsOnUtilityStack() {
        ModelSpecs specs = ModelSpecs.of(
            new InputLayer(Shape.of(2, 3)),
            new SliceLayer(Range.all(), Range.interval(0, 2)),
            new SqueezeLayer(-1),
            new ReshapeLayer(Shape.of(4)),
            new ActivationLayer(Activations.TANH),
            new DenseLayer(2)
        );
        
        Model model = specs.compile(9);
        
        List<Sample> samples = new ArrayList<>();
        
        for (int i = 0; i < 5; i++) {
            Tensor input = Tensors.random(2, 3);
            Tensor label = Tensors.vector(0.5f, -0.5f);
            samples.add(new Sample(input, label));
        }
        
        ListDataSource data = new ListDataSource(samples, false, 2);
        
        TrainingConfig config = TrainingConfig.of(
            new MeanSquaredError(),
            new Adam(0.01)
        );
        
        Trainer trainer = Trainer.of(model, config);
        
        trainer.fit(data, 1);
    }
    
    @Test
    void trainingRunsOnDualBranchSelect() {
        ModelSpecs specs = ModelSpecs.of(
            new InputLayer(Shape.of(4)),
            new DenseLayer(4),
            new DenseLayer(4),
            new SelectLayer(0),
            new DenseLayer(1)
        );
        
        Model model = specs.compile(11);
        
        List<Sample> samples = new ArrayList<>();
        
        for (int i = 0; i < 6; i++) {
            Tensor input = Tensors.vector(i, i + 1, i + 2, i + 3);
            Tensor label = Tensors.vector(1);
            samples.add(new Sample(input, label));
        }
        
        ListDataSource data = new ListDataSource(samples, false, 3);
        
        TrainingConfig config = TrainingConfig.of(
            new MeanSquaredError(),
            new Adam(0.01)
        );
        
        Trainer trainer = Trainer.of(model, config);
        
        assertDoesNotThrow(() -> trainer.fit(data, 1));
    }
    
    @Test
    void trainingRunsOnNormStack() {
        ModelSpecs specs = ModelSpecs.of(
            new InputLayer(Shape.of(4)),
            new DenseLayer(4),
            new NormLayer(),
            new RMSNormLayer(),
            new DenseLayer(2)
        );
        
        Model model = specs.compile(13);
        
        List<Sample> samples = new ArrayList<>();
        
        for (int i = 0; i < 6; i++) {
            Tensor input = Tensors.vector(i, i + 1, i + 2, i + 3);
            Tensor label = Tensors.vector(0.25f, -0.25f);
            samples.add(new Sample(input, label));
        }
        
        ListDataSource data = new ListDataSource(samples, false, 2);
        
        TrainingConfig config = TrainingConfig.of(
            new MeanSquaredError(),
            new Adam(0.01)
        );
        
        Trainer trainer = Trainer.of(model, config);
        
        assertDoesNotThrow(() -> trainer.fit(data, 1));
    }
    
    @Test
    void trainingRunsOnTransformerEncoder() {
        ModelSpecs specs = ModelSpecs.of(
            new InputLayer(Shape.of(3, 4)),
            new TransformerEncoder(2, 4, 0.0),
            new DenseLayer(2)
        );
        
        Model model = specs.compile(21);
        
        List<Sample> samples = new ArrayList<>();
        
        for (int i = 0; i < 4; i++) {
            Tensor input = Tensors.random(3, 4);
            Tensor label = Tensors.random(3, 2);
            samples.add(new Sample(input, label));
        }
        
        ListDataSource data = new ListDataSource(samples, false, 2);
        
        TrainingConfig config = TrainingConfig.of(
            new MeanSquaredError(),
            new Adam(0.001)
        );
        
        Trainer trainer = Trainer.of(model, config);
        
        assertDoesNotThrow(() -> trainer.fit(data, 1));
    }
    
    @Test
    void trainingRunsOnTransformerDecoder() {
        ModelSpecs specs = ModelSpecs.of(
            new InputLayer(Shape.of(3, 4)),
            new TransformerDecoder(2, 4, 0.0),
            new DenseLayer(2)
        );
        
        Model model = specs.compile(23);
        
        List<Sample> samples = new ArrayList<>();
        
        for (int i = 0; i < 4; i++) {
            Tensor input = Tensors.random(3, 4);
            Tensor label = Tensors.random(3, 2);
            samples.add(new Sample(input, label));
        }
        
        ListDataSource data = new ListDataSource(samples, false, 2);
        
        TrainingConfig config = TrainingConfig.of(
            new MeanSquaredError(),
            new Adam(0.001)
        );
        
        Trainer trainer = Trainer.of(model, config);
        
        assertDoesNotThrow(() -> trainer.fit(data, 1));
    }
    
    @Test
    void trainingRunsOnEmbeddingAndPositional() {
        ModelSpecs specs = ModelSpecs.of(
            new InputLayer(Shape.of(5)),
            new EmbeddingLayer(10, 4),
            new PosEncodeLayer(32, 4),
            new DenseLayer(3)
        );
        
        Model model = specs.compile(31);
        
        List<Sample> samples = new ArrayList<>();
        
        for (int i = 0; i < 4; i++) {
            Tensor input = Tensors.create(Shape.of(5), 0, 1, 2, 3, 4);
            Tensor label = Tensors.random(5, 3);
            samples.add(new Sample(input, label));
        }
        
        ListDataSource data = new ListDataSource(samples, false, 2);
        
        TrainingConfig config = TrainingConfig.of(
            new MeanSquaredError(),
            new Adam(0.001)
        );
        
        Trainer trainer = Trainer.of(model, config);
        
        assertDoesNotThrow(() -> trainer.fit(data, 1));
    }
    
    @Test
    void trainingRunsOnLstm() {
        ModelSpecs specs = ModelSpecs.of(
            new InputLayer(Shape.of(3, 4)),
            new LSTMLayer(5, true),
            new DenseLayer(2)
        );
        
        Model model = specs.compile(51);
        
        List<Sample> samples = new ArrayList<>();
        
        for (int i = 0; i < 4; i++) {
            Tensor input = Tensors.random(3, 4);
            Tensor label = Tensors.random(3, 2);
            samples.add(new Sample(input, label));
        }
        
        ListDataSource data = new ListDataSource(samples, false, 2);
        
        TrainingConfig config = TrainingConfig.of(
            new MeanSquaredError(),
            new Adam(0.001)
        );
        
        Trainer trainer = Trainer.of(model, config);
        
        assertDoesNotThrow(() -> trainer.fit(data, 1));
    }
    
    @Test
    void trainingRunsOnAttentionBlock() {
        ModelSpecs specs = ModelSpecs.of(
            new InputLayer(Shape.of(3, 4)),
            new MultiHeadAttention(2, 4),
            new DenseLayer(2)
        );
        
        Model model = specs.compile(61);
        
        List<Sample> samples = new ArrayList<>();
        
        for (int i = 0; i < 4; i++) {
            Tensor input = Tensors.random(3, 4);
            Tensor label = Tensors.random(3, 2);
            samples.add(new Sample(input, label));
        }
        
        ListDataSource data = new ListDataSource(samples, false, 2);
        
        TrainingConfig config = TrainingConfig.of(
            new MeanSquaredError(),
            new Adam(0.001)
        );
        
        Trainer trainer = Trainer.of(model, config);
        
        assertDoesNotThrow(() -> trainer.fit(data, 1));
    }
    
    @Test
    void trainingRunsOnMaskedAttentionBlock() {
        ModelSpecs specs = ModelSpecs.of(
            new InputLayer(Shape.of(3, 4)),
            new MaskedMultiHeadAttention(new HardClipper(5), 2, 4),
            new DenseLayer(2)
        );
        
        Model model = specs.compile(71);
        
        List<Sample> samples = new ArrayList<>();
        
        for (int i = 0; i < 4; i++) {
            Tensor input = Tensors.random(3, 4);
            Tensor label = Tensors.random(3, 2);
            samples.add(new Sample(input, label));
        }
        
        ListDataSource data = new ListDataSource(samples, false, 2);
        
        TrainingConfig config = TrainingConfig.of(
            new MeanSquaredError(),
            new Adam(0.001)
        );
        
        Trainer trainer = Trainer.of(model, config);
        
        assertDoesNotThrow(() -> trainer.fit(data, 1));
    }
    
    @Test
    void trainingRunsOnDropoutAndActivation() {
        ModelSpecs specs = ModelSpecs.of(
            new InputLayer(Shape.of(4)),
            new DenseLayer(4),
            new DropoutLayer(0.2),
            new ActivationLayer(Activations.RELU),
            new DenseLayer(1)
        );
        
        Model model = specs.compile(41);
        
        List<Sample> samples = new ArrayList<>();
        
        for (int i = 0; i < 6; i++) {
            Tensor input = Tensors.random(4);
            Tensor label = Tensors.vector(0.1f);
            samples.add(new Sample(input, label));
        }
        
        ListDataSource data = new ListDataSource(samples, false, 3);
        
        TrainingConfig config = TrainingConfig.of(
            new MeanSquaredError(),
            new Adam(0.005)
        );
        
        Trainer trainer = Trainer.of(model, config);
        
        assertDoesNotThrow(() -> trainer.fit(data, 1));
    }
}

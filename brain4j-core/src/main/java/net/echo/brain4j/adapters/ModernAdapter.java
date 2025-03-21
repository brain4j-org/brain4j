package net.echo.brain4j.adapters;

import com.github.luben.zstd.Zstd;
import net.echo.math4j.BrainUtils;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;

import java.io.*;
import java.util.List;

public class ModernAdapter {

    public static void serialize(String path, Model<?, ?, ?> model) throws Exception {
        String suffix = path.endsWith(".bin") ? "" : ".bin";
        serialize(new File(path + suffix), model);
    }

    public static void serialize(File file, Model<?, ?, ?> model) throws Exception {
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        DataOutputStream dataStream = new DataOutputStream(outputStream);

        dataStream.writeInt(model.getSeed()); // seed
        dataStream.writeUTF(model.getLossFunction().getClass().getName()); // loss_function
        dataStream.writeUTF(model.getWeightInit().getClass().getName()); // weight_init
        dataStream.writeUTF(model.getUpdater().getClass().getName()); // updater
        dataStream.writeUTF(model.getOptimizer().getClass().getName());

        model.getOptimizer().serialize(dataStream); // optimizer
        model.serialize(dataStream); // serializes layers

        int layers = model.getLayers().size(); // layers

        for (int i = 0; i < layers; i++) {
            Layer<?, ?> layer = model.getLayers().get(i);

            for (Neuron neuron : layer.getNeurons()) {
                dataStream.writeDouble(neuron.getBias());
            }

            int synapses = layer.getSynapses().size();

            for (int j = 0; j < synapses; j++) {
                Synapse synapse = layer.getSynapses().get(j);
                dataStream.writeDouble(synapse.getWeight());
            }
        }

        byte[] bytes = outputStream.toByteArray();

        try (FileOutputStream fileOutputStream = new FileOutputStream(file)) {
            //byte[] compressed = compress(bytes);
            fileOutputStream.write(bytes);
        }
    }

    public static <T extends Model<?, ?, ?>> T deserialize(String path, T model) throws Exception {
        return deserialize(new File(path), model);
    }

    public static <T extends Model<?, ?, ?>> T deserialize(File file, T model) throws Exception {
        try (FileInputStream fileInputStream = new FileInputStream(file)) {
            //byte[] bytes = decompress(fileInputStream.readAllBytes());

            // DataInputStream dataStream = new DataInputStream(new ByteArrayInputStream(bytes));
            DataInputStream dataStream = new DataInputStream(fileInputStream);
            int seed = dataStream.readInt();

            String lossFunctionClass = dataStream.readUTF();
            String weightInitClass = dataStream.readUTF();
            String updaterClass = dataStream.readUTF();
            String optimizerClass = dataStream.readUTF();

            LossFunction lossFunction = BrainUtils.newInstance(lossFunctionClass);
            WeightInitializer weightInit = BrainUtils.newInstance(weightInitClass);
            Updater updater = BrainUtils.newInstance(updaterClass);
            Optimizer optimizer = BrainUtils.newInstance(optimizerClass);

            optimizer.deserialize(dataStream);

            model.setSeed(seed);
            model.deserialize(dataStream);

            model.compile(weightInit, lossFunction, optimizer, updater);

            int layers = model.getLayers().size();

            for (int i = 0; i < layers; i++) {
                Layer<?, ?> layer = model.getLayers().get(i);

                List<Neuron> neurons = layer.getNeurons();

                for (Neuron neuron : neurons) {
                    double bias = dataStream.readDouble();
                    neuron.setBias(bias);
                }

                List<Synapse> synapses = layer.getSynapses();

                for (Synapse synapse : synapses) {
                    double weight = dataStream.readDouble();
                    synapse.setWeight(weight);
                }
            }

            model.reloadWeights();
            return model;
        }
    }

    public static byte[] compress(byte[] data) {
        return Zstd.compress(data);
    }

    public static byte[] decompress(byte[] data) {
        long decompressedSize = Zstd.decompressedSize(data);
        return Zstd.decompress(data, (int) decompressedSize);
    }
}

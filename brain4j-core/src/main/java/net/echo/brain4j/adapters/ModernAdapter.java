package net.echo.brain4j.adapters;

import com.github.luben.zstd.Zstd;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.training.optimizer.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.math4j.BrainUtils;
import net.echo.math4j.math.tensor.Tensor;

import java.io.*;

public class ModernAdapter {

    public static void serialize(String path, Model model) throws Exception {
        String suffix = path.endsWith(".bin") ? "" : ".bin";
        serialize(new File(path + suffix), model);
    }

    public static void serialize(File file, Model model) throws Exception {
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
            Layer layer = model.getLayers().get(i);

            Tensor biases = layer.getBias();
            Tensor weights = layer.getWeights();

            for (int j = 0; j < biases.elements(); j++) {
                dataStream.writeDouble(biases.get(j));
            }

            Tensor reshapedWeights = weights.reshape(weights.elements());

            for (int j = 0; j < reshapedWeights.elements(); j++) {
                dataStream.writeDouble(reshapedWeights.get(j));
            }
        }

        dataStream.writeInt(outputStream.size());
        byte[] bytes = outputStream.toByteArray();

        try (FileOutputStream fileOutputStream = new FileOutputStream(file)) {
            byte[] compressed = compress(bytes);
            fileOutputStream.write(compressed);
        }
    }

    public static <T extends Model> T deserialize(String path, T model) throws Exception {
        return deserialize(new File(path), model);
    }

    public static <T extends Model> T deserialize(File file, T model) throws Exception {
        try (FileInputStream fileInputStream = new FileInputStream(file)) {
            DataInputStream dataStream = new DataInputStream(fileInputStream);
            int size = dataStream.readInt();

            byte[] bytes = decompress(size, fileInputStream.readAllBytes());
            dataStream = new DataInputStream(new ByteArrayInputStream(bytes));

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
                Layer layer = model.getLayers().get(i);

                Tensor biases = layer.getBias();
                Tensor weights = layer.getWeights();

                for (int j = 0; j < biases.elements(); j++) {
                    double bias = dataStream.readDouble();
                    biases.set(bias, j);
                }

                int[] shape = weights.shape();

                for (int j = 0; j < shape[0]; j++) {
                    for (int k = 0; k < shape[1]; k++) {
                        double weight = dataStream.readDouble();
                        weights.set(weight, j, k);
                    }
                }
            }

            return model;
        }
    }

    public static byte[] compress(byte[] data) {
        return Zstd.compress(data);
    }

    public static byte[] decompress(int size, byte[] data) {
        return Zstd.decompress(data, size);
    }
}

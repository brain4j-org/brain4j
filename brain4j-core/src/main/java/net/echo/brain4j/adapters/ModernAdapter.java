package net.echo.brain4j.adapters;

import com.github.luben.zstd.Zstd;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.training.BackPropagation;
import net.echo.brain4j.training.optimizer.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.math4j.BrainUtils;

import java.io.*;

public class ModernAdapter {

    public static void serialize(String path, Model model) throws Exception {
        String suffix = path.endsWith(".b4j") ? "" : ".b4j";
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
            byte[] bytes = decompress(fileInputStream.readAllBytes());
            DataInputStream dataStream = new DataInputStream(new ByteArrayInputStream(bytes));

            int seed = dataStream.readInt();

            LossFunction lossFunction = BrainUtils.newInstance(dataStream.readUTF());
            WeightInitializer weightInit = BrainUtils.newInstance(dataStream.readUTF());
            Updater updater = BrainUtils.newInstance(dataStream.readUTF());
            Optimizer optimizer = BrainUtils.newInstance(dataStream.readUTF());

            optimizer.deserialize(dataStream);

            model.setSeed(seed);
            model.deserialize(dataStream);

            model.setLossFunction(lossFunction);
            model.setWeightInit(weightInit);
            model.setUpdater(updater);
            model.setOptimizer(optimizer);

            model.getOptimizer().postInitialize(model);
            model.getUpdater().postInitialize();
            model.setPropagation(new BackPropagation(model, optimizer, updater));

            for (Layer layer : model.getLayers()) {
                layer.compile(weightInit, lossFunction, optimizer, updater);
            }

            return model;
        }
    }

    public static byte[] compress(byte[] data) {
        return Zstd.compress(data);
    }

    public static byte[] decompress(byte[] data) {
        int size = (int) Zstd.getFrameContentSize(data);
        return Zstd.decompress(data, size);
    }
}

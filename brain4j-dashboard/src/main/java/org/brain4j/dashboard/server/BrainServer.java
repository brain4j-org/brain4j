package org.brain4j.dashboard.server;

import com.sun.management.OperatingSystemMXBean;
import org.brain4j.core.importing.Format;
import org.brain4j.core.importing.ModelIO;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.impl.Graph;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.core.monitor.impl.TimingMonitor;
import org.brain4j.core.monitor.impl.EvalMonitor;
import org.brain4j.core.training.Trainer;
import org.brain4j.core.training.wrappers.EvaluationResult;
import org.brain4j.dashboard.BrainDashboard;
import org.brain4j.dashboard.miniserver.*;
import org.brain4j.math.data.ListDataSource;

import java.io.File;
import java.lang.management.ManagementFactory;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class BrainServer extends MiniServer {

    private final BrainDashboard dashboard;
    private final ListDataSource trainSet;
    private final TimingMonitor timingMonitor;
    private final EvalMonitor evalMonitor;
    private final LossRecorder lossRecorder;
    private final int epochs;
    
    public BrainServer(BrainDashboard dashboard, ListDataSource trainSet, ListDataSource testSet, int epochs) {
        this.dashboard = dashboard;
        this.trainSet = trainSet;
        this.timingMonitor = dashboard.attach(TimingMonitor.class, () -> new TimingMonitor(20));
        this.evalMonitor = dashboard.attach(EvalMonitor.class, () -> new EvalMonitor(testSet, 1, false));
        this.lossRecorder = dashboard.attach(LossRecorder.class, () -> new LossRecorder(evalMonitor));
        this.epochs = epochs;
    }

    @Override
    public List<Object> getServices() {
        return List.of(this);
    }
    
    @Route("/")
    public Response html(Request request) {
        return serveResource("index.html", "text/html; charset=utf-8");
    }

    @Route("/style.css")
    public Response style(Request request) {
        return serveResource("style.css", "text/css; charset=utf-8");
    }

    @Route("/script.js")
    public Response script(Request request) {
        return serveResource("script.js", "application/javascript; charset=utf-8");
    }

    @Route(value = "/api/training/start", accepted = { HttpMethod.POST })
    public Response start(Request request) {
        Trainer trainer = dashboard.trainer();

        if (trainer.isTraining()) {
            return Response.json(400, Map.of("message", "Already training"));
        }

        lossRecorder.getRecordedLoss().clear();
        trainer.start(trainSet, epochs);

        return Response.ok();
    }

    @Route(value = "/api/training/pause", accepted = { HttpMethod.POST })
    public Response pause(Request request) {
        Trainer trainer = dashboard.trainer();

        if (!trainer.isTraining()) {
            return Response.json(400, Map.of("message", "Not training"));
        }

        trainer.pause();
        return Response.ok();
    }

    @Route(value = "/api/training/resume", accepted = { HttpMethod.POST })
    public Response resume(Request request) {
        Trainer trainer = dashboard.trainer();

        if (!trainer.isTraining()) {
            return Response.json(400, Map.of("message", "Not training"));
        }

        trainer.resume();
        return Response.ok();
    }

    @Route(value = "/api/training/stop", accepted = { HttpMethod.POST })
    public Response stop(Request request) {
        Trainer trainer = dashboard.trainer();

        if (!trainer.isTraining()) {
            return Response.json(400, Map.of("message", "Not training"));
        }

        trainer.stop();
        return Response.ok();
    }

    @Route("/api/training/info")
    public Response info(Request request) {
        Trainer trainer = dashboard.trainer();
        EvaluationResult result = evalMonitor.getEvalResult();

        double loss = result == null ? 0.0 : result.loss();
        double accuracy = result == null ? 0.0 : result.accuracy();

        Map<String, Object> info = new HashMap<>(
            Map.of(
                "training", trainer.isTraining(),
                "paused", trainer.isPaused(),
                "epoch", trainer.currentEpoch(),
                "total_epochs", trainer.totalEpochs(),
                "batch", trainer.currentBatch(),
                "total_batches", trainer.totalBatches(),
                "loss", loss,
                "accuracy", accuracy,
                "average_time_per_batch", timingMonitor.averagePerBatch()
            )
        );
        
        info.put("loss_table", lossRecorder.getRecordedLoss());
        info.put("accuracy_table", lossRecorder.getRecordedAccuracy());
        info.put("f1_table", lossRecorder.getRecordedF1());

        return Response.json(200, info);
    }

    @Route("/api/system/resources")
    public Response resources(Request request) {
        OperatingSystemMXBean os = (OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();
        double cpuUsage = os.getCpuLoad() * 100;
        
        return Response.json(200, Map.of(
            "cpu_usage", cpuUsage,
            "gpu_usage", 0.0,
            "total", os.getTotalMemorySize(),
            "free", os.getFreeMemorySize()
        ));
    }
    
    @Route(value = "/api/save-model", accepted = { HttpMethod.POST })
    public Response saveModel(Request request) {
        String rawPath = request.queryParams().get("path");

        if (rawPath == null || rawPath.isBlank()) {
            return Response.json(400, Map.of("message", "Missing required query param: path"));
        }

        String decodedPath = URLDecoder.decode(rawPath, StandardCharsets.UTF_8);
        File modelFile = new File(decodedPath);

        if (modelFile.isDirectory()) {
            return Response.json(400, Map.of("message", "Path points to a directory, not a file"));
        }

         if (!modelFile.getName().toLowerCase().endsWith(".zip")) {
            modelFile = new File(modelFile.getAbsolutePath() + ".zip");
        }

        File parent = modelFile.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            return Response.json(500, Map.of("message", "Cannot create destination directory"));
        }

        try {
            Model model = dashboard.model();

            if (model instanceof Sequential sequential) {
                ModelIO.save(sequential, Format.BRAIN4J, modelFile);
            } else if (model instanceof Graph graph) {
                ModelIO.save(graph, Format.ONNX, modelFile);
            } else {
                throw new UnsupportedOperationException("Unsupported model type!");
            }

            return Response.json(200, Map.of(
                "message", "Model saved successfully",
                "path", modelFile.getAbsolutePath()
            ));
        } catch (Exception ex) {
            ex.printStackTrace(System.err);
            return Response.json(500, Map.of(
                "message", "Model save failed",
                "error", String.valueOf(ex.getMessage())
            ));
        }
    }
}

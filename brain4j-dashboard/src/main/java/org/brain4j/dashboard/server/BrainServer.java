package org.brain4j.dashboard.server;

import com.sun.management.OperatingSystemMXBean;
import org.brain4j.core.monitor.impl.TimingMonitor;
import org.brain4j.core.monitor.impl.EvalMonitor;
import org.brain4j.core.training.Trainer;
import org.brain4j.core.training.wrappers.EvaluationResult;
import org.brain4j.dashboard.BrainDashboard;
import org.brain4j.dashboard.miniserver.*;
import org.brain4j.math.data.ListDataSource;

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

public class BrainServer extends MiniServer {

    private final BrainDashboard dashboard;
    private final ListDataSource trainSet;
    private final ListDataSource testSet;
    private final TimingMonitor timingMonitor;
    private final EvalMonitor evalMonitor;
    private final int epochs;

    private double lastLoss;

    public BrainServer(BrainDashboard dashboard, ListDataSource trainSet, ListDataSource testSet, int epochs) {
        this.dashboard = dashboard;
        this.trainSet = trainSet;
        this.testSet = testSet;
        this.timingMonitor = dashboard.attach(TimingMonitor.class, () -> new TimingMonitor(20));
        this.evalMonitor = dashboard.attach(EvalMonitor.class, () -> new EvalMonitor(testSet, 1, false));
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
        Trainer trainer = dashboard.getTrainer();

        if (trainer.isTraining()) {
            return Response.json(400, Map.of("message", "Already training"));
        }

        trainer.start(trainSet, epochs);
        return Response.ok();
    }

    @Route(value = "/api/training/pause", accepted = { HttpMethod.POST })
    public Response pause(Request request) {
        Trainer trainer = dashboard.getTrainer();

        if (!trainer.isTraining()) {
            return Response.json(400, Map.of("message", "Not training"));
        }

        trainer.pause();
        return Response.ok();
    }

    @Route(value = "/api/training/resume", accepted = { HttpMethod.POST })
    public Response resume(Request request) {
        Trainer trainer = dashboard.getTrainer();

        if (!trainer.isTraining()) {
            return Response.json(400, Map.of("message", "Not training"));
        }

        trainer.resume();
        return Response.ok();
    }

    @Route(value = "/api/training/stop", accepted = { HttpMethod.POST })
    public Response stop(Request request) {
        Trainer trainer = dashboard.getTrainer();

        if (!trainer.isTraining()) {
            return Response.json(400, Map.of("message", "Not training"));
        }

        trainer.stop();
        return Response.ok();
    }

    @Route("/api/training/info")
    public Response info(Request request) {
        Trainer trainer = dashboard.getTrainer();
        EvaluationResult result = evalMonitor.getEvalResult();

        double loss = result == null ? 0.0 : result.loss();
        double accuracy = result == null ? 0.0 : result.accuracy();

        var info = Map.of(
            "training", trainer.isTraining(),
            "paused", trainer.isPaused(),
            "epoch", trainer.currentEpoch(),
            "total_epochs", trainer.totalEpochs(),
            "batch", trainer.currentBatch(),
            "total_batches", trainer.totalBatches(),
            "loss", loss,
            "accuracy", accuracy,
            "average_time_per_batch", timingMonitor.averagePerBatch()
        );
        return Response.json(200, info);
    }

    @Route("/api/system/resources")
    public Response resources(Request request) {
        OperatingSystemMXBean os = (OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();
        double cpuUsage = os.getSystemLoadAverage();

        return Response.json(200, Map.of(
            "cpu_usage", cpuUsage,
            "gpu_usage", 0.0,
            "total", os.getTotalMemorySize(),
            "free", os.getFreeMemorySize()
        ));
    }

    @Route("/hello")
    public Response hello(Request request) {
        return Response.text(200, "Hello!");
    }

    private Response serveResource(String resourceName, String contentType) {
        try (var stream = getClass().getClassLoader().getResourceAsStream(resourceName)) {
            if (stream == null) {
                return Response.json(404, Map.of("message", "Resource not found"));
            }

            String resource = new String(stream.readAllBytes(), StandardCharsets.UTF_8);
            return Response.text(200, resource).addHeader("Content-Type", contentType);
        } catch (IOException e) {
            return Response.json(500, Map.of("message", "Cannot load resource"));
        }
    }
}

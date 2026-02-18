package org.brain4j.core.monitor.impl;

import org.brain4j.core.Brain4J;
import org.brain4j.core.training.Trainer;
import org.brain4j.core.training.events.*;
import org.brain4j.core.utils.Colored;
import org.brain4j.math.commons.Commons;

public final class ProgressMonitor extends TimingMonitor {
    
    private static final long PRINT_THRESHOLD = 20 * 1_000_000; // 20 ms in ns
    private long lastLogTimestamp;
    
    public ProgressMonitor() {
        this(20);
    }
    
    public ProgressMonitor(int timeWindow) {
        super(timeWindow);
    }
    
    @Override
    public void onEvent(TrainingEvent event, Trainer trainer) {
        super.onEvent(event, trainer);
        switch (event) {
            case BatchEnd(int batch, int totalBatches) -> batchCompleted(batch, totalBatches);
            case TrainingEnd() -> trainingEnd();
            default -> {}
        }
    }
    
    private void batchCompleted(int batch, int total) {
        if (Brain4J.isLogging()) {
            printProgress(batch + 1, total, averagePerBatch());
        }
    }
    private void trainingEnd() {
        if (!Brain4J.isLogging()) return;
        
        System.out.println(); // go to new line to avoid future formatting issues
    }
    
    private void printProgress(int batch, int totalBatches, double tookMs) {
        long diff = System.nanoTime() - lastLogTimestamp;
        
        if (diff < PRINT_THRESHOLD && batch != totalBatches) return;
        
        String barChar = Commons.HEADER_CHAR;
        
        int progressBarLength = 25;
        
        double percentage = (double) batch / totalBatches;
        double tookInSeconds = tookMs / 1000.0;
        
        String timeStr = Commons.formatDuration(tookInSeconds);
        
        String progressBar = Commons.createProgressBar(
            percentage, progressBarLength,
            "<green>", barChar,
            "<reset>", barChar
        ) + " ";
        String progress = Colored.renderText(progressBar);
        
        String intro = Colored.renderText("Epoch <yellow>%s<white>/<yellow>%s ", currentEpoch + 1, totalEpochs);
        String batches = Colored.renderText("<blue>%s<white>/<blue>%s <white>batches", batch, totalBatches);
        String time = Colored.renderText("<gray> [%s/batch]<reset>", timeStr);
        
        String message = intro + progress + batches + time;
        System.out.print("\r" + message);
        
        this.lastLogTimestamp = System.nanoTime();
    }
}

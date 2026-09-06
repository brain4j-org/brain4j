package org.brain4j.core.training.events;

public record BatchEnd(int batch, int totalBatches) implements TrainingEvent {}

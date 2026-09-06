package org.brain4j.core.training.events;

import org.brain4j.core.training.Trainer;

public record BatchStart(Trainer trainer, int batch, int totalBatches) implements TrainingEvent {}

package org.brain4j.datasets.api;

import org.brain4j.math.data.ListDataSource;

public record Dataset(double percentage, ListDataSource train, ListDataSource test) {

}

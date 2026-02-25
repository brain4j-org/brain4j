package org.brain4j.core.importing;

import org.brain4j.core.importing.format.impl.BrainAdapter;
import org.brain4j.core.importing.format.impl.OnnxAdapter;

public class Format {
    public static BrainAdapter BRAIN4J = new BrainAdapter();
    public static OnnxAdapter ONNX = new OnnxAdapter();
}

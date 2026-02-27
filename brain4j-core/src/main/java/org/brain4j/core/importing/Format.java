package org.brain4j.core.importing;

import org.brain4j.core.importing.format.impl.BrainFormat;
import org.brain4j.core.importing.format.impl.OnnxFormat;

public class Format {
    public static BrainFormat BRAIN4J = new BrainFormat();
    public static OnnxFormat ONNX = new OnnxFormat();
}

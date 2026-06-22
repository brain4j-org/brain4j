package org.brain4j.transformers.core.architecture;

import org.brain4j.transformers.core.architecture.impl.GPT2Adapter;

import java.util.ArrayList;
import java.util.List;

public class ArchitectureRegistry {
    
    private static final List<ArchitectureAdapter> ADAPTERS = new ArrayList<>();
    
    static {
        register(new GPT2Adapter());
    }
    
    public static void register(ArchitectureAdapter adapter) {
        ADAPTERS.add(adapter);
    }
    
    public static ArchitectureAdapter findAdapter(String modelType) {
        return ADAPTERS.stream()
            .filter(a -> a.supports(modelType))
            .findFirst()
            .orElseThrow(() -> new UnsupportedOperationException(
                "No adapter found for model type: " + modelType));
    }
}

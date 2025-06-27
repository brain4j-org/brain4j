package org.brain4j.core.importing;

import org.brain4j.core.model.Model;

public interface ModelLoader {
    
    Model deserialize(byte[] bytes) throws Exception;

    
}

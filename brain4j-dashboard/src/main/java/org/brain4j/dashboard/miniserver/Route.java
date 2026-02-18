package org.brain4j.dashboard.miniserver;

import java.lang.annotation.*;

@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface Route {
    String value();
    HttpMethod[] accepted() default { HttpMethod.GET };
}

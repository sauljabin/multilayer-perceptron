/**
 * Copyright (c) 2014 Saúl Piña <sauljabin@gmail.com>.
 * <p>
 * This file is part of MultilayerPerceptron.
 * <p>
 * MultilayerPerceptron is licensed under The MIT License.
 * For full copyright and license information please see the LICENSE file.
 */

package app;

import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Properties;

public class Translate {

    private static Properties properties = new Properties();

    public static void load() throws IOException {
        String path = String.format("%s%s", "translate/", Config.get("TRANSLATE"));
        properties.load(new InputStreamReader(ClassLoader.getSystemResourceAsStream(path), "UTF-8"));
    }

    public static String get(String key) {
        return properties.getProperty(key);
    }

}

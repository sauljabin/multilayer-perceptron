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
import java.util.Enumeration;
import java.util.Properties;
import java.util.Vector;

public class Config {

    public static String configPath = "CONFIG.properties";
    private static Properties properties = new Properties();

    public static void load() throws IOException {
        properties.load(new InputStreamReader(ClassLoader.getSystemResourceAsStream(configPath), "UTF-8"));
        properties.put("OS", System.getProperty("os.name"));
        properties.put("OS_ARCH", System.getProperty("os.arch"));
    }

    public static String get(String key) {
        return properties.getProperty(key);
    }

    public static Vector<String> getKeys() {
        Vector<String> vectorKeys = new Vector<>();
        Enumeration<Object> keys = properties.keys();
        while (keys.hasMoreElements()) {
            vectorKeys.add((String) keys.nextElement());
        }
        return vectorKeys;
    }

}

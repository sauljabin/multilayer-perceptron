/**
 * Copyright (c) 2014 Saúl Piña <sauljabin@gmail.com>.
 * 
 * This file is part of MultilayerPerceptron.
 * 
 * MultilayerPerceptron is licensed under The MIT License.
 * For full copyright and license information please see the LICENSE file.
 */

package app;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.Enumeration;
import java.util.Properties;
import java.util.Vector;

public class Config {

	private static Properties properties = new Properties();
	public static String configPath = "CONFIG";

	public static void load() throws FileNotFoundException, IOException {
		properties.load(new InputStreamReader(new FileInputStream(configPath), "UTF-8"));
		properties.put("OS", System.getProperty("os.name"));
		properties.put("OS_ARCH", System.getProperty("os.arch"));
	}

	public static String get(String key) {
		return properties.getProperty(key);
	}

	public static Vector<String> getKeys() {
		Vector<String> vectorKeys = new Vector<String>();
		Enumeration<Object> keys = properties.keys();
		while (keys.hasMoreElements()) {
			vectorKeys.add((String) keys.nextElement());
		}
		return vectorKeys;
	}

	public static void set(String key, String value) {
		properties.put(key, value);
	}

	public static void save() throws FileNotFoundException, IOException {
		properties.store(new OutputStreamWriter(new FileOutputStream(configPath), "UTF-8"), "CONFIGURATION");
	}

}

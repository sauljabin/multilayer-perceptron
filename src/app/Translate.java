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
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Enumeration;
import java.util.Properties;

public class Translate {

	private static Properties properties = new Properties();

	public static void load() throws FileNotFoundException, IOException {
		properties.load(new InputStreamReader(new FileInputStream(String.format("%s%s", Config.get("TRANSLATE_PATH"), Config.get("TRANSLATE"))), "UTF-8"));
	}

	public static String get(String key) {
		return properties.getProperty(key);
	}

	public static Enumeration<Object> getKeys() {
		return properties.keys();
	}

}

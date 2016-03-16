/**
 * Copyright (c) 2014 Saúl Piña <sauljabin@gmail.com>.
 * 
 * This file is part of MultilayerPerceptron.
 * 
 * MultilayerPerceptron is licensed under The MIT License.
 * For full copyright and license information please see the LICENSE file.
 */

package app.util;

import java.util.Random;

public class UtilRand {

	public static double random(double min, double max) {
		Random random = new Random();
		return (random.nextDouble() * (max - min)) + min;
	}

	public static double random() {
		Random random = new Random();
		return random.nextDouble();
	}
}

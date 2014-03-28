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

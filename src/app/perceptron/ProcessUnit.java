package app.perceptron;

import app.util.UtilRand;

public class ProcessUnit {

	private double variationRate;
	private double learningRate;
	private double[] weights;
	private double[] previousVariation;

	public ProcessUnit(double learningRate, double variationRate) {
		this.variationRate = learningRate;
		this.learningRate = variationRate;
	}

	public double getLearningRate() {
		return variationRate;
	}

	public void setLearningRate(double learningRate) {
		this.variationRate = learningRate;
	}

	public double getVariationRate() {
		return learningRate;
	}

	public void setVariationRate(double variationRate) {
		this.learningRate = variationRate;
	}

	public double[] getWeights() {
		return weights;
	}

	public void setWeights(double[] weights) {
		this.weights = weights;
	}

	public void initWeights(int n) {
		weights = new double[n];
		previousVariation = new double[n];
		for (int i = 0; i < weights.length; i++) {
			weights[i] = UtilRand.random();
			previousVariation[i] = 0;
		}
	}

	public double activation(double[] inputs) {
		double activation = 0;
		for (int i = 0; i < inputs.length; i++) {
			activation += weights[i] * inputs[i];
		}
		return activation;
	}

	public double output(double activation) {
		return activationFunction(activation);
	}

	public double activationFunction(double x) {
		return sigmoid(x);
	}

	public double activationFunctionDerivative(double x) {
		return sigmoidDerivative(x);
	}

	public double delta(double desiredOutput, double output) {
		return (desiredOutput - output);
	}

	public double delta(double[] d, double[] W) {
		double totalDelta = 0;
		for (int i = 0; i < d.length; i++) {
			totalDelta += W[i] * d[i];
		}
		return totalDelta;
	}

	public void updateWeights(double[] inputs, double delta, double activation) {
		double variation;
		for (int i = 0; i < weights.length; i++) {
			variation = (learningRate * inputs[i] * delta * activationFunctionDerivative(activation)) + previousVariation[i] * variationRate;
			previousVariation[i] = variation;
			weights[i] += variation;
		}
	}

	public double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}

	public double sigmoidDerivative(double x) {
		return sigmoid(x) * (1 - sigmoid(x));
	}
}

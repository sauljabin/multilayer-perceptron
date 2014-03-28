package app.perceptron;

import java.util.LinkedList;
import java.util.List;

public class MultilayerPerceptron {
	private ProcessUnit[] processUnitsOutput;// PROCESSING UNITS OUTPUT LAYER
	private ProcessUnit[] processUnitHidden;// PROCESSING UNITS HIDDEN LAYER
	private int inputLayerSize;// NUMBER OF INPUTS
	private int hiddenLayerSize;// NUMBER PROCESSING UNITS IN HIDDEN LAYER
	private int outputLayerSize;// NUMBER PROCESSING UNITS IN OUTPUT LAYER
	private List<Double> errors;
	private int periods;
	private int maxPeriods;
	private double maxError;
	private double variationRate;
	private double learningRate;

	public double getVariationRate() {
		return variationRate;
	}

	public double getLearningRate() {
		return learningRate;
	}

	public int getInputLayerSize() {
		return inputLayerSize;
	}

	public int getHiddenLayerSize() {
		return hiddenLayerSize;
	}

	public int getOutputLayerSize() {
		return outputLayerSize;
	}

	public double getMaxError() {
		return maxError;
	}

	public List<Double> getErrors() {
		return errors;
	}

	public int getPeriods() {
		return periods;
	}

	public int getMaxPeriods() {
		return maxPeriods;
	}

	public MultilayerPerceptron(int inputLayerSize, int hiddenLayerSize, int outputLayerSize, double learningRate, double variationRate, int maxPeriods, double maxError) {
		this.inputLayerSize = inputLayerSize;
		this.hiddenLayerSize = hiddenLayerSize;
		this.outputLayerSize = outputLayerSize;
		this.maxPeriods = maxPeriods;
		this.variationRate = variationRate;
		this.learningRate = learningRate;
		this.maxError = maxError;

		// INIT HIDDEN LAYER
		processUnitHidden = new ProcessUnit[hiddenLayerSize];

		for (int i = 0; i < processUnitHidden.length; i++) {
			processUnitHidden[i] = new ProcessUnit(learningRate, variationRate);
			processUnitHidden[i].initWeights(inputLayerSize);
		}

		// INIT OUTPUT LAYER

		processUnitsOutput = new ProcessUnit[outputLayerSize];

		for (int i = 0; i < processUnitsOutput.length; i++) {
			processUnitsOutput[i] = new ProcessUnit(learningRate, variationRate);
			processUnitsOutput[i].initWeights(hiddenLayerSize);
		}
	}

	public double[] output(double[] input) {
		// CALCULATE OUTPUT LAYER HIDDEN
		double[] V = new double[hiddenLayerSize];
		for (int i = 0; i < processUnitHidden.length; i++) {
			double activation = processUnitHidden[i].activation(input);
			V[i] = processUnitHidden[i].output(activation);
		}

		// CALCULATE OUTPUT
		double[] O = new double[outputLayerSize];
		for (int i = 0; i < processUnitsOutput.length; i++) {
			double activation = processUnitsOutput[i].activation(V);
			O[i] = processUnitsOutput[i].output(activation);
		}

		return O;
	}

	public double calculateError(double[][] desiredOutput, double[][] inputs) {
		double error = 0;
		for (int i = 0; i < inputs.length; i++) {

			double[] output = output(inputs[i]);

			for (int j = 0; j < output.length; j++) {
				error += Math.pow(desiredOutput[i][j] - output[j], 2) / 2;
			}

		}

		return error;
	}

	public int training(double[][] trainingValues, double[][] desiredOutput) {
		periods = 0;
		errors = new LinkedList<Double>();
		double error = 0;
		do {

			double[] activationHiddenLayer = new double[hiddenLayerSize];
			double[] activationOutputLayer = new double[outputLayerSize];

			double[] deltaHiddenLayer = new double[hiddenLayerSize];
			double[] deltaOutputLayer = new double[outputLayerSize];

			double[] V = new double[hiddenLayerSize];
			double[] O = new double[outputLayerSize];

			for (int e = 0; e < trainingValues.length; e++) {

				// CALCULATE OUTPUT FORWARD
				for (int i = 0; i < processUnitHidden.length; i++) {
					activationHiddenLayer[i] = processUnitHidden[i].activation(trainingValues[e]);
					V[i] = processUnitHidden[i].output(activationHiddenLayer[i]);
				}

				for (int i = 0; i < processUnitsOutput.length; i++) {
					activationOutputLayer[i] = processUnitsOutput[i].activation(V);
					O[i] = processUnitsOutput[i].output(activationOutputLayer[i]);
				}

				// UPDATE WEIGHTS BACKWARDS

				for (int i = 0; i < processUnitsOutput.length; i++) {
					deltaOutputLayer[i] = processUnitsOutput[i].delta(desiredOutput[e][i], O[i]);
					processUnitsOutput[i].updateWeights(V, deltaOutputLayer[i], activationOutputLayer[i]);
				}

				for (int i = 0; i < processUnitHidden.length; i++) {

					double[] W = new double[outputLayerSize];

					for (int j = 0; j < processUnitsOutput.length; j++) {
						W[j] = processUnitsOutput[j].getWeights()[i];
					}

					deltaHiddenLayer[i] = processUnitHidden[i].delta(deltaOutputLayer, W);
					processUnitHidden[i].updateWeights(trainingValues[e], deltaHiddenLayer[i], activationHiddenLayer[i]);
				}
			}
			periods++;

			// CCALCULATE ERROR
			error = calculateError(desiredOutput, trainingValues);
			errors.add(error);
		} while (error > maxError && periods < maxPeriods);
		return periods;
	}

}

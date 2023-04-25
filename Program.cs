using System;
using System.Collections.Generic;
using AITest;
using Microsoft.ML;
using Microsoft.ML.Data;


class Program
{
    static void Main(string[] args)
    {
        List<Tuple<double[], double[]>> trainingData = new List<Tuple<double[], double[]>>()
        {
            // Площадь: 50 кв.м., комнат: 1, цена: 100000
            new Tuple<double[], double[]>(new double[] { 50, 1 }, new double[] { 100000 }),
            // Площадь: 70 кв.м., комнат: 2, цена: 150000
            new Tuple<double[], double[]>(new double[] { 70, 2 }, new double[] { 150000 }),
            // Площадь: 90 кв.м., комнат: 3, цена: 200000
            new Tuple<double[], double[]>(new double[] { 90, 3 }, new double[] { 200000 }),
            // и т.д.
        };
        int[] layerSizes = new int[] { 2, 5, 1 }; // 2 входных нейрона, 5 нейронов в скрытом слое и 1 выходной нейрон
        Func<double, double> activationFunction = ActivationFucn.LeakyReLU;
        NeuralNetwork network = new NeuralNetwork(layerSizes, activationFunction);
        int epochs = 1000;
        double learningRate = 0.01;
        Trainer.Train(network, trainingData, epochs, learningRate);

        //
        double[] inputForPrediction = new double[] { 80, 2 }; // Площадь: 80 кв.м., комнат: 2
        double[] predictedPrice = network.FeedForward(inputForPrediction);
        Console.WriteLine($"Predicted price: {predictedPrice[0]}");
    }
}
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AITest
{
    internal static class Trainer
    {
        public static void Train(NeuralNetwork network, List<Tuple<double[], double[]>> trainingData, int epochs, double learningRate)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                foreach (var dataPoint in trainingData)
                {
                    double[] inputs = dataPoint.Item1;
                    double[] targetOutputs = dataPoint.Item2;

                    // Выполните прямое и обратное распространение ошибки для обновления весов и смещений
                    network.FeedForward(inputs);
                    network.Backpropagate(inputs, targetOutputs, learningRate);
                }
            }
        }
    }
}

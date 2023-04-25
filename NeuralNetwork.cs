using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AITest
{
    /// <summary>
    /// Класс NeuralNetwork представляет нейронную сеть, состоящую из слоев нейронов.
    /// Нейронная сеть применяется для задач машинного обучения, таких как классификация, регрессия и генерация данных.
    /// </summary>
    public class NeuralNetwork
    {
        /// <summary>
        /// Список слоев нейронов, представляющих нейронную сеть.
        /// Каждый элемент списка содержит экземпляр класса Layer, который представляет отдельный слой нейронов в сети.
        /// </summary>
        private Layer[] Layers;

        /// <summary>
        /// Конструктор для создания нейронной сети.
        /// Принимает массив целых чисел, где каждое число определяет количество нейронов в соответствующем слое.
        /// </summary>
        /// <param name="layerSizes">Массив, определяющий количество нейронов в каждом слое нейронной сети.</param>
        public NeuralNetwork(int[] layerSizes, Func<double, double> activationFunction)
        {
            Layers = new Layer[layerSizes.Length - 1];

            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i] = new Layer(layerSizes[i + 1], layerSizes[i], activationFunction);
            }
        }


        /// <summary>
        /// Применяет процедуру прямого распространения на входные данные, чтобы получить выходные данные нейронной сети.
        /// </summary>
        /// <param name="inputs">Массив входных данных, который передается первому слою нейронов.</param>
        /// <returns>Массив выходных данных, полученных после применения процедуры прямого распространения.</returns>
        public double[] FeedForward(double[] inputs)
        {
            double[] activations = inputs;
            for (int i = 0; i < Layers.Length; i++)
            {
                activations = CalculateLayerOutput(Layers[i], activations);
            }

            return activations;
        }

        private double[] CalculateLayerOutput(Layer layer, double[] inputs)
        {
            double[] outputs = new double[layer.Neurons.Length];

            for (int i = 0; i < layer.Neurons.Length; i++)
            {
                double weightedSum = 0.0;
                for (int j = 0; j < layer.Neurons[i].Weights.Length; j++)
                {
                    weightedSum += inputs[j] * layer.Neurons[i].Weights[j];
                }

                weightedSum += layer.Neurons[i].Bias;
                outputs[i] = layer.Neurons[i].ActivationFunction(weightedSum);
            }

            return outputs;
        }

        public void Backpropagate(double[] inputs, double[] targetOutputs, double learningRate)
        {
            // Прямое распространение для получения выходных значений
            double[] outputs = FeedForward(inputs);

            // Вычисление градиента функции потерь по выходным значениям (Сейчас используется LeakyReLu)
            double[] outputGradients = Calculate.GradientLossFunc_LeakyReLu(outputs, targetOutputs);

            // Обновление весов и смещений для последнего слоя
            for (int i = 0; i < Layers[^1].Neurons.Length; i++)
            {
                for (int j = 0; j < Layers[^1].Neurons[i].Weights.Length; j++)
                {
                    Layers[^1].Neurons[i].Weights[j] -= learningRate * outputGradients[i] * Layers[^2].Neurons[i].ActivationFunction(inputs[j]);
                }
                Layers[^1].Neurons[i].Bias -= learningRate * outputGradients[i];
            }

            // Обратное распространение ошибки и обновление весов и смещений для остальных слоев
            double[] previousLayerOutputs = Layers[^2].FeedForward(inputs);
            for (int i = Layers.Length - 2; i >= 0; i--)
            {
                double[] currentLayerGradients = new double[Layers[i].Neurons.Length];

                for (int j = 0; j < Layers[i].Neurons.Length; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < Layers[i + 1].Neurons.Length; k++)
                    {
                        sum += Layers[i + 1].Neurons[k].Weights[j] * outputGradients[k];
                    }

                    currentLayerGradients[j] = sum * Layers[i].Neurons[j].ActivationFunction(inputs[j]) * (1 - Layers[i].Neurons[j].ActivationFunction(inputs[j]));

                    for (int l = 0; l < Layers[i].Neurons[j].Weights.Length; l++)
                    {
                        Layers[i].Neurons[j].Weights[l] -= learningRate * currentLayerGradients[j] * (i > 0 ? previousLayerOutputs[l] : inputs[l]);
                    }
                    Layers[i].Neurons[j].Bias -= learningRate * currentLayerGradients[j];
                }

                // Если мы не достигли входного слоя, обновляем выходные значения предыдущего слоя
                if (i > 0)
                {
                    previousLayerOutputs = Layers[i - 1].FeedForward(inputs);
                }
                outputGradients = currentLayerGradients;
            }
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AITest
{
    /*
     * Класс Layer представляет слой нейронов в нейронной сети.
     * Слой содержит массив нейронов и предоставляет методы для выполнения операций на уровне слоя.
     */
    public class Layer
    {
        // Массив нейронов в слое.
        public Neuron[] Neurons { get; }

        /*
         * Конструктор для создания нового объекта слоя с заданным количеством нейронов,
         * количеством входных данных для каждого нейрона и функцией активации.
         */
        public Layer(int numberOfNeurons, int numberOfInputsPerNeuron, Func<double, double> activationFunction)
        {
            Neurons = new Neuron[numberOfNeurons];
            for (int i = 0; i < numberOfNeurons; i++)
            {
                // Создание случайных весов и смещения для каждого нейрона.
                double[] weights = new double[numberOfInputsPerNeuron];
                for (int j = 0; j < numberOfInputsPerNeuron; j++)
                {
                    weights[j] = new Random().NextDouble() * 2 - 1;
                }
                double bias = new Random().NextDouble() * 2 - 1;

                // Создание объекта нейрона с заданными весами, смещением и функцией активации.
                Neurons[i] = new Neuron(weights, bias, activationFunction);
            }
        }

        /*
         * Метод FeedForward принимает входные данные для слоя и вычисляет выходные значения для всех нейронов в слое.
         * Входные данные представляют собой массив чисел, каждое из которых соответствует выходному значению нейрона из предыдущего слоя.
         */
        public double[] FeedForward(double[] inputs)
        {
            // Вычисление выходных значений для всех нейронов в слое.
            double[] outputs = new double[Neurons.Length];
            for (int i = 0; i < Neurons.Length; i++)
            {
                outputs[i] = Neurons[i].FeedForward(inputs);
            }

            // Возвращение массива выходных значений нейронов текущего слоя.
            return outputs;
        }
    }
}

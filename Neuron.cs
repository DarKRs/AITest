using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AITest
{
    // Класс Neuron представляет один нейрон в нейронной сети.
    public class Neuron
    {
        // Веса нейрона - это числовые значения, которые определяют силу связи между этим нейроном и предыдущим слоем.
        public double[] Weights { get; }

        // Смещение нейрона - это дополнительное числовое значение, которое добавляется к взвешенной сумме входов нейрона.
        public double Bias { get; set; }

        // Функция активации определяет, как нейрон будет реагировать на входные данные.
        private Func<double, double> _activationFunction;

        public Func<double, double> ActivationFunction => _activationFunction;

        // Конструктор для создания нового объекта нейрона с заданными весами, смещением и функцией активации.
        public Neuron(double[] weights, double bias, Func<double, double> activationFunction)
        {
            Weights = weights;
            Bias = bias;
            _activationFunction = activationFunction;
        }

        // Метод FeedForward принимает входные данные (inputs) и вычисляет выходное значение нейрона.
        // Входные данные представляют собой массив чисел, каждое из которых соответствует выходному значению нейрона из предыдущего слоя.
        public double FeedForward(double[] inputs)
        {
            // Вычисление взвешенной суммы входов и смещения.
            double weightedSum = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                weightedSum += inputs[i] * Weights[i];
            }
            weightedSum += Bias;

            // Применение функции активации к взвешенной сумме и возврат результата в качестве выходного значения нейрона.
            return _activationFunction(weightedSum);
        }
    }
}

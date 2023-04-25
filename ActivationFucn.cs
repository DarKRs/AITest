using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AITest
{
    internal class ActivationFucn
    {
        public static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }


        public static double LeakyReLU(double x)
        {
            double alpha = 0.01;
            return x >= 0 ? x : alpha * x;
        }

        //Производная функции leakyReLu
        public static double LeakyReLUDerivative(double x, double alpha = 0.01)
        {
            return x >= 0 ? 1 : alpha;
        }
    }
}

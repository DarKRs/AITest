using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AITest
{
    internal static class Calculate
    {
        public static double[] GradientLossFunc_Sigmoid(double[] outputs, double[] targetOutputs)
        {
            double[] outputGradients = new double[outputs.Length];
            for (int i = 0; i < outputs.Length; i++)
            {
                outputGradients[i] = 2 * (outputs[i] - targetOutputs[i]) * outputs[i] * (1 - outputs[i]);
            }
            return outputGradients;
        }

        public static double[] GradientLossFunc_LeakyReLu(double[] outputs, double[] targetOutputs)
        {
            double[] outputGradients = new double[outputs.Length];
            for (int i = 0; i < outputs.Length; i++)
            {
                outputGradients[i] = 2 * (outputs[i] - targetOutputs[i]) * ActivationFucn.LeakyReLUDerivative(outputs[i]);
            }
            return outputGradients;
        }

    }
}

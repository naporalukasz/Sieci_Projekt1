using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sieci_Projekt1
{
    class Parameters
    {
        public List<int> Layers { get; set; }

        public int CountInput { get; set; }

        public int CountOutput { get; set; }

        public FunctionTypeEnum FunctionType { get; set; }

        public bool Bias { get; set; }

        public int IterationsCount { get; set; }

        public double LearingCoefficient { get; set; }

        public double InertiaCoefficient { get; set; }

        public ProblemTypeEnum ProblemType { get; set; }

        public double AcceptedError { get; set; }

        public string TrainFile { get; set; }

        public string TestFile { get; set; }

        public void Construct(
            List<int> layers,
            FunctionTypeEnum function,
            bool bias,
            int iterations,
            double learning,
            double inertia,
            ProblemTypeEnum problem,
            double error = 0.001)
        {
            Layers = layers;
            FunctionType = function;
            Bias = bias;
            IterationsCount = iterations;
            LearingCoefficient = learning;
            InertiaCoefficient = inertia;
            ProblemType = problem;
            AcceptedError = error;
        }

    }
}

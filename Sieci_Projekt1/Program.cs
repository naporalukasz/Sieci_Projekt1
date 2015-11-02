using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sieci_Projekt1
{
    class Program
    {
        static void Main(string[] args)
        {

            var parameters = new Parameters
            {
                Layers = new List<int> { 5,5 },
                CountInput = 1,
                CountOutput = 1,
                FunctionType = FunctionTypeEnum.Unipolar,
                Bias = true,
                IterationsCount = 5000,
                LearingCoefficient = 0.2,
                InertiaCoefficient = 0.9,
                ProblemType = ProblemTypeEnum.Regression
            };

            var problem = new Problem(parameters);
            problem.Execute();


           // Console.ReadKey();
        }
    }
}

using System;
using System.Collections.Generic;
using System.Configuration;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sieci_Projekt1
{
    class Program
    {
        static void Main(string[] args)
        {
            int countInput,countOutput,iterationsCount;
            List<int> layers = new List<int>();
            FunctionTypeEnum functionType;
            ProblemTypeEnum problemType;
            bool bias;
            double learningCoefficient, inertiaCoefficient;

            #region Handle configuration

            var countInputString = ConfigurationManager.AppSettings["CountInput"];
            if (!int.TryParse(countInputString, out countInput) || countInput<=0) { HandleError("CountInput"); return; }

            var countOutputString = ConfigurationManager.AppSettings["CountOutput"];
            if (!int.TryParse(countOutputString, out countOutput) || countOutput<=0) { HandleError("CountOutput"); return; }

            var iterationsCountString = ConfigurationManager.AppSettings["IterationsCount"];
            if (!int.TryParse(iterationsCountString, out iterationsCount) || iterationsCount <= 0) { HandleError("IterationsCount"); return; }

            var layerString = ConfigurationManager.AppSettings["Layers"];
            var layerStrings = layerString.Split(',');
            foreach(var layerStr in layerStrings)
            {
                int layerValue;
                if (!int.TryParse(layerStr, out layerValue) || layerValue <= 0) { HandleError("Layers"); return; }
                else { layers.Add(layerValue); }
            }

            var functionTypeString = ConfigurationManager.AppSettings["FunctionType"];
            if (functionTypeString == "Bipolar") functionType = FunctionTypeEnum.Bipolar;
            else if (functionTypeString == "Unipolar") functionType = FunctionTypeEnum.Unipolar;
            else { HandleError("FunctionType"); return; }

            var problemTypeString = ConfigurationManager.AppSettings["ProblemType"];
            if (problemTypeString == "Classification") problemType = ProblemTypeEnum.Classification;
            else if (problemTypeString == "Regression") problemType = ProblemTypeEnum.Regression;
            else { HandleError("ProblemType"); return; }

            var biasString = ConfigurationManager.AppSettings["Bias"];
            if (biasString == "true") bias = true;
            else if (biasString == "false") bias = false;
            else { HandleError("Bias"); return; }

            var learningCoefficientString = ConfigurationManager.AppSettings["LearningCoefficient"];
            if (!double.TryParse(learningCoefficientString,NumberStyles.AllowDecimalPoint,new CultureInfo("en-US"), out learningCoefficient) || learningCoefficient <= 0 || learningCoefficient >= 1) { HandleError("LearningCoefficient"); return; }

            var inertiaCoefficientString = ConfigurationManager.AppSettings["InertiaCoefficient"];
            if (!double.TryParse(inertiaCoefficientString, NumberStyles.AllowDecimalPoint, new CultureInfo("en-US"), out inertiaCoefficient) || inertiaCoefficient <= 0 || inertiaCoefficient >= 1) { HandleError("InertiaCoefficient"); return; }
            #endregion

            var parameters = new Parameters
            {
                Layers = layers,
                CountInput =countInput,
                CountOutput = countOutput,
                FunctionType = functionType,
                Bias = bias,
                IterationsCount = iterationsCount,
                LearingCoefficient = learningCoefficient,
                InertiaCoefficient = inertiaCoefficient,
                ProblemType = problemType
            };

            var problem = new Problem(parameters);
            problem.Execute();

        }

        private static void HandleError(string settingName)
        {
            Console.WriteLine("Wrong data for property " + settingName);
            Console.ReadKey();
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Configuration;
using System.Text;
using System.Threading.Tasks;
using System.Globalization;
using System.IO;

using Encog;
using Encog.Engine.Network.Activation;
using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.ML.Train;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Back;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Encog.Neural.NeuralData;
using Encog.Neural.Data.Basic;
using Encog.ML.Data.Versatile;
using Encog.ML.Data.Versatile.Columns;
using Encog.ML.Data.Versatile.Sources;
using Encog.ML.Factory;
using Encog.ML.Model;
using Encog.Util.CSV;
using Encog.ML;
using Encog.ML.Data.Versatile.Normalizers.Strategy;
using Encog.Util.Normalize;
using Encog.App.Analyst;
using Encog.App.Analyst.Wizard;

namespace Sieci_Projekt1
{
    class Problem
    {

        Parameters parameters;    

        public BasicNeuralDataSet TrainSet { get; set; }
        public BasicNeuralDataSet ValidationSet { get; set; }
        public double[][] testSet { get; set; }
        public BasicNeuralDataSet AnswerSet { get; set; }
        public List<double[]> errorSet { get; set; }
        public BasicNetwork network { get; set; }
        public int classCount { get; set; }
        public double[] means { get; set; }
        public double[] factor { get; set; }
        public double ansMeans { get; set; }
        public double ansfactor { get; set; }
        public double[] Max { get; set; }
        public double[] Min { get; set; }
        public double ansMIn { get; set; }

        public Problem (Parameters Parameteras)
        {
            parameters = Parameteras;
        }

        public void Execute()
        {
            if (parameters.ProblemType == ProblemTypeEnum.Classification)
            {
                LoadCSVFileTrainingClass();
                CerateNetwork();
                TrainNetwork();
                SaveError();
                LoadCSVFileTestClass();
            }
            else
            {
                LoadCSVFileTrainingRegg();
                CerateNetwork();
                TrainNetwork();
                SaveError();
                LoadCSVFileTestRegg();
            }

            CalculateAnswer();
            SaveAnswer();
        }

        public void CerateNetwork()
        {
            network = new BasicNetwork();
            network.AddLayer(new BasicLayer(parameters.CountInput));
            foreach( var NueronNumber in parameters.Layers)
            {
                switch (parameters.FunctionType)
                {
                    case FunctionTypeEnum.Bipolar:
                        network.AddLayer(new BasicLayer(new ActivationBiPolar(),parameters.Bias,NueronNumber));
                        break;
                    case FunctionTypeEnum.Unipolar:
                        network.AddLayer(new BasicLayer(new ActivationSigmoid(),parameters.Bias,NueronNumber));//sprawdzić
                        break;
                }
            }
            if(parameters.ProblemType==ProblemTypeEnum.Classification)
                network.AddLayer(new BasicLayer(new ActivationSoftMax(), parameters.Bias, classCount));
            else
                network.AddLayer(new BasicLayer(parameters.CountOutput));
            network.Structure.FinalizeStructure();
            network.Reset();

           
        }

        public void TrainNetwork()
        {
           var  train = new Backpropagation(network, TrainSet);//, parameters.LearingCoefficient, parameters.InertiaCoefficient);
           errorSet = new List<double[]>();// iteracja, bład, błąd walidacji
           int epoch = 1;
           do
           {
               train.Iteration();   
               errorSet.Add(new double[] { epoch, train.Error, network.CalculateError(ValidationSet) }); ;
               epoch++;
           } while (train.Error > 0.01 && epoch < parameters.IterationsCount);

        }

        public void CalculateAnswer()
        {
            var values = new List<double[]>();
            var answers = new List<double[]>();
            var outputSize = parameters.ProblemType == ProblemTypeEnum.Classification ? classCount : parameters.CountOutput;
            foreach (var input in testSet)
            {
                var output = new double[outputSize];
                network.Compute(input, output);
                
                values.Add(input);
                answers.Add(output);
            }

            if (parameters.ProblemType == ProblemTypeEnum.Regression)
            {
                DenormalizationRegg(values,false);
                DenormalizationRegg(answers,true);
            }
           AnswerSet = new BasicNeuralDataSet(values.ToArray(), answers.ToArray());
        }

        public void LoadCSVFileTrainingClass()
        {
            var path = "data.train.csv";
            var csvRead = new ReadCSV(new FileStream(path, FileMode.Open), true, CSVFormat.DecimalPoint);
            var values = new List<double[]>();
            var classes = new List<double[]>();

            while (csvRead.Next())
            {
                values.Add(new double[2] { csvRead.GetDouble(0), csvRead.GetDouble(1) });
                classes.Add(new double[1] { csvRead.GetDouble(2) });
            }
            csvRead.Close();

            Normalization(values);

             classCount = classes.Select(c => c[0]).Distinct().Count();
            var normalizeClasses = new List<double[]>();

            for (int i = 0; i < classes.Count; ++i)
            {
                var newClasses = new double[classCount];
                newClasses[(int)classes[i][0] - 1] = 1;// dodoac normalizacje na -1 
                normalizeClasses.Add(newClasses);
            }

            var trainSetCount = (int)((double)values.Count * ((100.0 - 15) / 100));

            values.Shuffle();
            normalizeClasses.Shuffle();
            MyExtensions.ResetStableShuffle();

            TrainSet = new BasicNeuralDataSet(values.Take(trainSetCount).ToArray(), 
                                                normalizeClasses.Take(trainSetCount).ToArray());
            ValidationSet = new BasicNeuralDataSet(values.Skip(trainSetCount).ToArray(), 
                                                normalizeClasses.Skip(trainSetCount).ToArray());
        }

        public void LoadCSVFileTestClass()
        {
            var path = "data.test.csv";
            var csvRead = new ReadCSV(new FileStream(path, FileMode.Open), true, CSVFormat.DecimalPoint);
            var values = new List<double[]>();
            
            while (csvRead.Next())
                values.Add(new double[2] { csvRead.GetDouble(0), csvRead.GetDouble(1) });               

            csvRead.Close();
            Normalization(values);
            testSet = values.ToArray();
        }

        public void LoadCSVFileTrainingRegg()
        {
            var path = "data.xsq.train.csv";
            var csvRead = new ReadCSV(new FileStream(path, FileMode.Open), true, CSVFormat.DecimalPoint);
            var valuesX = new List<double[]>();
            var valuesY = new List<double[]>();

            while (csvRead.Next())
            {

                valuesX.Add(new double[1] { csvRead.GetDouble(0) });
                valuesY.Add(new double[1] { csvRead.GetDouble(1) });

            }
            csvRead.Close();

            Normalization(valuesX);
            Normalization(valuesY);
            //if(parameters.FunctionType==FunctionTypeEnum.Bipolar)
                    ansMeans = means[0];
            ansfactor = factor[0];
       
            //ansMIn = Min[0];
            var trainSetCount = (int)((double)valuesX.Count * ((100.0 - 15) / 100));

            valuesX.Shuffle();
            valuesY.Shuffle();
            MyExtensions.ResetStableShuffle();

            TrainSet = new BasicNeuralDataSet(valuesX.Take(trainSetCount).ToArray(), 
                                                valuesY.Take(trainSetCount).ToArray());
            ValidationSet = new BasicNeuralDataSet(valuesX.Skip(trainSetCount).ToArray(), 
                                                valuesY.Skip(trainSetCount).ToArray());
        }

        public void LoadCSVFileTestRegg()
        {
            var path = "data.xsq.test.csv";
            var csvRead = new ReadCSV(new FileStream(path, FileMode.Open), true, CSVFormat.DecimalPoint);
            var values = new List<double[]>();

            while (csvRead.Next())
                values.Add(new double[1] { csvRead.GetDouble(0) });

            csvRead.Close();
            Normalization(values);
            testSet = values.ToArray();
        }

        public void SaveAnswer()
        {
            
            var answerPath =  "data.solved.csv";
            var lines = new List<string>();
            var header = parameters.ProblemType == ProblemTypeEnum.Classification ? "x,y,cls" : "x,y";

            lines.Add( header);
            if(parameters.ProblemType == ProblemTypeEnum.Classification)
                lines.AddRange(AnswerSet.Select(r => GetTextClass(r)));
            else
                lines.AddRange(AnswerSet.Select(r => GetTextRegg(r)));
            File.WriteAllLines(answerPath, lines);
           // AnswerPath = answerPath;
        
        }

        public void SaveError()
        {
            var errorPath = "data.error.csv";
            var lines = new List<string>();
            lines.Add("iter,TrainError,ValidationError");
            lines.AddRange(errorSet.Select(r => GetErrorText(r)));
            File.WriteAllLines(errorPath, lines);
        }

        private string GetErrorText(double[] line)
        {
            return line[0].ToString(CultureInfo.InvariantCulture) + ',' +
                    line[1].ToString(CultureInfo.InvariantCulture) + ',' +
                    line[2].ToString(CultureInfo.InvariantCulture);
        }

        private string GetTextRegg(IMLDataPair data)
        {
            return data.Input[0].ToString(CultureInfo.InvariantCulture) +
                    "," + data.Ideal[0].ToString(CultureInfo.InvariantCulture);
        }

        private string GetTextClass(IMLDataPair data)
        {
                return data.Input[0].ToString(CultureInfo.InvariantCulture) + 
                    "," + data.Input[1].ToString(CultureInfo.InvariantCulture) + 
                    "," + GetClassFromResult(data);
        }

        private int GetClassFromResult(IMLDataPair result)
        {
            int classValue = 0;
            double classFit = result.Ideal[0];

            for (int i = 1; i < classCount; ++i)
                if (classFit <= result.Ideal[i])
                {
                    classFit = result.Ideal[i];
                    classValue = i;
                }

            return classValue + 1;
        }

        private int DenormalizeClass(IMLDataPair data)
        {
            var finalClass= 0;
            var outputClass = data.Ideal[0];
            for (int i =1 ; i<3;i++) // zmienic na zależność od ilosci klas
            {
                if (outputClass < data.Ideal[i])
                    finalClass = i ;
            }

            return finalClass + 1;
        }

        private void Normalization(List<double[]> values)
        {
           

            if (parameters.ProblemType == ProblemTypeEnum.Classification)
            {
                Max = new double[] { values.Max(v => v[0]), values.Max(v => v[1]) };
                Min = new double[] { values.Min(v => v[0]), values.Min(v => v[1]) };
            }
            else
            {
                Max = new double[] { values.Max(v => v[0])};
                Min = new double[] { values.Min(v => v[0]) };
            }
            //if (parameters.FunctionType == FunctionTypeEnum.Bipolar)
            //{


                var columnCount = values[0].Length;
                means = new double[columnCount];

                foreach (var value in values)
                    for (int i = 0; i < columnCount; ++i)
                        means[i] += value[i];

                for (int i = 0; i < columnCount; ++i)
                    means[i] = means[i] / values.Count;

                factor = new double[columnCount];
                for (int i = 0; i < columnCount; i++)
                    if (Math.Abs(Max[i] - means[i]) > Math.Abs(Min[i] - means[i]))
                        factor[i] = Math.Abs(Max[i] - means[i]);
                    else
                        factor[i] = Math.Abs(Min[i] - means[i]);

                foreach (var value in values)
                    for (int i = 0; i < columnCount; ++i)
                        value[i] = (value[i] - means[i]) / factor[i];
            //}
            //else
            //{

            //    var columnCount = values[0].Length;
            //    factor = new double[columnCount];
            //    for (int i = 0; i < columnCount; i++)
            //            factor[i] = Math.Abs(Max[i] - Min[i]);

            //    foreach (var value in values)
            //        for (int i = 0; i < columnCount; ++i)
            //            value[i] = (value[i] - Min[i]) / factor[i];
                   
            //}
        
        } 
  
        private void DenormalizationRegg(List<double[]> values, bool answer)
        {
            //if (parameters.FunctionType == FunctionTypeEnum.Bipolar)
            //{
                foreach (var value in values)
                    for (int i = 0; i < values[0].Length; ++i)
                    {
                        if (answer)
                            value[i] = value[i] * ansfactor + ansMeans;
                        else
                            value[i] = value[i] * factor[0] + means[0];
                    }
            //}
            //else
            //{
            //    foreach (var value in values)
            //        for (int i = 0; i < values[0].Length; ++i)
            //        {
            //            if (answer)
            //                value[i] = value[i] * ansfactor + ansMIn;
            //            else
            //                value[i] = value[i] * factor[0] + Min[0];
            //        }
            //}

        }
    }

    public static class MyExtensions
    {

        private static int? seed = null;
        private static Random seedGenerator = new Random();

        public static void Shuffle<T>(this IList<T> list)
        {
            if (seed == null)
                seed = seedGenerator.Next();

            var rng = new Random(seed.Value);

            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

        public static void ResetStableShuffle()
        {
            seed = null;
        }
    }
}

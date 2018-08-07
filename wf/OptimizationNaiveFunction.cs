using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataSetsSparsity
{
    class OptimizationNaiveFunction : LibOptimization.Optimization.absObjectiveFunction
    {
        private double[][][] m_data;
        private int m_dim;
        public OptimizationNaiveFunction(double [][][] data)
        {
            this.m_data = data;
            this.m_dim = data[0][0].Count() + 1;
        }

        public override double F(List<double> xtmp)
        {
            double func = 0;
            if (xtmp.Count() != this.m_dim)
            {
                throw new Exception("Somthing wrong with the state vector dimnsions");
            }
            double[] x = new double[this.m_dim];
            for (int i = 0; i < this.m_dim; i++)
            {
                x[i] = xtmp[i];
            }

            // clculate the mean of each side
            double[] mean0 = new double[m_data[1][0].Count()];
            double[] mean1 = new double[m_data[1][0].Count()];

            double error0 = 0;
            double error1 = 0;

            List<double[]> valuesChild0 = new List<double[]>();
            List<double[]> valuesChild1 = new List<double[]>();

            for (int passingIndex = 0; passingIndex < m_data[0].Count(); passingIndex++)
            {
                double score = 0;
                // compute score (up or down the hyperplane)
                for (int positionIndex = 0; positionIndex < m_data[0][0].Count(); positionIndex++)
                {
                    score += x[positionIndex] * m_data[0][passingIndex][positionIndex];
                }

                if (score > 0) // go to child 0 
                {
                    valuesChild0.Add(m_data[1][passingIndex]);
                    for (int valueIndex = 0; valueIndex < m_data[1][0].Count(); valueIndex++)
                    {
                        mean0[valueIndex] += m_data[1][passingIndex][valueIndex];
                    }

                }
                else // go to child1
                {
                    valuesChild1.Add(m_data[1][passingIndex]);
                    for (int valueIndex = 0; valueIndex < m_data[1][0].Count(); valueIndex++)
                    {
                        mean1[valueIndex] += m_data[1][passingIndex][valueIndex];
                    }
                }
            }

            // devide the mean (so ... it will be actully be mean)
            for (int valueIndex = 0; valueIndex < m_data[1][0].Count(); valueIndex++)
            {
                if (valuesChild0.Count() != 0)
                {
                    mean0[valueIndex] /= valuesChild0.Count();
                }


                if (valuesChild1.Count() != 0)
                {
                    mean1[valueIndex] /= valuesChild1.Count();
                }
            }
            error0 = decicionTree.CalculateErrorOFChild(mean0, valuesChild0);
            error1 = decicionTree.CalculateErrorOFChild(mean1, valuesChild1);

            int count0 = valuesChild0.Count();
            int count1 = valuesChild1.Count();
            bool flagError =(double) Math.Max(count1, count0) /(double) Math.Min(count0, count1) > 50;

            if (count1 == 0 || count0 == 0 || flagError)
            {
                Random rnd = new Random();

                error1 += rnd.Next(100, 120);
            }


            // List<int> originalIndexList = tree;
            //for (int tmpIndex = 0; t)
            func = error0 + error1;
            return func;
        }

        public override List<double> Gradient(List<double> x)
        {
            return null;
        }

        public override List<List<double>> Hessian(List<double> x)
        {
            return null;
        }

        public override int NumberOfVariable()
        {
            return this.m_dim;
        }
    }
}

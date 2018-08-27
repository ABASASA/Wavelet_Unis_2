using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataSetsSparsity
{
    class Optimization2MeansSVMFunctiton : LibOptimization.Optimization.absObjectiveFunction
    {
        private double[][][] m_data;
        private int m_dim;
        public Optimization2MeansSVMFunctiton(double[][][] data)
        {
            this.m_data = data;
            this.m_dim = data[0][0].Count() + 1;
        }

        public override double F(List<double> xtmp)
        {
            double func = 0;
            // Cast bsck to tree
            double[][][] dataForOptimizer = m_data;
            if (xtmp.Count() != this.m_dim)
            {
                throw new Exception("Somthing wrong with the state vector dimnsions");
            }
            double[] x = new double[this.m_dim];
            for (int i = 0; i < this.m_dim; i++)
            {
                x[i] = xtmp[i];
            }
            double error = 0;

            List<double[]> valuesChild0 = new List<double[]>();
            List<double[]> valuesChild1 = new List<double[]>();

            for (int passingIndex = 0; passingIndex < dataForOptimizer[0].Count(); passingIndex++)
            {
                double score = 0;
                // compute score (up or down the hyperplane)
                for (int positionIndex = 0; positionIndex < dataForOptimizer[0][0].Count() - 1; positionIndex++)
                {
                    score += x[positionIndex] * dataForOptimizer[0][passingIndex][positionIndex];
                }
                score += x[dataForOptimizer[0][0].Count() - 1];
                error += Math.Max(0, dataForOptimizer[1][passingIndex][0] * score);

                if (score > 0) // go to child 0 
                {
                    valuesChild0.Add(dataForOptimizer[1][passingIndex]);
                }
                else // go to child1
                {
                    valuesChild1.Add(dataForOptimizer[1][passingIndex]);
                }
            }
            /*double norm = 0;
            for (int positionIndex = 0; positionIndex < x.Count() - 1; positionIndex++)
            {
                norm += x[positionIndex] * x[positionIndex];
            }
            norm = Math.Sqrt(norm);*/

            error /= dataForOptimizer[0].Count(); // avrage the score

            int count0 = valuesChild0.Count();
            int count1 = valuesChild1.Count();            

            // adding extra loss if the 
            if (count1 == 0 || count0 == 0)
            {
                Random rnd = new Random();
                error += rnd.Next(100, 120);

            }else
            {
                double lambda = Math.Abs( 0.001 * Math.Log((double)dataForOptimizer[0].Count()) );
                double proportion = Math.Abs(Math.Log10((double)valuesChild0.Count() / (double)valuesChild1.Count()) / Math.Log10(2));
                error += proportion * lambda;
            }

            // List<int> originalIndexList = tree;
            //for (int tmpIndex = 0; t)
            func = error;
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

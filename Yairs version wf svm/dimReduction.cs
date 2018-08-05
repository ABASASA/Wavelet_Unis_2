using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using Accord.Statistics.Analysis;

namespace DataSetsSparsity
{
    class dimReduction
    {
        public dimReduction(double[][] training_matrix) 
        {
            //Create the Principal Component Analysis
            pca = new PrincipalComponentAnalysis(training_matrix); 
            pca.Compute();

            Form1.printList(pca.Eigenvalues.ToList(), Form1.MainFolderName + "eigvalues.txt");
        }
        PrincipalComponentAnalysis pca;

        public double[][] getPCA(double[][] matrix)
        {
            return pca.Transform(matrix);
        }
    }
}

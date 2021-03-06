﻿using System.Linq;
using Accord.Statistics.Analysis;

namespace DataScienceAnalysis
{
    class DimReduction
    {
        private readonly ModifedPca _pca;
        public DimReduction(double[][] trainingMatrix) 
        {
            //Create the Principal Component Analysis
            _pca = new ModifedPca(trainingMatrix); 
            _pca.Compute();
            PrintEngine.printList(_pca.Eigenvalues.ToList(), Form1.MainFolderName + "eigvalues.txt");
        }

     

        public double[][] getGlobalPca(double[][] matrix)
        {
            return _pca.Transform(matrix);
        }

        //construct node pca and return original (before transform) node matrix
        public static double[][] constructNodePca(double[][] trainingAll, GeoWave node)
        {
            double[][] nodeMatrix = node.pointsIdArray.Select(id => trainingAll[id]).ToArray();
            node.localPca = new ModifedPca(nodeMatrix,AnalysisMethod.Standardize); 
            node.localPca.Compute();
            return nodeMatrix;
        }

        public static void constructNodePcaByOriginalData(double[][] nodeOriginalData, GeoWave node)
        {
            node.localPca = new ModifedPca(nodeOriginalData, AnalysisMethod.Standardize);
            node.localPca.Compute();
        }

    }
}

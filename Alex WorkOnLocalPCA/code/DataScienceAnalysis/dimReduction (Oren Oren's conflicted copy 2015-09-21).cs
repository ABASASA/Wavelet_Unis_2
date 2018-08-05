using System.Linq;

namespace DataScienceAnalysis
{
    class DimReduction
    {
        private ModifedPca _pca;
        public DimReduction(double[][] trainingMatrix) 
        {
            //Create the Principal Component Analysis
            _pca = new ModifedPca(trainingMatrix); 
            _pca.Compute();
            PrintEngine.printList(_pca.Eigenvalues.ToList(), Form1.MainFolderName + "eigvalues.txt");
        }

        public static void originalFeatureRole(ModifedPca nodePca)
        {
            //TO DO: baced on http://venom.cs.utsa.edu/dmz/techrep/2007/CS-TR-2007-011.pdf

        }

        public double[][] getGlobalPca(double[][] matrix)
        {
            return _pca.Transform(matrix);
        }

        //construct node pca and return original (before transform) node matrix
        public static double[][] constructNodePca(double[][] trainingAll, GeoWave node)
        {
            double[][] nodeMatrix = node.pointsIdArray.Select(id => trainingAll[id]).ToArray();
            node.localPca = new ModifedPca(nodeMatrix); 
            node.localPca.Compute();
            return nodeMatrix;
        }
    }
}

using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Windows.Forms.VisualStyles;
using Accord.Math;
using Accord.Math.Decompositions;

namespace DataScienceAnalysis
{
    class DiffusionMaps
    {
        public static MLApp.MLApp matlab=new MLApp.MLApp();
       
        public static double[][] getTransformedMatrix( double[][] data, double p)
        {
            // MLApp.MLApp matlab = new MLApp.MLApp();
             const string pathToMatlabFolderAmazon = @"cd C:\Users\Administrator\Desktop\AlexDropbox\Dropbox\Alex-Oren\matlabDiffusionMaps\diff_code";
            const string pathToMatlabFolderLocal = @"cd C:\Users\Alex\Dropbox\Alex-Oren\matlabDiffusionMaps\diff_code";
            const string checkAmazonFolderExist= "C:\\Users\\Administrator\\Desktop\\AlexDropbox\\Dropbox\\Alex-Oren\\matlabDiffusionMaps\\diff_code";
            string executePath = (Directory.Exists(checkAmazonFolderExist))
                ? pathToMatlabFolderAmazon
                : pathToMatlabFolderLocal;
           // MLApp.MLApp matlab = new MLApp.MLApp();
           // matlab.Execute(@"cd C:\Users\Alex\Dropbox\Alex-Oren\matlabDiffusionMaps\diff_code");
            // Define the output 
            List<string[]> strTransData;
            try
            {
                matlab.Execute(executePath);
                //convert matrix to matlab type string
                 string strMat="[";
                foreach (double[] row in data)
                {
                    strMat = row.Aggregate(strMat, (current, val) => current + (val.ToString(CultureInfo.InvariantCulture) + ","));
                    strMat = strMat.Remove(strMat.Length - 1);
                    strMat += ";";
                }
                strMat += "]";
                object res;
                matlab.Feval("sharpCallVisual", 1, out res, strMat, p);

                ArrayList a = new ArrayList((object[])res);
                //can't calcualate eigenvectors
                if (a[0].ToString() == "-1")
                {
                    return null;
                }

                string[] prepareResult = ((string)a[0]).Split('\n');

                List<string> strList = new List<string>(prepareResult);
                strList.ForEach(s => { if (s.Length == 0) strList.Remove(s); });
                strTransData = strList.ConvertAll(s => prepareMatlabStr(s).Split('\t'));      
            }
            catch (Exception)
            {
                
                return null;
            }
           
            double[][] transformedData;
            try
            {
              transformedData = strTransData.ConvertAll(s => Array.ConvertAll(s, Double.Parse)).ToArray();
            }
            catch (Exception)
            {
                throw new Exception("DiffusionMaps convert transformed data error!!!");                
            }
            matlab.Execute("clear;");
            matlab.Quit();
            
             return transformedData;
        }

        private static string prepareMatlabStr(string s)
        {
            Regex firstSpaceCleaner = new Regex("^\\s+");
            Regex separator = new Regex("\\s+");
            Regex matlabInf = new Regex("Inf");
            string result = s;
            result = matlabInf.Replace(result, "1e+300");
            result = firstSpaceCleaner.Replace(result, "");
            result= separator.Replace(result, "\t");
            return result;
        }

        public static void reCreateMatlabCom()
        {
            matlab = null;
            matlab = new MLApp.MLApp {Visible = 0};
        }

        public static double[,] calculateGlobalDist(double[][] data)
        {
            double[,] distMatrix = new double[data.Length, data.Length];
            int dataDim = data.First().Count();
            for (int i = 0; i < data.Length; i++)
            {
                for (int j = 0; j < data.Length; j++) // Lenght/2
                {
                    double normSquare = 0;
                    for (int dim = 0; dim < dataDim; dim++)
                    {
                        double tmp = data[i][dim] - data[j][dim];
                        normSquare += tmp*tmp;
                    }
                    distMatrix[i, j] = -normSquare; // [i,j]=[j,i]
                }
            }
            double[,] expMat = distMatrix.Exp();
           
            SingularValueDecomposition theDecomposition=new SingularValueDecomposition(expMat,true,false,false,true);
            
            return distMatrix;
        }

    }
}

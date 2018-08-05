using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using Accord.Math;

namespace DataSetsSparsity
{
    public class GeoWave
    {
        // Yair's variables
        public double[,] Y_dPLSConversionMatrix;   // Should hold the conversion matrix for the node's PLS
        public static int Y_nPLSDim = 1;            // The amount of PLS components to take into consideration
        public Accord.Statistics.Analysis.PartialLeastSquaresAnalysis Y_PLSTransformObject;     // The PLS object to use inorder to recover the conversion for test instances
        public bool Y_bIsPLSSplit = false;     // True is the split made was over the PLS variables
        public int Y_nDimPPLSSplitIndex = -1;            // The dimention of the PLS split
        public double Y_dPLSSplitValue = 0;        // The value which seperates the childs (the split actual value)    
        // end of Yair's variables

        public int parentID, child0, child1, level;
        public double proximity;
        public double norm;
        public double[] MeanValue; //vector with means at each dimention
        public int[][] boubdingBox;
        public List<int> pointsIdArray = new List<int>();//points in regeion (row index of static input data)
        public int ID;
        public recordConfig rc;
        public int dimIndex;//of partition
        public int Maingridindex;//of partition
        public double MaingridValue;//of partition
        // GiniBased. each vector element is gini estimation on this dimention
        public struct GStuff
        {
            public double[] GiniVector;
            public double GiniNorm;
            public double GiniAvg;
            public Dictionary<double, double>[] dicLabelCount;
        }
        public GeoWave.GStuff MgStuff;

        public GeoWave(int dataDim, int labelDim, recordConfig rc)
        {
            this.rc = rc;
            Init(dataDim, labelDim);
        }

        private void Init(int dataDim, int labelDim)
        {
            parentID = -1;
            child0 = -1;
            child1 = -1;
            level = -1;
            norm = -1;
            //approx_solution = new double[dataDim, labelDim];
            boubdingBox = new int[2][];
            boubdingBox[0] = new int[dataDim];  // YAIR added the pls variables for each GW's bounding box
            boubdingBox[1] = new int[dataDim];  // YAIR added the pls variables for each GW's bounding box

            MeanValue = new double[labelDim];
            MgStuff = new GStuff { GiniVector = new double[labelDim] };
            MgStuff.dicLabelCount = new Dictionary<double, double>[labelDim];
            ID = -1;
            dimIndex = -1;
            Maingridindex = -1;
            MaingridValue = -1;
            
            //Y_nPLSDim = 2;      // How many of the PLS dimentions to take

            Y_dPLSConversionMatrix = new double[dataDim,dataDim];
            //Y_dPLSConversionMatrix = new double[dataDim][];   // init the matrix
            //for (int i = 0; i < dataDim; i++)
            //    Y_dPLSConversionMatrix[i] = new double[Y_nPLSDim];
            
        }

        public GeoWave(int[][] BOX, int labelDim, recordConfig rc)
        {
            this.rc = rc;
            Init(rc.dim, labelDim);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < rc.dim; j++)    // YAIR only copy all the 'regular' variables, leave the PLS as is
                    boubdingBox[i][j] = BOX[i][j];
            //Array.Copy(BOX, boubdingBox, BOX[0].Count() * 2);
            //boubdingBox = BOX.Select(s => s.ToArray()).ToArray();
            //BOX.CopyTo(boubdingBox, 0);
        }



        public double[] calc_MeanValue(double[][] Labels_dt, List<int> indexArr)
        {
            double[] tmpMeanValue = new double[MeanValue.Count()];

            //GO OVER ALL POINTS IN REGION
            foreach (int index in indexArr)
            {
                for (int j = 0; j < Labels_dt[0].Count(); j++)
                    tmpMeanValue[j] += Labels_dt[index][j];
            }

            if (indexArr.Count * Labels_dt[0].Count() > 0)
                tmpMeanValue = tmpMeanValue.Divide(indexArr.Count);

            return tmpMeanValue;
        }
        //calculate sum of distances from mean vector
        public double calc_MeanValueReturnError(double[][] labelsDt, List<int> indexArr)
        {
            //NULLIFY
            double[] tmpMeanValue = calc_MeanValue(labelsDt, indexArr);
            double Error = 0;

            switch (rc.partitionErrType)
            {
                case 2: //L2
                    foreach (int index in indexArr)
                    {
                        for (int j = 0; j < labelsDt[0].Count(); j++)
                            Error += (labelsDt[index][j] - tmpMeanValue[j]) * (labelsDt[index][j] - tmpMeanValue[j]);//insert rc of norm type
                    }
                    return Error;
                case 1: //L1
                    foreach (int index in indexArr)
                    {
                        for (int j = 0; j < labelsDt[0].Count(); j++)
                            Error += Math.Abs(labelsDt[index][j] - tmpMeanValue[j]);
                    }
                    return Error;
                case 0:
                    foreach (int index in indexArr)
                    {
                        double sign = 0;
                        for (int j = 0; j < labelsDt[0].Count(); j++)
                            sign += Math.Abs(labelsDt[index][j] - tmpMeanValue[j]); ;
                        if (sign != 0)
                            Error += 1;
                    }
                    return Error;
                default:
                    {
                        double p = rc.partitionErrType;
                        //GO OVER ALL POINTS IN REGION
                        foreach (int index in indexArr)
                        {
                            for (int j = 0; j < labelsDt[0].Count(); j++)
                                Error += Math.Pow(labelsDt[index][j] - tmpMeanValue[j], p);
                        }
                        //return Error.Sum();
                        //return Math.Pow(Error, 1/p);
                        return Error;//same as l2 don't do sqrt  
                    }
            }
        }

        public double[] calc_MeanValueReturnError(double[][] labelsDt, List<int> indexArr, ref double[] calcedMeanValue)
        {
            //NULLIFY
            calcedMeanValue = calc_MeanValue(labelsDt, indexArr);
            double[] error = new double[labelsDt[0].Count()];

            switch (rc.partitionErrType)
            {
                case 2://L2
                    foreach (int index in indexArr)
                    {
                        for (int j = 0; j < labelsDt[0].Count(); j++)
                            error[j] += (labelsDt[index][j] - calcedMeanValue[j]) * (labelsDt[index][j] - calcedMeanValue[j]);//insert rc of norm type
                    }
                    return error;
                case 1://L1
                    foreach (int index in indexArr)
                    {
                        for (int j = 0; j < labelsDt[0].Count(); j++)
                            error[j] += Math.Abs(labelsDt[index][j] - calcedMeanValue[j]);
                    }
                    return error;
                case 0:
                    MessageBox.Show("shoud not get here rc.partitionErrType == 0");
                    for (int i = 0; i < indexArr.Count; i++)
                    {
                        double sign = 0;
                    }
                    return error;
                default:
                    {
                        double p = rc.partitionErrType;
                        //GO OVER ALL POINTS IN REGION
                        for (int i = 0; i < indexArr.Count; i++)
                        {
                            for (int j = 0; j < labelsDt[0].Count(); j++)
                                error[j] += Math.Pow(labelsDt[indexArr[i]][j] - calcedMeanValue[j], p);
                        }
                        //return Error.Sum();
                        //return Math.Pow(Error, 1/p);
                        return error;//same as l2 don't do sqrt  
                    }
            }
        }
        public void computeNormOfConsts(GeoWave parent)
        {
            norm = 0;
            //GO OVER ALL POINTS IN THE REGION
            for (int j = 0; j < MeanValue.Count(); j++)
            {
                norm += ((MeanValue[j] - parent.MeanValue[j]) * (MeanValue[j] - parent.MeanValue[j]));
            }
            norm = norm * pointsIdArray.Count();
        }

        public void computeNormOfConsts()
        {
            norm = 0;
            for (int j = 0; j < MeanValue.Count(); j++)
                norm += (MeanValue[j] * MeanValue[j]);
            norm = norm * pointsIdArray.Count();
        }
    }
}

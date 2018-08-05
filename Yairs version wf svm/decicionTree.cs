using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;
using System.IO;

namespace DataSetsSparsity
{
    class decicionTree
    {
        private recordConfig rc;
        private double[][] training_dt;
        private long[][] training_GridIndex_dt;
        private double[][] training_label;
        private bool[] Dime2Take;

        public decicionTree(recordConfig rc, DB db)
        {
            this.training_dt = db.PCAtraining_dt;
            this.training_label = db.training_label;
            this.training_GridIndex_dt = db.PCAtraining_GridIndex_dt;
            this.rc = rc;
        }

        public decicionTree(recordConfig rc, DB db, bool[] Dime2Take)
        {
            this.training_dt = db.PCAtraining_dt;
            this.training_label = db.training_label;
            this.training_GridIndex_dt = db.PCAtraining_GridIndex_dt;
            this.rc = rc;
            this.Dime2Take = Dime2Take;
        }
        public decicionTree(recordConfig rc, double[][] training_dt, double[][] training_label)
        {
            this.training_dt = training_dt;
            this.training_label = training_label;
            this.rc = rc;
        }
        public decicionTree(recordConfig rc, double[][] training_dt, double[][] training_label, long[][] training_GridIndex_dt, bool[] Dime2Take)
        {
            this.training_dt = training_dt;
            this.training_label = training_label;
            this.rc = rc;
            this.training_GridIndex_dt = training_GridIndex_dt;
            this.Dime2Take = Dime2Take;
        }

        public List<GeoWave> getdecicionTree(List<int> trainingArr, int[][] boundingBox, int seed = -1)
        {
            //CREATE DECISION_GEOWAVEARR
            List<GeoWave> decision_GeoWaveArr = new List<GeoWave>();

            //SET ROOT WAVELETE
            GeoWave gwRoot = new GeoWave(rc.dim, training_label[0].Count(), rc);

            //SET REGION POINTS IDS
            gwRoot.pointsIdArray = trainingArr;
            boundingBox.CopyTo(gwRoot.boubdingBox, 0);

            decision_GeoWaveArr.Add(gwRoot);
            DecomposeWaveletsByConsts(decision_GeoWaveArr, seed);

            //consider next twofunctions ?????

            //SET ID
            for (int i = 0; i < decision_GeoWaveArr.Count; i++)
                decision_GeoWaveArr[i].ID = i;

            //get sorted list
            decision_GeoWaveArr = decision_GeoWaveArr.OrderByDescending(o => o.norm).ToList();

            return decision_GeoWaveArr;
        }

        public void DecomposeWaveletsByConsts(List<GeoWave> GeoWaveArr, int seed = -1)//SHOULD GET LIST WITH ROOT GEOWAVE
        {
            GeoWaveArr[0].MeanValue = GeoWaveArr[0].calc_MeanValue(training_label, GeoWaveArr[0].pointsIdArray);
            GeoWaveArr[0].computeNormOfConsts();
            GeoWaveArr[0].level = 0;

            if (seed == -1)
                recursiveBSP_WaveletsByConsts(GeoWaveArr, 0);
            else recursiveBSP_WaveletsByConsts(GeoWaveArr, 0, seed);//0 is the root index
            //NonrecursiveBSP_WaveletsByConsts(GeoWaveArr, 0);//0 is the root index
        }

        private void recursiveBSP_WaveletsByConsts(List<GeoWave> GeoWaveArr, int GeoWaveID, int seed=0)
        {
            //CALC APPROX_SOLUTION FOR GEO WAVE
            double Error = GeoWaveArr[GeoWaveID].calc_MeanValueReturnError(training_label, GeoWaveArr[GeoWaveID].pointsIdArray);
            if (Error < rc.approxThresh || GeoWaveArr[GeoWaveID].pointsIdArray.Count() <= rc.minWaveSize || rc.boundDepthTree <=  GeoWaveArr[GeoWaveID].level)
                return;
            double tmpError = Error;

            int dimIndex = -1;
            int Maingridindex = -1;

            bool IsPartitionOK = false;
            if (rc.split_type == 0)
                IsPartitionOK = getBestPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, ref tmpError, Dime2Take);
            else if (rc.split_type == 1)//rand split
                IsPartitionOK = getRandPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, Error, seed);
            else if (rc.split_type == 2)//rand features in each node
            {
                var ran1 = new Random(seed);
                var ran2 = new Random(GeoWaveID);
                int one = ran1.Next(0, int.MaxValue / 10);
                int two = ran2.Next(0, int.MaxValue / 10);
                bool[] Dim2TakeNode = getDim2Take(rc, one + two);
                IsPartitionOK = getBestPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, ref tmpError, Dim2TakeNode);
            }
            else if (rc.split_type == 3)//Gini split
            {
                IsPartitionOK = GetGiniPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, Error, Dime2Take);
            }
            else if (rc.split_type == 4)//Gini split + rand node
            {
                var ran1 = new Random(seed);
                var ran2 = new Random(GeoWaveID);
                int one = ran1.Next(0, int.MaxValue / 10);
                int two = ran2.Next(0, int.MaxValue / 10);
                bool[] Dim2TakeNode = getDim2Take(rc, one + two);
                IsPartitionOK = GetGiniPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, Error, Dim2TakeNode);
            }
            else if (rc.split_type == 11) //Here will add the PLS method. YTODO: add PLS method
            {
                // Here we do the same as split value 2, but with new variables added to the data.
                var ran1 = new Random(seed);
                var ran2 = new Random(GeoWaveID);
                int one = ran1.Next(0, int.MaxValue / 10);
                int two = ran2.Next(0, int.MaxValue / 10);
                bool[] Dim2TakeNode = getDim2Take(rc, one + two);   // Gets <rc.NDimsinRF> random dimentions from the original dimentions of the data.
              /*  bool[] Y_dim2TakeNode = new bool[Dim2TakeNode.Length + GeoWaveArr[GeoWaveID].Y_nPLSDim];
                for (int i = 0; i < Dim2TakeNode.Length; i++)   // Copying the random variables created to a new vector
                    Y_dim2TakeNode[i] = Dim2TakeNode[i];
                for (int i = 0; i < GeoWaveArr[GeoWaveID].Y_nPLSDim; i++)   // Adding the new variables of the PLS
                    Y_dim2TakeNode[i + Dim2TakeNode.Length] = true;  */

                double Y_dTmpError = Error;
                int Y_dimIndex = -1, Y_Maingridindex = -1;
                bool Y_IsPartitionOK = false;
                double[][] Y_PLSdata = new double[GeoWaveArr[GeoWaveID].pointsIdArray.Count][]; //YTODO: need to create the PLS data here...

                ///////// CREATING THE PLS DATA
                double[,] Y_PLSTrain = new double[GeoWaveArr[GeoWaveID].pointsIdArray.Count, rc.dim];
                double[,] Y_PLSLabel = new double[GeoWaveArr[GeoWaveID].pointsIdArray.Count, 1];
                for (int i = 0; i < GeoWaveArr[GeoWaveID].pointsIdArray.Count; i++)
                {
                    for (int j = 0; j < rc.dim; j++)
                    {
                        Y_PLSTrain[i, j] = training_dt[i][j];
                        
                    }
                    Y_PLSLabel[i, 0] = training_label[i][0];
                }

                Accord.Statistics.Analysis.PartialLeastSquaresAnalysis pls =
                    new Accord.Statistics.Analysis.PartialLeastSquaresAnalysis(Y_PLSTrain, Y_PLSLabel,
                        Accord.Statistics.Analysis.AnalysisMethod.Center, Accord.Statistics.Analysis.PartialLeastSquaresAlgorithm.SIMPLS);
                pls.Compute();
                double[,] tempPLSDATA = pls.Transform(Y_PLSTrain, GeoWave.Y_nPLSDim);
                for (int i = 0; i < GeoWaveArr[GeoWaveID].pointsIdArray.Count; i++)
                {
                    Y_PLSdata[i] = new double[GeoWave.Y_nPLSDim];
                    for (int j = 0; j < GeoWave.Y_nPLSDim; j++)
                    {
                        Y_PLSdata[i][j] = tempPLSDATA[i, j];
                    }
                }

                /////////

                IsPartitionOK = Y_getBestPartitionResult(ref dimIndex, ref Maingridindex, ref Y_dTmpError, GeoWaveArr, GeoWaveID, Dim2TakeNode, training_dt, rc.dim); // Running the same function ran in the random variables section (best partition over all random variables only)
                double Y_dTmpError2 = Y_dTmpError;
                bool[] Y_Dim2Take = new bool[GeoWave.Y_nPLSDim];
                for (int i = 0; i < GeoWave.Y_nPLSDim; i++)
                {
                    Y_Dim2Take[i] = true;
                }

                Y_IsPartitionOK = Y_getBestPartitionResult(ref Y_dimIndex, ref Y_Maingridindex, ref Y_dTmpError2, GeoWaveArr, GeoWaveID, Y_Dim2Take, Y_PLSdata, GeoWave.Y_nPLSDim); // Running the same function over the new variables
                IsPartitionOK = IsPartitionOK | Y_IsPartitionOK; // YTODO: see what exactly is the meaning of this bool var and if this is correct here
                if (Y_dTmpError2 < Y_dTmpError) // The new variable is better than the old variables.
                {
                    dimIndex = Y_dimIndex/*need to add the number of 'regular' variables so it fits in the whole array*/ + rc.dim;
                    Maingridindex = Y_Maingridindex;
                    //GeoWaveArr[GeoWaveID].Y_PLSTransformObject = pls;
                    //GeoWave
                }
            }
            else if (rc.split_type == 12) //Here will add the PLS method. YTODO: add PLS method
            {
                // Here we do the same as split value 2, but with new variables added to the data.
                var ran1 = new Random(seed);
                var ran2 = new Random(GeoWaveID);
                int one = ran1.Next(0, int.MaxValue / 10);
                int two = ran2.Next(0, int.MaxValue / 10);
                bool[] Dim2TakeNode = getDim2Take(rc, one + two);   // Gets <rc.NDimsinRF> random dimentions from the original dimentions of the data.
                bool[] Y_Dim2TakeNode = new bool[rc.NDimsinRF + GeoWave.Y_nPLSDim];
                for (int i = 0; i < rc.NDimsinRF; i++)
                    Y_Dim2TakeNode[i] = Dim2TakeNode[i];
                for (int i = rc.NDimsinRF; i < rc.NDimsinRF + GeoWave.Y_nPLSDim; i++)
                    Y_Dim2TakeNode[i] = true;

                double Y_dTmpError = Error;
                int Y_dimIndex = -1, Y_Maingridindex = -1;
                bool Y_IsPartitionOK = false;   


                ///////// CREATING THE PLS DATA
                //YTODO: need to create the PLS data here...
                double[][] Y_PLSdata = new double[GeoWaveArr[GeoWaveID].pointsIdArray.Count][];     // Take only the points that are within the wavelet and make the PLS on them
                double[,] Y_PLSTrain = new double[GeoWaveArr[GeoWaveID].pointsIdArray.Count, rc.NDimsinRF];
                double[,] Y_PLSLabel = new double[GeoWaveArr[GeoWaveID].pointsIdArray.Count, 1];
                for (int i = 0; i < GeoWaveArr[GeoWaveID].pointsIdArray.Count; i++)
                {
                    for (int j = 0; j < rc.NDimsinRF; j++)
                    {
                        Y_PLSTrain[i, j] = training_dt[i][j];

                    }
                    Y_PLSLabel[i, 0] = training_label[i][0];
                }

                Accord.Statistics.Analysis.PartialLeastSquaresAnalysis pls =
                    new Accord.Statistics.Analysis.PartialLeastSquaresAnalysis(Y_PLSTrain, Y_PLSLabel,
                        Accord.Statistics.Analysis.AnalysisMethod.Center, Accord.Statistics.Analysis.PartialLeastSquaresAlgorithm.SIMPLS);
                pls.Compute();
                double[,] tempPLSDATA = pls.Transform(Y_PLSTrain, GeoWave.Y_nPLSDim);
                for (int i = 0; i < GeoWaveArr[GeoWaveID].pointsIdArray.Count; i++)
                {
                //    Y_PLSdata[i] = new double[GeoWave.Y_nPLSDim];
                    //for (int j = 0; j < GeoWave.Y_nPLSDim; j++)
                    for (int j = 0; j < GeoWave.Y_nPLSDim; j++)   // only change the variables of the pls 
                    {
                  //      Y_PLSdata[i][j] = tempPLSDATA[i, j];
                        training_dt[i][rc.NDimsinRF + j] = tempPLSDATA[i, j];
                    }
                }

                ////for DBG !!!
                //Form1.printtable(training_dt, @"C:\Users\212441441\Dropbox\Yair\DBG\PLS_DATA.txt");
                //Form1.printtable(training_label, @"C:\Users\212441441\Dropbox\Yair\DBG\PLS_LABEL.txt");

                ///////// end of - CREATING PLS DATA

 //               if (rc.split_type == 12)    // In that case we need to create bounding box and grid for the new PLS data as well.
 //               {
                    double[][] Y_PLSVars = new double[training_dt.Count()][];
                    for (int j = 0; j < training_dt.Count(); j++)
                    {
                        Y_PLSVars[j] = new double[GeoWave.Y_nPLSDim];
                        for (int i = 0; i < GeoWave.Y_nPLSDim; i++)
                            Y_PLSVars[j][i] = training_dt[j][rc.NDimsinRF + i];   // only the PLS added variables are taken into an array to create their bounding box and grid
                    }

                    DB tmp = new DB();
                    double[][] Y_PLSBoundingBox = tmp.getboundingBox(Y_PLSVars);
                    long[][] Y_PLSTrainingGridIndex = new long[training_dt.Count()][];
                    for (int i = 0; i < training_dt.Count(); i++)
                    {
                        Y_PLSTrainingGridIndex[i] = new long[GeoWave.Y_nPLSDim];
                    }
                    List<List<double>> Y_PLSMainGrid = tmp.getMainGrid(Y_PLSVars, Y_PLSBoundingBox, ref Y_PLSTrainingGridIndex);

                    //bounding intervals
                    int[][] BB = new int[2][];
                    BB[0] = new int[Y_PLSBoundingBox[0].Count()];
                    BB[1] = new int[Y_PLSBoundingBox[0].Count()];
                    for (int i = 0; i < Y_PLSBoundingBox[0].Count(); i++)
                    {
                        BB[1][i] = Y_PLSMainGrid[i].Count() - 1;//set last index in each dim
                        for (int j = 0; j < training_dt.Count(); j++)
		               	{
                             Form1.MainGrid[rc.NDimsinRF + i][j] = Y_PLSMainGrid[i][j];
		               	}
                              
                    }

                    Form1.MainGrid.Count();
                    for (int j = 0; j < 2; j++)
                    {
                        for (int i = 0; i < GeoWave.Y_nPLSDim; i++)
                        {
                            GeoWaveArr[GeoWaveID].boubdingBox[j][rc.NDimsinRF + i] = BB[j][i];
                            // main grid... ??
                        }
                    }
//                }

                IsPartitionOK = Y_getBestPartitionResult(ref dimIndex, ref Maingridindex, ref Y_dTmpError,
                    GeoWaveArr, GeoWaveID, Y_Dim2TakeNode, training_dt, rc.dim + GeoWave.Y_nPLSDim); // Running the same function ran in the random variables section (best partition over all random variables only)
                //double Y_dTmpError2 = Y_dTmpError;
                //bool[] Y_Dim2Take = new bool[GeoWave.Y_nPLSDim];
                //for (int i = 0; i < GeoWave.Y_nPLSDim; i++)
                //{
                //    Y_Dim2Take[i] = true;
                //}

                //Y_IsPartitionOK = Y_getBestPartitionResult(ref Y_dimIndex, ref Y_Maingridindex, ref Y_dTmpError2,
                //    GeoWaveArr, GeoWaveID, Y_Dim2Take, Y_PLSdata, GeoWave.Y_nPLSDim); // Running the same function over the new variables
                //IsPartitionOK = IsPartitionOK | Y_IsPartitionOK; // YTODO: see what exactly is the meaning of this bool var and if this is correct here
                //if (Y_dTmpError2 < Y_dTmpError) // The new variable is better than the old variables.
                //{
                //    dimIndex = Y_dimIndex/*need to add the number of 'regular' variables so it fits in the whole array*/ + rc.dim;
                //    Maingridindex = Y_Maingridindex;
                //    GeoWaveArr[GeoWaveID].Y_PLSTransformObject = pls;
                //    //GeoWave
                //}
            }
            else if (rc.split_type == 13) //Here will add the PLS method. (running the regualr variables with Oren's functions and the PLS variables with a new function)
            {
                // Here we do the same as split value 2, but with new variables added to the data.
                var ran1 = new Random(seed);
                var ran2 = new Random(GeoWaveID);
                int one = ran1.Next(0, int.MaxValue / 10);
                int two = ran2.Next(0, int.MaxValue / 10);
                bool[] Dim2TakeNode = getDim2Take(rc, one + two);   // Gets <rc.NDimsinRF> random dimentions from the original dimentions of the data.
                double orgVarError = Error, PLSSplitValue = -1;
                int PLSDimIndex = -1;
                bool IsPLSPartitionOK = false;
                bool bUseOnlyPLS = false;

                if (!bUseOnlyPLS)
                    IsPartitionOK = getBestPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, ref orgVarError, Dim2TakeNode);


                ///// CREATING PLS DATA BEFORE ATTEMPTING SPLIT
                //YTODO: need to create the PLS data here...
                //double[][] Y_PLSdata = new double[GeoWaveArr[GeoWaveID].pointsIdArray.Count][];     // Take only the points that are within the wavelet and make the PLS on them
                double[][] Y_PLSdata = new double[training_dt.Count()][];   // size is of the same as the data set (for further calculation), however, only the points that are in the geowavelet will be filled with informaton
                double[][] Y_Label = new double[training_label.Count()][];
                double[][] Y_data = new double[training_dt.Count()][];
                double[,] Y_PLSTrain = new double[GeoWaveArr[GeoWaveID].pointsIdArray.Count, rc.dim];
                double[,] Y_PLSLabel = new double[GeoWaveArr[GeoWaveID].pointsIdArray.Count, 1];
                for (int i = 0; i < GeoWaveArr[GeoWaveID].pointsIdArray.Count; i++)
                {
                    int index = GeoWaveArr[GeoWaveID].pointsIdArray[i];
                    Y_data[index] = new double[rc.dim];
                    for (int j = 0; j < rc.dim; j++)
                    {
                        Y_PLSTrain[i, j] = training_dt[index][j];
                        Y_data[index][j] = training_dt[index][j];
                    }
                    Y_PLSLabel[i, 0] = training_label[index][0];
                    Y_Label[index] = new double[1];
                    Y_Label[index][0] = Y_PLSLabel[i, 0];
                }

                Accord.Statistics.Analysis.PartialLeastSquaresAnalysis pls =
                    new Accord.Statistics.Analysis.PartialLeastSquaresAnalysis(Y_PLSTrain, Y_PLSLabel,
                        Accord.Statistics.Analysis.AnalysisMethod.Center, Accord.Statistics.Analysis.PartialLeastSquaresAlgorithm.SIMPLS);
                pls.Compute();
                ////
                //double[,] Id = Matrix.Identity(11);
                //double[,] Id2 = Matrix.Identity(11);
                //Id = pls.Transform(Id, 11);
                //Id2 = Matrix.Multiply(Id2, pls.Weights);
                ////
                int plsDim = GeoWave.Y_nPLSDim; ;
                if (GeoWave.Y_nPLSDim >= GeoWaveArr[GeoWaveID].pointsIdArray.Count())
	            {
                    plsDim = GeoWaveArr[GeoWaveID].pointsIdArray.Count() - 1;
	            }
                
                double[,] tempPLSDATA = pls.Transform(Y_PLSTrain, plsDim);

                for (int i = 0; i < GeoWaveArr[GeoWaveID].pointsIdArray.Count; i++)
                {
                    Y_PLSdata[GeoWaveArr[GeoWaveID].pointsIdArray[i]] = new double[/*GeoWave.Y_nPLSDim*/plsDim];

                    for (int j = 0; j < /*GeoWave.Y_nPLSDim*/plsDim; j++)   // only change the variables of the PLS and only over the geowavelet points
                    {
                        Y_PLSdata[GeoWaveArr[GeoWaveID].pointsIdArray[i]][j] = tempPLSDATA[i, j];
                    }
                }

                ////for DBG !!!
                //if (GeoWaveArr[GeoWaveID].pointsIdArray.Count() > 1)
                //{
                //    //// Laptop path
                //    Form1.printtable(Y_Label, @"D:\Dropbox\Yair\data\testingPLS\PLS_LABEL.txt");
                //    //Form1.printtable(Y_data, @"D:\Dropbox\Yair\data\testingPLS\PLS_DATA.txt");
                //    Form1.printtable(Y_PLSdata, @"D:\Dropbox\Yair\data\testingPLS\PLS_DATA.txt");

                //    //Form1.printtable(training_dt, @"D:\Dropbox\Yair\data\testingPLS\OriginalData.txt");
                //    //Form1.printtable(training_label, @"D:\Dropbox\Yair\data\testingPLS\OriginalLabel.txt");
                //    //Form1.printtable(training_label, @"D:\Dropbox\Yair\data\testingPLS\PLS_LABEL.txt");


                //    //// desktop path
                //    //Form1.printtable(Y_Label, @"D:\YairMS\Dropbox\Yair\data\testingPLS\PLS_LABEL.txt");
                //    //Form1.printtable(Y_data, @"D:\YairMS\Dropbox\Yair\data\testingPLS\PLS_DATA.txt");

                //    Error = Error;
                //}

                /////
                double PLSError = Error;
                IsPLSPartitionOK = Y_getBestPartitionPLSVars(ref PLSDimIndex, ref PLSSplitValue, ref PLSError, GeoWaveArr, GeoWaveID, Y_PLSdata, plsDim/*GeoWave.Y_nPLSDim*/);
                

                if ((PLSError < orgVarError) && (IsPLSPartitionOK))     // Means that the error of the PLS split is better - a PLS split is chosen 
                {
                    GeoWaveArr[GeoWaveID].Y_bIsPLSSplit = true;         // Remember that the split was done over the PLS for the testing
                    GeoWaveArr[GeoWaveID].Y_dPLSSplitValue = PLSSplitValue;
                    GeoWaveArr[GeoWaveID].Y_nDimPPLSSplitIndex = PLSDimIndex;
                    GeoWaveArr[GeoWaveID].Y_PLSTransformObject = pls;
                    //GeoWaveArr[GeoWaveID].Y_dPLSConversionMatrix = Matrix.Inverse(pls.Transform(Matrix.Identity(rc.dim)));

                    GeoWave childA = new GeoWave(GeoWaveArr[GeoWaveID].boubdingBox, training_label[0].Count(), GeoWaveArr[GeoWaveID].rc);
                    GeoWave childB = new GeoWave(GeoWaveArr[GeoWaveID].boubdingBox, training_label[0].Count(), GeoWaveArr[GeoWaveID].rc);

                    //childA.dimIndex = PLSDimIndex;
                    //childA.Maingridindex = Maingridindex;
                    //childB.dimIndex = PLSDimIndex;
                    //childB.Maingridindex = Maingridindex;
                    double[] splitVector = new double[Y_PLSdata.Count()];
                    for (int i = 0; i < GeoWaveArr[GeoWaveID].pointsIdArray.Count(); i++)
                    {
                        int index = GeoWaveArr[GeoWaveID].pointsIdArray[i];
                        splitVector[index] = Y_PLSdata[index][PLSDimIndex];
                    }

                    Y_setChildrensPointsAndMeanValuePLSSplit(ref childA, ref childB, splitVector, PLSSplitValue, GeoWaveArr[GeoWaveID].pointsIdArray);

                    childA.parentID = childB.parentID = GeoWaveID;
                    childA.child0 = childB.child0 = -1;
                    childA.child1 = childB.child1 = -1;
                    childA.level = childB.level = GeoWaveArr[GeoWaveID].level + 1;
                    //childA.dimIndex = dimIndex;
                    //childB.dimIndex = dimIndex;

                    childA.computeNormOfConsts(GeoWaveArr[GeoWaveID]);    // need to make sure that by this line the points and mean values are set
                    childB.computeNormOfConsts(GeoWaveArr[GeoWaveID]);
                    GeoWaveArr.Add(childA);
                    GeoWaveArr.Add(childB);
                    GeoWaveArr[GeoWaveID].child0 = GeoWaveArr.Count - 2;
                    GeoWaveArr[GeoWaveID].child1 = GeoWaveArr.Count - 1;

                    recursiveBSP_WaveletsByConsts(GeoWaveArr, GeoWaveArr[GeoWaveID].child0, seed);
                    recursiveBSP_WaveletsByConsts(GeoWaveArr, GeoWaveArr[GeoWaveID].child1, seed);

                    return;
                }
 
            }

            

            ////bool IsPartitionOK = getBestPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, Error);
            //bool IsPartitionOK = getRandPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, Error, seed);

            if (!IsPartitionOK)
                return;


            

            GeoWave child0 = new GeoWave(GeoWaveArr[GeoWaveID].boubdingBox, training_label[0].Count(), GeoWaveArr[GeoWaveID].rc);
            GeoWave child1 = new GeoWave(GeoWaveArr[GeoWaveID].boubdingBox, training_label[0].Count(), GeoWaveArr[GeoWaveID].rc);



            //set partition
            child0.boubdingBox[1][dimIndex] = Maingridindex;
            child1.boubdingBox[0][dimIndex] = Maingridindex;

            //DOCUMENT ON CHILDREN
            child0.dimIndex = dimIndex;
            child0.Maingridindex = Maingridindex;
            child1.dimIndex = dimIndex;
            child1.Maingridindex = Maingridindex;

            child0.MaingridValue = Form1.MainGrid[dimIndex][Maingridindex]; // YTODO: main grid is not defined for the new PLS vars. will crash here...!
            child1.MaingridValue = Form1.MainGrid[dimIndex][Maingridindex]; // YAIR seems like it can be left with garbage

            //calc norm
            //calc mean value

            if (Form1.IsBoxSingular(child0.boubdingBox, rc.dim) || Form1.IsBoxSingular(child1.boubdingBox, rc.dim))
                return;

            //SHOULD I VERIFY THAT THE CHILD IS NOT ITS PARENT ? (IN CASES WHERE CAN'T MODEFY THE PARTITION)

            setChildrensPointsAndMeanValue(ref child0, ref child1, dimIndex, GeoWaveArr[GeoWaveID].pointsIdArray);
            //SET TWO CHILDS
            child0.parentID = child1.parentID = GeoWaveID;
            child0.child0 = child1.child0 = -1;
            child0.child1 = child1.child1 = -1;
            child0.level = child1.level = GeoWaveArr[GeoWaveID].level + 1;

            child0.computeNormOfConsts(GeoWaveArr[GeoWaveID]);  
            child1.computeNormOfConsts(GeoWaveArr[GeoWaveID]);
            GeoWaveArr.Add(child0);
            GeoWaveArr.Add(child1);
            GeoWaveArr[GeoWaveID].child0 = GeoWaveArr.Count - 2;
            GeoWaveArr[GeoWaveID].child1 = GeoWaveArr.Count - 1;


            //// calculate gini index for childrens
            //if (rc.split_type == 3 || rc.split_type == 4)
            //{
            //    //could set information gain here
            //}

            //RECURSION STEP !!!
            recursiveBSP_WaveletsByConsts(GeoWaveArr, GeoWaveArr[GeoWaveID].child0, seed);
            recursiveBSP_WaveletsByConsts(GeoWaveArr, GeoWaveArr[GeoWaveID].child1, seed);
        }

        private void set_BSP_WaveletsByConsts(List<GeoWave> GeoWaveArr, int GeoWaveID, int seed=0)
        {
            //CALC APPROX_SOLUTION FOR GEO WAVE
            double Error = GeoWaveArr[GeoWaveID].calc_MeanValueReturnError(training_label, GeoWaveArr[GeoWaveID].pointsIdArray);
            if (Error < rc.approxThresh || GeoWaveArr[GeoWaveID].pointsIdArray.Count() <= rc.minWaveSize || rc.boundDepthTree <= GeoWaveArr[GeoWaveID].level)
                return;
            double tmpError = Error;

            int dimIndex = -1;
            int Maingridindex = -1;

            bool IsPartitionOK = false;
            if (rc.split_type == 0)
                IsPartitionOK = getBestPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, ref tmpError, Dime2Take);//consider dropping Dime2Take
            else if (rc.split_type == 1)//rand split
                IsPartitionOK = getRandPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, Error);
            else if (rc.split_type == 2)//rand features in each node
            {
                var ran1 = new Random(seed);
                var ran2 = new Random(GeoWaveID);
                int one = ran1.Next(0, int.MaxValue / 10);
                int two = ran2.Next(0, int.MaxValue / 10);
                bool[] Dim2TakeNode = getDim2Take(rc, one + two);
                IsPartitionOK = getBestPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, ref tmpError, Dim2TakeNode);
            }
            else if (rc.split_type == 3)//Gini split
            {
                IsPartitionOK = GetGiniPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, Error, Dime2Take);
            }
            else if (rc.split_type == 4)//Gini split + rand node
            {
                var ran1 = new Random(seed);
                var ran2 = new Random(GeoWaveID);
                int one = ran1.Next(0, int.MaxValue / 10);
                int two = ran2.Next(0, int.MaxValue / 10);
                bool[] Dim2TakeNode = getDim2Take(rc, one + two);
                IsPartitionOK = GetGiniPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, Error, Dim2TakeNode);
            }

            //bool IsPartitionOK = getBestPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, Error);
            //bool IsPartitionOK = getRandPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, Error, seed);
            //bool IsPartitionOK = getBestPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, Error, Dime2Take);

            if (!IsPartitionOK)
                return;

            GeoWave child0 = new GeoWave(GeoWaveArr[GeoWaveID].boubdingBox, training_label[0].Count(), GeoWaveArr[GeoWaveID].rc);
            GeoWave child1 = new GeoWave(GeoWaveArr[GeoWaveID].boubdingBox, training_label[0].Count(), GeoWaveArr[GeoWaveID].rc);

            //set partition
            child0.boubdingBox[1][dimIndex] = Maingridindex;
            child1.boubdingBox[0][dimIndex] = Maingridindex;

            //calc norm
            //calc mean value

            //DOCUMENT ON CHILDREN
            child0.dimIndex = dimIndex;
            child0.Maingridindex = Maingridindex;
            child1.dimIndex = dimIndex;
            child1.Maingridindex = Maingridindex;

            child0.MaingridValue = Form1.MainGrid[dimIndex][Maingridindex];
            child1.MaingridValue = Form1.MainGrid[dimIndex][Maingridindex];

            if (Form1.IsBoxSingular(child0.boubdingBox, rc.dim) || Form1.IsBoxSingular(child1.boubdingBox, rc.dim))
                return;

            //SHOULD I VERIFY THAT THE CHILD IS NOT ITS PARENT ? (IN CASES WHERE CAN'T MODEFY THE PARTITION)

            setChildrensPointsAndMeanValue(ref child0, ref child1, dimIndex, GeoWaveArr[GeoWaveID].pointsIdArray);
            //SET TWO CHILDS
            child0.parentID = child1.parentID = GeoWaveID;
            child0.child0 = child1.child0 = -1;
            child0.child1 = child1.child1 = -1;
            child0.level = child1.level = GeoWaveArr[GeoWaveID].level + 1;

            child0.computeNormOfConsts(GeoWaveArr[GeoWaveID]);
            child1.computeNormOfConsts(GeoWaveArr[GeoWaveID]);
            GeoWaveArr.Add(child0);
            GeoWaveArr.Add(child1);
            GeoWaveArr[GeoWaveID].child0 = GeoWaveArr.Count - 2;
            GeoWaveArr[GeoWaveID].child1 = GeoWaveArr.Count - 1;

            //// calculate gini index for childrens
            //if (rc.split_type == 3 || rc.split_type == 4)
            //{
            //    //could set information gain here
            //}

            ////RECURSION STEP !!!
            //recursiveBSP_WaveletsByConsts(GeoWaveArr, GeoWaveArr[GeoWaveID].child0);
            //recursiveBSP_WaveletsByConsts(GeoWaveArr, GeoWaveArr[GeoWaveID].child1);        
        }        

        private bool getBestPartitionResult(ref int dimIndex, ref int Maingridindex, List<GeoWave> GeoWaveArr, int GeoWaveID,ref double Error, bool[] Dims2Take)
        {
            double[][] error_dim_partition = new double[2][];//error, Maingridindex
            error_dim_partition[0] = new double[rc.dim];
            error_dim_partition[1] = new double[rc.dim];

            //PARALLEL RUN - SEARCHING BEST PARTITION IN ALL DIMS
            if (Form1.rumPrallel)
            {
                Parallel.For(0, rc.dim, i =>
                {
                    //double[] tmpResult = getBestPartition(i, GeoWaveArr[GeoWaveID]);
                    if (Dims2Take[i])
                    {
                        double[] tmpResult = getBestPartitionLargeDB(i, GeoWaveArr[GeoWaveID]);
                        error_dim_partition[0][i] = tmpResult[0];//error
                        error_dim_partition[1][i] = tmpResult[1];//Maingridindex                    
                    }
                    else
                    {
                        error_dim_partition[0][i] = double.MaxValue;//error
                        error_dim_partition[1][i] = -1;//Maingridindex                    
                    }
                });
            }
            else
            {
                for (int i = 0; i < rc.dim; i++)
                {
                    //double[] tmpResult = getBestPartition(i, GeoWaveArr[GeoWaveID]);
                    if (Dims2Take[i])
                    {
                        double[] tmpResult = getBestPartitionLargeDB(i, GeoWaveArr[GeoWaveID]);
                        error_dim_partition[0][i] = tmpResult[0];//error
                        error_dim_partition[1][i] = tmpResult[1];//Maingridindex                    
                    }
                    else
                    {
                        error_dim_partition[0][i] = double.MaxValue;//error
                        error_dim_partition[1][i] = -1;//Maingridindex                    
                    }
                }
            }

            dimIndex = Enumerable.Range(0, error_dim_partition[0].Count())
                .Aggregate((a, b) => (error_dim_partition[0][a] < error_dim_partition[0][b]) ? a : b);

            if (error_dim_partition[0][dimIndex] >= Error)
                return false;//if best partition doesn't help - return
            else
                Error = error_dim_partition[0][dimIndex];

            Maingridindex = Convert.ToInt32(error_dim_partition[1][dimIndex]);
            return true;
        }

        private bool Y_getBestPartitionResult(ref int dimIndex, ref int Maingridindex, ref double Error, List<GeoWave> GeoWaveArr, int GeoWaveID, bool[] Dims2Take, double[][] Y_DataBase, int dim)
        {
            double[][] error_dim_partition = new double[2][];//error, Maingridindex
            error_dim_partition[0] = new double[dim];
            error_dim_partition[1] = new double[dim];

            //PARALLEL RUN - SEARCHING BEST PARTITION IN ALL DIMS
            if (Form1.rumPrallel)
            {
                Parallel.For(0, dim, i =>
                {
                    //double[] tmpResult = getBestPartition(i, GeoWaveArr[GeoWaveID]);
                    if (Dims2Take[i])
                    {
                        double[] tmpResult = Y_getBestPartitionLargeDB(i, GeoWaveArr[GeoWaveID], Y_DataBase);
                        error_dim_partition[0][i] = tmpResult[0];//error
                        error_dim_partition[1][i] = tmpResult[1];//Maingridindex                    
                    }
                    else
                    {
                        error_dim_partition[0][i] = double.MaxValue;//error
                        error_dim_partition[1][i] = -1;//Maingridindex                    
                    }
                });
            }
            else
            {
                for (int i = 0; i < dim; i++)
                {
                    //double[] tmpResult = getBestPartition(i, GeoWaveArr[GeoWaveID]);
                    if (Dims2Take[i])
                    {
                        double[] tmpResult = Y_getBestPartitionLargeDB(i, GeoWaveArr[GeoWaveID], Y_DataBase);
                        error_dim_partition[0][i] = tmpResult[0];//error
                        error_dim_partition[1][i] = tmpResult[1];//Maingridindex                    
                    }
                    else
                    {
                        error_dim_partition[0][i] = double.MaxValue;//error
                        error_dim_partition[1][i] = -1;//Maingridindex                    
                    }
                }
            }

            dimIndex = Enumerable.Range(0, error_dim_partition[0].Count())
                .Aggregate((a, b) => (error_dim_partition[0][a] < error_dim_partition[0][b]) ? a : b);

            if (error_dim_partition[0][dimIndex] >= Error)
                return false;//if best partition doesn't help - return
            else
                Error = error_dim_partition[0][dimIndex];

            Maingridindex = Convert.ToInt32(error_dim_partition[1][dimIndex]);
            return true;
        }

        private bool Y_getBestPartitionPLSVars(ref int dimIndex, ref double splitValue, ref double Error, List<GeoWave> GeoWaveArr, int GeoWaveID, double[][] Y_DataBase, int dim)
        {
            double[][] error_dim_partition = new double[2][];//error, Maingridindex
            error_dim_partition[0] = new double[dim];
            error_dim_partition[1] = new double[dim];

            //PARALLEL RUN - SEARCHING BEST PARTITION IN ALL DIMS
            if (Form1.rumPrallel)
            {
                Parallel.For(0, dim, i =>
                {
                    double[] tmpResult = Y_getBestPartitionLargeDB(i, GeoWaveArr[GeoWaveID], Y_DataBase);
                    error_dim_partition[0][i] = tmpResult[0];//error
                    error_dim_partition[1][i] = tmpResult[1];//Maingridindex                    
                });
            }
            else
            {
                for (int i = 0; i < dim; i++)
                {
                    double[] tmpResult = Y_getBestPartitionLargeDB(i, GeoWaveArr[GeoWaveID], Y_DataBase);
                    error_dim_partition[0][i] = tmpResult[0];//error
                    error_dim_partition[1][i] = tmpResult[1];//Maingridindex                    
                }
            }

            dimIndex = Enumerable.Range(0, error_dim_partition[0].Count())
                .Aggregate((a, b) => (error_dim_partition[0][a] < error_dim_partition[0][b]) ? a : b);

            if (error_dim_partition[0][dimIndex] >= Error)
                return false;//if best partition doesn't help - return
            else
                Error = error_dim_partition[0][dimIndex];

            int splitIndex = Convert.ToInt32(error_dim_partition[1][dimIndex]);
            splitValue = Y_DataBase[splitIndex][dimIndex]; /// YTODO: split value is the value of the 'split instance'. where will this instance go?
            return true;
        }
        

        private double[] getBestPartitionLargeDB(int dimIndex, GeoWave geoWave)
        {
            double[] error_n_point = new double[2];//error index
            if (Form1.MainGrid[dimIndex].Count == 1)//empty feature
            {
                error_n_point[0] = double.MaxValue;
                error_n_point[1] = -1;
                return error_n_point;
            }
            //sort ids (for labels) acording to position at Form1.MainGrid[dimIndex][index]
            List<int> tmpIDs = new List<int>(geoWave.pointsIdArray);
            tmpIDs.Sort(delegate(int c1, int c2) { return training_dt[c1][dimIndex].CompareTo(training_dt[c2][dimIndex]); });

            if (training_dt[tmpIDs[0]][dimIndex] == training_dt[tmpIDs[tmpIDs.Count - 1]][dimIndex])//all values are the same 
            {
                error_n_point[0] = double.MaxValue;
                error_n_point[1] = -1;
                return error_n_point;
            }

            int best_ID = -1;
            double lowest_err = double.MaxValue;
            double[] leftAvg = new double[geoWave.MeanValue.Count()];
            double[] rightAvg = new double[geoWave.MeanValue.Count()];
            double[] leftErr = geoWave.calc_MeanValueReturnError(training_label, geoWave.pointsIdArray, ref leftAvg);//CONTAINES ALL POINTS - AT THE BEGINING
            double[] rightErr = new double[geoWave.MeanValue.Count()];
            //for (int i = 0; i < tmpIDs.Count; i++)
            //for (int j = 0; j < geoWave.MeanValue.Count(); j++)
            //{ 
            //    leftLabelSum[j] += training_label[tmpIDs[i]][j];
            //}

            double N_points = Convert.ToDouble(tmpIDs.Count);
            //double[] errorPointsArr = new double[tmpIDs.Count];//error 
            //double[] leftPointsArr = new double[tmpIDs.Count];//error
            //double[] rightPointsArr = new double[tmpIDs.Count];//error
            //double[] leftPointsArrAVG = new double[tmpIDs.Count];//error
            //double[] rightPointsArrAVG = new double[tmpIDs.Count];//error
            double tmp_err;


            //StreamWriter sw = null;
            //if (dimIndex == 1)
            //{
            //    sw = new StreamWriter(@"C:\Users\Oren\Dropbox\ADA\tmp\right_points_new.txt", false);
            //    sw.WriteLine("right_points_new_ID right_points_new_Val");
            //}


            for (int i = 0; i < tmpIDs.Count - 1; i++)//we dont calc the last (rightmost) boundary - it equal to the left most
            {
                tmp_err = 0;
                for (int j = 0; j < geoWave.MeanValue.Count(); j++)
                {
                    leftErr[j] = leftErr[j] - (N_points - i) * (training_label[tmpIDs[tmpIDs.Count - i - 1]][j] - leftAvg[j]) * (training_label[tmpIDs[tmpIDs.Count - i - 1]][j] - leftAvg[j]) / (N_points - i - 1);
                    leftAvg[j] = (N_points - i) * leftAvg[j] / (N_points - i - 1) - training_label[tmpIDs[tmpIDs.Count - i - 1]][j] / (N_points - i - 1);
                    rightErr[j] = rightErr[j] + (training_label[tmpIDs[tmpIDs.Count - i - 1]][j] - rightAvg[j]) * (training_label[tmpIDs[tmpIDs.Count - i - 1]][j] - rightAvg[j]) * Convert.ToDouble(i) / Convert.ToDouble(i + 1);
                    rightAvg[j] = rightAvg[j] * Convert.ToDouble(i) / Convert.ToDouble(i + 1) + training_label[tmpIDs[tmpIDs.Count - i - 1]][j] / Convert.ToDouble(i + 1);
                    tmp_err += leftErr[j] + rightErr[j];

                    //errorPointsArr[i] += leftErr[j] + rightErr[j];
                    //leftPointsArr[i] = leftErr[j];
                    //rightPointsArr[i] = rightErr[j];
                    //leftPointsArrAVG[i] = leftAvg[j];
                    //rightPointsArrAVG[i] = rightAvg[j];
                    //if (dimIndex == 1)
                    //{
                    //    sw.WriteLine(tmpIDs[tmpIDs.Count - i - 1] + " " + training_dt[tmpIDs[tmpIDs.Count - i - 1]][dimIndex]);
                    //    if (tmpIDs[tmpIDs.Count - i - 1] == 990)
                    //    {
                    //        sw.Close();
                    //    }
                    //}


                }
                //in case some points has the same values - we calc the avarage (relevant for splitting) only after all the points (with same values) had moved to the right
                //we don't alow "improving" the same split with two points with the same position (sort is not unique)
                if (lowest_err > tmp_err && training_dt[tmpIDs[tmpIDs.Count - i - 1]][dimIndex] != training_dt[tmpIDs[tmpIDs.Count - i - 2]][dimIndex]
                    && (i + 1) >= rc.minWaveSize && (i + rc.minWaveSize) < tmpIDs.Count && !Form1.trainNaTable.ContainsKey(new Tuple<int, int>(tmpIDs[tmpIDs.Count - i - 1], dimIndex)))
                {
                    best_ID = tmpIDs[tmpIDs.Count - i - 1];
                    lowest_err = tmp_err;
                }
            }


            //errorPointsArr[tmpIDs.Count - 1] = errorPointsArr[0];//we dont calc the last (rightmost) boundary - it equal to the left most

            ////search lowest error
            //int minIndex = Enumerable.Range(0, errorPointsArr.Length).Aggregate((a, b) => (errorPointsArr[a] < errorPointsArr[b]) ? a : b);

            if (best_ID == -1)
            {
                error_n_point[0] = double.MaxValue;
                error_n_point[1] = double.MaxValue;
                return error_n_point;
            }

            error_n_point[0] = Math.Max(lowest_err, 0);
            error_n_point[1] = training_GridIndex_dt[best_ID][dimIndex];
            //if (best_ID == tmpIDs[0] || best_ID == tmpIDs[tmpIDs.Count() - 1])// 
            //{
            //    long stop = 0;
            //    stop++;
            //}
            //=getMaingridIndex(geoWave.boubdingBox[0][dimIndex], Form1.MainGrid[dimIndex], training_dt[best_ID][dimIndex]);//MaingridIndex
            return error_n_point;
        }

        private double[] Y_getBestPartitionLargeDB(int dimIndex, GeoWave geoWave, double[][] Y_PLSvars)
        {
            double[] error_n_point = new double[2];//error index
            if (Form1.MainGrid[dimIndex].Count == 1)//empty feature
            {
                error_n_point[0] = double.MaxValue;
                error_n_point[1] = -1;
                return error_n_point;
            }
            //sort ids (for labels) acording to position at Form1.MainGrid[dimIndex][index]
            List<int> tmpIDs = new List<int>(geoWave.pointsIdArray);
            tmpIDs.Sort(delegate(int c1, int c2) { return Y_PLSvars[c1][dimIndex].CompareTo(Y_PLSvars[c2][dimIndex]); }); // 

            if (Y_PLSvars[tmpIDs[0]][dimIndex] == Y_PLSvars[tmpIDs[tmpIDs.Count - 1]][dimIndex])//all values are the same 
            {
                error_n_point[0] = double.MaxValue;
                error_n_point[1] = -1;
                return error_n_point;
            }

            int best_ID = -1;
            double lowest_err = double.MaxValue;
            double[] leftAvg = new double[geoWave.MeanValue.Count()];
            double[] rightAvg = new double[geoWave.MeanValue.Count()];
            double[] leftErr = geoWave.calc_MeanValueReturnError(training_label, geoWave.pointsIdArray, ref leftAvg);//CONTAINES ALL POINTS - AT THE BEGINING
            double[] rightErr = new double[geoWave.MeanValue.Count()];
            

            double N_points = Convert.ToDouble(tmpIDs.Count);
            double tmp_err;


            for (int i = 0; i < tmpIDs.Count - 1; i++)//we dont calc the last (rightmost) boundary - it equal to the left most
            {
                tmp_err = 0;
                for (int j = 0; j < geoWave.MeanValue.Count(); j++)
                {
                    leftErr[j] = leftErr[j] - (N_points - i) * (training_label[tmpIDs[tmpIDs.Count - i - 1]][j] - leftAvg[j]) * (training_label[tmpIDs[tmpIDs.Count - i - 1]][j] - leftAvg[j]) / (N_points - i - 1);
                    leftAvg[j] = (N_points - i) * leftAvg[j] / (N_points - i - 1) - training_label[tmpIDs[tmpIDs.Count - i - 1]][j] / (N_points - i - 1);
                    rightErr[j] = rightErr[j] + (training_label[tmpIDs[tmpIDs.Count - i - 1]][j] - rightAvg[j]) * (training_label[tmpIDs[tmpIDs.Count - i - 1]][j] - rightAvg[j]) * Convert.ToDouble(i) / Convert.ToDouble(i + 1);
                    rightAvg[j] = rightAvg[j] * Convert.ToDouble(i) / Convert.ToDouble(i + 1) + training_label[tmpIDs[tmpIDs.Count - i - 1]][j] / Convert.ToDouble(i + 1);
                    tmp_err += leftErr[j] + rightErr[j];

                }
                //in case some points has the same values - we calc the avarage (relevant for splitting) only after all the points (with same values) had moved to the right
                //we don't alow "improving" the same split with two points with the same position (sort is not unique)
                if (lowest_err > tmp_err && Y_PLSvars[tmpIDs[tmpIDs.Count - i - 1]][dimIndex] != Y_PLSvars[tmpIDs[tmpIDs.Count - i - 2]][dimIndex]
                    && (i + 1) >= rc.minWaveSize && (i + rc.minWaveSize) < tmpIDs.Count && !Form1.trainNaTable.ContainsKey(new Tuple<int, int>(tmpIDs[tmpIDs.Count - i - 1], dimIndex)))
                {
                    best_ID = tmpIDs[tmpIDs.Count - i - 1];
                    lowest_err = tmp_err;
                }
            }


            //errorPointsArr[tmpIDs.Count - 1] = errorPointsArr[0];//we dont calc the last (rightmost) boundary - it equal to the left most

            ////search lowest error
            //int minIndex = Enumerable.Range(0, errorPointsArr.Length).Aggregate((a, b) => (errorPointsArr[a] < errorPointsArr[b]) ? a : b);

            if (best_ID == -1)
            {
                error_n_point[0] = double.MaxValue;
                error_n_point[1] = double.MaxValue;
                return error_n_point;
            }

            error_n_point[0] = Math.Max(lowest_err, 0);
            error_n_point[1] = best_ID;//training_GridIndex_dt[best_ID][dimIndex];
            //if (best_ID == tmpIDs[0] || best_ID == tmpIDs[tmpIDs.Count() - 1])// 
            //{
            //    long stop = 0;
            //    stop++;
            //}
            //=getMaingridIndex(geoWave.boubdingBox[0][dimIndex], Form1.MainGrid[dimIndex], training_dt[best_ID][dimIndex]);//MaingridIndex
            return error_n_point;
        }

        private bool GetGiniPartitionResult(ref int dimIndex, ref int Maingridindex, List<GeoWave> GeoWaveArr, int GeoWaveID, double Error, bool[] Dims2Take)
        {
            double[][] error_dim_partition = new double[2][];//information gain, Maingridindex
            error_dim_partition[0] = new double[rc.dim];
            error_dim_partition[1] = new double[rc.dim];

            //PARALLEL RUN - SEARCHING BEST PARTITION IN ALL DIMS
            if (Form1.rumPrallel)
            {
                Parallel.For(0, rc.dim, i =>
                {
                    if (Dims2Take[i])
                    {
                        double[] tmpResult = getGiniPartitionLargeDB(i, GeoWaveArr[GeoWaveID]);
                        error_dim_partition[0][i] = tmpResult[0];//information gain
                        error_dim_partition[1][i] = tmpResult[1];//Maingridindex                    
                    }
                    else
                    {
                        error_dim_partition[0][i] = double.MinValue;//information gain
                        error_dim_partition[1][i] = -1;//Maingridindex                    
                    }
                });
            }
            else
            {
                for (int i = 0; i < rc.dim; i++)
                {
                    if (Dims2Take[i])
                    {
                        double[] tmpResult = getGiniPartitionLargeDB(i, GeoWaveArr[GeoWaveID]);
                        error_dim_partition[0][i] = tmpResult[0];//information gain
                        error_dim_partition[1][i] = tmpResult[1];//Maingridindex                    
                    }
                    else
                    {
                        error_dim_partition[0][i] = double.MinValue;//information gain
                        error_dim_partition[1][i] = -1;//Maingridindex                    
                    }
                }
            }

            dimIndex = Enumerable.Range(0, error_dim_partition[0].Count())
                .Aggregate((a, b) => (error_dim_partition[0][a] > error_dim_partition[0][b]) ? a : b); //maximal gain (>)

            if (error_dim_partition[0][dimIndex] <= 0)
                return false;//if best partition doesn't help - return

            Maingridindex = Convert.ToInt32(error_dim_partition[1][dimIndex]);
            return true;
        }

        private double[] getGiniPartitionLargeDB(int dimIndex, GeoWave geoWave)
        {
            double[] error_n_point = new double[2];//gain index
            if (Form1.MainGrid[dimIndex].Count == 1)//empty feature
            {
                error_n_point[0] = double.MinValue;//min gain
                error_n_point[1] = -1;
                return error_n_point;
            }
            //sort ids (for labels) acording to position at Form1.MainGrid[dimIndex][index]
            List<int> tmpIDs = new List<int>(geoWave.pointsIdArray);
            tmpIDs.Sort(delegate(int c1, int c2) { return training_dt[c1][dimIndex].CompareTo(training_dt[c2][dimIndex]); });

            if (training_dt[tmpIDs[0]][dimIndex] == training_dt[tmpIDs[tmpIDs.Count - 1]][dimIndex])//all values are the same 
            {
                error_n_point[0] = double.MinValue;//min gain
                error_n_point[1] = -1;
                return error_n_point;
            }

            Dictionary<double, double> leftcategories = new Dictionary<double, double>(); //double as counter to enable devision
            Dictionary<double, double> rightcategories = new Dictionary<double, double>(); //double as counter to enable devision
            for (int i = 0; i < tmpIDs.Count(); i++)
            {
                if (leftcategories.ContainsKey(training_label[tmpIDs[i]][0]))
                    leftcategories[training_label[tmpIDs[i]][0]] += 1;
                else
                    leftcategories.Add(training_label[tmpIDs[i]][0], 1);
            }
            double N_points = Convert.ToDouble(tmpIDs.Count);
            double initialGini = calcGini(leftcategories, N_points);
            double NpointsLeft = N_points;
            double NpointsRight = 0;
            double leftGini = 0;
            double rightGini = 0;
            double gain = 0;
            double bestGain = 0;
            int best_ID = -1;

            //Dictionary<double, double> dbgRight = new Dictionary<double, double>(); //double as counter to enable devision
            //Dictionary<double, double> dbgLeft = new Dictionary<double, double>(); //double as counter to enable devision

            for (int i = 0; i < tmpIDs.Count - 1; i++)//we dont calc the last (rightmost) boundary - it equal to the left most
            {
                double rightMostLable = training_label[tmpIDs[tmpIDs.Count - i - 1]][0];

                if (leftcategories[rightMostLable] == 1)
                    leftcategories.Remove(rightMostLable);
                else
                    leftcategories[rightMostLable] -= 1;

                if (rightcategories.ContainsKey(rightMostLable))
                    rightcategories[rightMostLable] += 1;
                else
                    rightcategories.Add(rightMostLable, 1);

                NpointsLeft -= 1;
                NpointsRight += 1;

                leftGini = calcGini(leftcategories, NpointsLeft);
                rightGini = calcGini(rightcategories, NpointsRight);

                gain = (initialGini - leftGini) * (NpointsLeft / N_points) + (initialGini - rightGini) * (NpointsRight / N_points);

                //in case some points has the same values (in this dim) - we calc the avarage (relevant for splitting) only after all the points (with same values) had moved to the right
                //we don't alow "improving" the same split with two points with the same position (sort is not unique)
                if (gain > bestGain && training_dt[tmpIDs[tmpIDs.Count - i - 1]][dimIndex] != training_dt[tmpIDs[tmpIDs.Count - i - 2]][dimIndex]
                    && (i + 1) >= rc.minWaveSize && (i + rc.minWaveSize) < tmpIDs.Count 
                    && !Form1.trainNaTable.ContainsKey(new Tuple<int, int>(tmpIDs[tmpIDs.Count - i - 1], dimIndex)))
                {
                    best_ID = tmpIDs[tmpIDs.Count - i - 1];
                    bestGain = gain;
                    //dbgRight = rightcategories.ToDictionary(entry => entry.Key,
                    //                           entry => entry.Value);
                    //dbgLeft = leftcategories.ToDictionary(entry => entry.Key,
                    //                           entry => entry.Value);

                }
            }

            if (best_ID == -1)
            {
                error_n_point[0] = double.MinValue;//min gain
                error_n_point[1] = -1;
                return error_n_point;
            }

            error_n_point[0] = bestGain;
            error_n_point[1] = training_GridIndex_dt[best_ID][dimIndex];

            return error_n_point;
        }

        private double calcGini(Dictionary<double, double> Totalcategories, double Npoints)
        {
            double gini = 0;
            for (int i = 0; i < Totalcategories.Count; i++)
            {
                gini += (Totalcategories.ElementAt(i).Value / Npoints) * (1 - (Totalcategories.ElementAt(i).Value / Npoints));
            }
            return gini;
        }

        private bool getRandPartitionResult(ref int dimIndex, ref int Maingridindex, List<GeoWave> GeoWaveArr, int GeoWaveID, double Error, int seed=0)
        {
            Random rnd0 = new Random(seed);
            int seedIndex = rnd0.Next(0, Int16.MaxValue/2); 

            Random rnd = new Random(seedIndex + GeoWaveID);

            int counter = 0;
            bool partitionFound= false;

            while(!partitionFound && counter < 20)
            {
                counter++;
                dimIndex = rnd.Next(0, GeoWaveArr[0].rc.dim); // creates a number between 0 and GeoWaveArr[0].rc.dim 
                int partition_ID = GeoWaveArr[GeoWaveID].pointsIdArray[rnd.Next(1, GeoWaveArr[GeoWaveID].pointsIdArray.Count() - 1)];

                Maingridindex = Convert.ToInt32(training_GridIndex_dt[partition_ID][dimIndex]);//this is dangerouse for Maingridindex > 2^32
                if (!Form1.trainNaTable.ContainsKey(new Tuple<int, int>(partition_ID, dimIndex)))
                    return true;
            }

            //dimIndex = rnd.Next(0, GeoWaveArr[0].rc.dim); // creates a number between 0 and GeoWaveArr[0].rc.dim 
            //int tmpDim = dimIndex;

            //sort ids (for labels) acording to position at Form1.MainGrid[dimIndex][index]
            //List<int> tmpIDs = new List<int>(GeoWaveArr[GeoWaveID].pointsIdArray);
            //tmpIDs.Sort(delegate(int c1, int c2) { return training_dt[c1][tmpDim].CompareTo(training_dt[c2][tmpDim]); });

            //int partition_ID = tmpIDs[rnd.Next(1, tmpIDs.Count - 1)];
            //int partition_ID = GeoWaveArr[GeoWaveID].pointsIdArray[rnd.Next(1, GeoWaveArr[GeoWaveID].pointsIdArray.Count() - 1)];


            //Maingridindex = Convert.ToInt32(training_GridIndex_dt[partition_ID][dimIndex]);//this is dangerouse for Maingridindex > 2^32

            return false;
        }

        private void setChildrensPointsAndMeanValue(ref GeoWave child0, ref GeoWave child1, int dimIndex, List<int> indexArr)
        {
            child0.MeanValue.Multiply(0);
            child1.MeanValue.Multiply(0);

            //GO OVER ALL POINTS IN REGION
            for (int i = 0; i < indexArr.Count; i++)
            {
                if (training_dt[indexArr[i]][dimIndex] < Form1.MainGrid[dimIndex].ElementAt(child0.boubdingBox[1][dimIndex]))
                {
                    for (int j = 0; j < training_label[0].Count(); j++)
                        child0.MeanValue[j] += training_label[indexArr[i]][j];
                    child0.pointsIdArray.Add(indexArr[i]);
                }
                else
                {
                    for (int j = 0; j < training_label[0].Count(); j++)
                        child1.MeanValue[j] += training_label[indexArr[i]][j];
                    child1.pointsIdArray.Add(indexArr[i]);
                }
            }
            if(child0.pointsIdArray.Count > 0)
                child0.MeanValue = child0.MeanValue.Divide(Convert.ToDouble(child0.pointsIdArray.Count));
            if (child1.pointsIdArray.Count > 0)
                child1.MeanValue = child1.MeanValue.Divide(Convert.ToDouble(child1.pointsIdArray.Count));
        }

        private void Y_setChildrensPointsAndMeanValuePLSSplit(ref GeoWave child0, ref GeoWave child1, double[] splitVector, double splitValue, List<int> indexArr)
        {
            child0.MeanValue.Multiply(0);
            child1.MeanValue.Multiply(0);

            //GO OVER ALL POINTS IN REGION
            for (int i = 0; i < indexArr.Count; i++)
            {
                int index = indexArr[i];
                if (splitVector[index] < splitValue)
                    //if (training_dt[indexArr[i]][dimIndex] < Form1.MainGrid[dimIndex].ElementAt(child0.boubdingBox[1][dimIndex]))
                {
                    for (int j = 0; j < training_label[0].Count(); j++)
                        child0.MeanValue[j] += training_label[index][j];
                    child0.pointsIdArray.Add(index);
                }
                else
                {
                    for (int j = 0; j < training_label[0].Count(); j++)
                        child1.MeanValue[j] += training_label[index][j];
                    child1.pointsIdArray.Add(index);
                }
            }
            if (child0.pointsIdArray.Count > 0)
                child0.MeanValue = child0.MeanValue.Divide(Convert.ToDouble(child0.pointsIdArray.Count));
            if (child1.pointsIdArray.Count > 0)
                child1.MeanValue = child1.MeanValue.Divide(Convert.ToDouble(child1.pointsIdArray.Count));
        }

        private bool[] getDim2Take(recordConfig rc, int Seed)
        {
            bool[] Dim2Take = new bool[rc.dim];

            var ran = new Random(Seed);
            //List<int> dimArr = Enumerable.Range(0, rc.dim).OrderBy(x => ran.Next()).ToList().GetRange(0, rc.dim);
            //List<int> dimArr = Enumerable.Range(0, rc.dim).OrderBy(x => ran.Next()).ToList().GetRange(0, rc.dim);
            for (int i = 0; i < rc.NDimsinRF; i++)
            {
                //Dim2Take[dimArr[i]] = true;
                int index = ran.Next(0, rc.dim);
                if (Dim2Take[index] == true)
                    i--;
                else
                    Dim2Take[index] = true;
            }
            return Dim2Take;
        }

        //private void NonrecursiveBSP_WaveletsByConsts(List<GeoWave> GeoWaveArr, int GeoWaveID)
        //{
        //    int i = 0;
        //    while (i < GeoWaveArr.Count)
        //    {
        //        set_BSP_WaveletsByConsts(GeoWaveArr, i);
        //        i++;
        //    }
        //}

        //private void NonrecursiveBSP_WaveletsByConsts(List<GeoWave> GeoWaveArr, int GeoWaveID, int seed)
        //{
        //    int i = 0;
        //    while (i < GeoWaveArr.Count)
        //    {
        //        set_BSP_WaveletsByConsts(GeoWaveArr, i, seed);
        //        i++;
        //    }
        //}

        //private void recursiveBSP_WaveletsByConsts(List<GeoWave> GeoWaveArr, int GeoWaveID)
        //{           
        //    //CALC APPROX_SOLUTION FOR GEO WAVE
        //    //int dbg = 0;
        //    //if (GeoWaveArr[GeoWaveID].pointsIdArray.Count == 0)
        //    //    dbg++;
        //    //if (GeoWaveID == 10)
        //    //    dbg++;
        //    double Error = GeoWaveArr[GeoWaveID].calc_MeanValueReturnError(training_label,GeoWaveArr[GeoWaveID].pointsIdArray);
        //    if (Error < rc.approxThresh || GeoWaveArr[GeoWaveID].pointsIdArray.Count() <= rc.minWaveSize
        //        || GeoWaveArr[GeoWaveID].level >= rc.BoundLevel)
        //        return;

        //    int dimIndex=-1;
        //    int Maingridindex=-1;

        //    bool IsPartitionOK = false;
        //    if(rc.split_type == 0)
        //        IsPartitionOK = getBestPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, Error, Dime2Take);//consider dropping Dime2Take
        //    else if (rc.split_type == 1)//rand split
        //        IsPartitionOK = getRandPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, Error);
        //    else if (rc.split_type == 2)//rand features in each node
        //    {
        //        bool[] Dim2TakeNode = getDim2Take(rc, GeoWaveID);
        //        IsPartitionOK = getBestPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, Error, Dim2TakeNode);
        //    }
        //    else if (rc.split_type == 3)//Gini split
        //    {
        //        IsPartitionOK = GetGiniPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, Error, Dime2Take);            
        //    }
        //    else if (rc.split_type == 4)//Gini split + rand node
        //    {
        //        bool[] Dim2TakeNode = getDim2Take(rc, GeoWaveID);
        //        IsPartitionOK = GetGiniPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, Error, Dim2TakeNode);
        //    }

        //    if (!IsPartitionOK)
        //        return;

        //    GeoWave child0 = new GeoWave(GeoWaveArr[GeoWaveID].boubdingBox,training_label[0].Count(),GeoWaveArr[GeoWaveID].rc);
        //    GeoWave child1 = new GeoWave(GeoWaveArr[GeoWaveID].boubdingBox, training_label[0].Count(), GeoWaveArr[GeoWaveID].rc);

        //    //set partition
        //    child0.boubdingBox[1][dimIndex] = Maingridindex;
        //    child1.boubdingBox[0][dimIndex] = Maingridindex;

        //    //DOCUMENT ON CHILDREN
        //    child0.dimIndex = dimIndex;
        //    child0.Maingridindex = Maingridindex;            
        //    child1.dimIndex = dimIndex;
        //    child1.Maingridindex = Maingridindex;

        //    child0.MaingridValue = Form1.MainGrid[dimIndex][Maingridindex];
        //    child1.MaingridValue = Form1.MainGrid[dimIndex][Maingridindex];

        //    //calc norm
        //    //calc mean value

        //    if (Form1.IsBoxSingular(child0.boubdingBox, rc.dim) || Form1.IsBoxSingular(child1.boubdingBox, rc.dim))
        //        return;

        //    //SHOULD I VERIFY THAT THE CHILD IS NOT ITS PARENT ? (IN CASES WHERE CAN'T MODEFY THE PARTITION)

        //    setChildrensPointsAndMeanValue(ref child0, ref child1, dimIndex, GeoWaveArr[GeoWaveID].pointsIdArray);
        //    //SET TWO CHILDS
        //    child0.parentID = child1.parentID = GeoWaveID;
        //    child0.child0 = child1.child0 = -1;
        //    child0.child1 = child1.child1 = -1;
        //    child0.level = child1.level = GeoWaveArr[GeoWaveID].level + 1;

        //    child0.computeNormOfConsts(GeoWaveArr[GeoWaveID]);
        //    child1.computeNormOfConsts(GeoWaveArr[GeoWaveID]);
        //    GeoWaveArr.Add(child0);
        //    GeoWaveArr.Add(child1);
        //    GeoWaveArr[GeoWaveID].child0 = GeoWaveArr.Count - 2;
        //    GeoWaveArr[GeoWaveID].child1 = GeoWaveArr.Count - 1;

        //    //// calculate gini index for childrens
        //    //if (rc.split_type == 3 || rc.split_type == 4)
        //    //{
        //    //    //could set information gain here
        //    //}

        //    //RECURSION STEP !!!
        //    recursiveBSP_WaveletsByConsts(GeoWaveArr, GeoWaveArr[GeoWaveID].child0);
        //    recursiveBSP_WaveletsByConsts(GeoWaveArr, GeoWaveArr[GeoWaveID].child1);
        //}

        //private void set_BSP_WaveletsByConsts(List<GeoWave> GeoWaveArr, int GeoWaveID)
        //{
        //    //CALC APPROX_SOLUTION FOR GEO WAVE
        //    double Error = GeoWaveArr[GeoWaveID].calc_MeanValueReturnError(training_label, GeoWaveArr[GeoWaveID].pointsIdArray);
        //    if (Error < rc.approxThresh || GeoWaveArr[GeoWaveID].pointsIdArray.Count() <= rc.minWaveSize)
        //        return;

        //    int dimIndex = -1;
        //    int Maingridindex = -1;

        //    bool IsPartitionOK = false;
        //    if (rc.split_type == 0)
        //        IsPartitionOK = getBestPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, Error, Dime2Take);//consider dropping Dime2Take
        //    else if (rc.split_type == 1)//rand split
        //        IsPartitionOK = getRandPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, Error);
        //    else if (rc.split_type == 2)//rand features in each node
        //    {
        //        bool[] Dim2TakeNode = getDim2Take(rc, GeoWaveID);
        //        IsPartitionOK = getBestPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, Error, Dim2TakeNode);
        //    }
        //    else if (rc.split_type == 3)//Gini split
        //    {
        //        IsPartitionOK = GetGiniPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, Error, Dime2Take);
        //    }
        //    else if (rc.split_type == 4)//Gini split + rand node
        //    {
        //        bool[] Dim2TakeNode = getDim2Take(rc, GeoWaveID);
        //        IsPartitionOK = GetGiniPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, Error, Dim2TakeNode);
        //    }

        //    //bool IsPartitionOK = getBestPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, Error);
        //    //bool IsPartitionOK = getRandPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, Error);
        //    //bool IsPartitionOK = getBestPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, Error, Dime2Take);

        //    if (!IsPartitionOK)
        //        return;

        //    GeoWave child0 = new GeoWave(GeoWaveArr[GeoWaveID].boubdingBox, training_label[0].Count(), GeoWaveArr[GeoWaveID].rc);
        //    GeoWave child1 = new GeoWave(GeoWaveArr[GeoWaveID].boubdingBox, training_label[0].Count(), GeoWaveArr[GeoWaveID].rc);

        //    //set partition
        //    child0.boubdingBox[1][dimIndex] = Maingridindex;
        //    child1.boubdingBox[0][dimIndex] = Maingridindex;

        //    //DOCUMENT ON CHILDREN
        //    child0.dimIndex = dimIndex;
        //    child0.Maingridindex = Maingridindex;
        //    child1.dimIndex = dimIndex;
        //    child1.Maingridindex = Maingridindex;

        //    child0.MaingridValue = Form1.MainGrid[dimIndex][Maingridindex];
        //    child1.MaingridValue = Form1.MainGrid[dimIndex][Maingridindex];

        //    //calc norm
        //    //calc mean value

        //    if (Form1.IsBoxSingular(child0.boubdingBox, rc.dim) || Form1.IsBoxSingular(child1.boubdingBox, rc.dim))
        //        return;

        //    //SHOULD I VERIFY THAT THE CHILD IS NOT ITS PARENT ? (IN CASES WHERE CAN'T MODEFY THE PARTITION)

        //    setChildrensPointsAndMeanValue(ref child0, ref child1, dimIndex, GeoWaveArr[GeoWaveID].pointsIdArray);
        //    //SET TWO CHILDS
        //    child0.parentID = child1.parentID = GeoWaveID;
        //    child0.child0 = child1.child0 = -1;
        //    child0.child1 = child1.child1 = -1;
        //    child0.level = child1.level = GeoWaveArr[GeoWaveID].level + 1;

        //    child0.computeNormOfConsts(GeoWaveArr[GeoWaveID]);
        //    child1.computeNormOfConsts(GeoWaveArr[GeoWaveID]);
        //    GeoWaveArr.Add(child0);
        //    GeoWaveArr.Add(child1);
        //    GeoWaveArr[GeoWaveID].child0 = GeoWaveArr.Count - 2;
        //    GeoWaveArr[GeoWaveID].child1 = GeoWaveArr.Count - 1;

        //    //// calculate gini index for childrens
        //    //if (rc.split_type == 3 || rc.split_type == 4)
        //    //{
        //    //    //could set information gain here
        //    //}

        //    ////RECURSION STEP !!!
        //    //recursiveBSP_WaveletsByConsts(GeoWaveArr, GeoWaveArr[GeoWaveID].child0);
        //    //recursiveBSP_WaveletsByConsts(GeoWaveArr, GeoWaveArr[GeoWaveID].child1);        
        //}

        //private bool getBestPartitionResult(ref int dimIndex, ref int Maingridindex, List<GeoWave> GeoWaveArr, int GeoWaveID, double Error)
        //{
        //    double[][] error_dim_partition = new double[2][];//error, Maingridindex
        //    error_dim_partition[0] = new double[rc.dim];
        //    error_dim_partition[1] = new double[rc.dim];

        //    //PARALLEL RUN - SEARCHING BEST PARTITION IN ALL DIMS
        //    if (Form1.rumPrallel)
        //    {
        //        Parallel.For(0, rc.dim, i =>
        //        {
        //            //double[] tmpResult = getBestPartition(i, GeoWaveArr[GeoWaveID]);
        //            double[] tmpResult = getBestPartitionLargeDB(i, GeoWaveArr[GeoWaveID]);
        //            error_dim_partition[0][i] = tmpResult[0];//error
        //            error_dim_partition[1][i] = tmpResult[1];//Maingridindex
        //        });
        //    }
        //    else
        //    {
        //        for (int i = 0; i < rc.dim; i++)
        //        {
        //            //double[] tmpResult = getBestPartition(i, GeoWaveArr[GeoWaveID]);
        //            double[] tmpResult = getBestPartitionLargeDB(i, GeoWaveArr[GeoWaveID]);
        //            error_dim_partition[0][i] = tmpResult[0];//error
        //            error_dim_partition[1][i] = tmpResult[1];//Maingridindex
        //        }
        //    }

        //    dimIndex = Enumerable.Range(0, error_dim_partition[0].Count())
        //        .Aggregate((a, b) => (error_dim_partition[0][a] < error_dim_partition[0][b]) ? a : b);

        //    if (error_dim_partition[0][dimIndex] >= Error)
        //        return false;//if best partition doesn't help - return

        //    Maingridindex = Convert.ToInt32(error_dim_partition[1][dimIndex]);
        //    return true;
        //}

        //private double[] getBestPartition(int dimIndex, GeoWave geoWave)
        //{
        //    ////find grid boundaries in geowave
        //    //int lowerIndex = getIndexinGrid(geoWave.boubdingBox[0][dimIndex],dimIndex );
        //    //int upperIndex = getIndexinGrid(geoWave.boubdingBox[1][dimIndex], dimIndex);
        //    //int iterations = 1+ upperIndex - lowerIndex;

        //    int arrSize = geoWave.boubdingBox[1][dimIndex] - geoWave.boubdingBox[0][dimIndex] + 1;

        //    double[] errorPointsArr = new double[arrSize];//error 

        //    ////PARALLEL RUN OF EVALUATING ERROR IN PARTITION
        //    if (Form1.rumPrallel)
        //    {
        //        Parallel.For(0, arrSize, i =>
        //        {
        //            errorPointsArr[i] = evaluateErrorInPartition(dimIndex, geoWave.boubdingBox[0][dimIndex] + i, geoWave);
        //        });
        //    }
        //    else
        //    {
        //        for (int i = 0; i < arrSize; i++)
        //        {
        //            errorPointsArr[i] = evaluateErrorInPartition(dimIndex, geoWave.boubdingBox[0][dimIndex] + i, geoWave);
        //        }
        //    }

        //    //search lowest error
        //    int minIndex = Enumerable.Range(0, errorPointsArr.Length).Aggregate((a, b) => (errorPointsArr[a] < errorPointsArr[b]) ? a : b);

        //    double[] error_n_point = new double[2];//error index
        //    error_n_point[0] = errorPointsArr[minIndex];
        //    error_n_point[1] = geoWave.boubdingBox[0][dimIndex] + minIndex;//MaingridIndex
        //    return error_n_point;
        //}

        //private bool getRandPartitionResult(ref int dimIndex, ref int Maingridindex, List<GeoWave> GeoWaveArr, int GeoWaveID, double Error)
        //{
        //    Random rnd = new Random(GeoWaveID);
        //    dimIndex = rnd.Next(0, GeoWaveArr[0].rc.dim); // creates a number between 0 and GeoWaveArr[0].rc.dim 

        //    int tmpDim = dimIndex;

        //    //sort ids (for labels) acording to position at Form1.MainGrid[dimIndex][index]
        //    List<int> tmpIDs = new List<int>(GeoWaveArr[GeoWaveID].pointsIdArray);
        //    tmpIDs.Sort(delegate(int c1, int c2) { return training_dt[c1][tmpDim].CompareTo(training_dt[c2][tmpDim]); });

        //    int partition_ID = tmpIDs[rnd.Next(1, tmpIDs.Count - 1)];
        //    if (training_dt[tmpIDs[0]][tmpDim] == training_dt[tmpIDs[tmpIDs.Count()-1]][tmpDim])//all values are the same - dont part 
        //        return false;
        //    if (tmpIDs.Count == 2 ||  training_GridIndex_dt[partition_ID][dimIndex] == training_GridIndex_dt[tmpIDs[0]][dimIndex])//the { null ][ 20 20 50 but 20 20 ][ 50 is ok }
        //        partition_ID =tmpIDs[tmpIDs.Count()-1];
        //        //int partition_ID = 0;// GeoWaveArr[GeoWaveID].pointsIdArray[rnd.Next(1, GeoWaveArr[GeoWaveID].pointsIdArray.Count() - 1)];

        //    //if (GeoWaveArr[GeoWaveID].pointsIdArray.Count <= 2 )
        //    //    partition_ID = GeoWaveArr[GeoWaveID].pointsIdArray[0];
        //    //else
        //    //    partition_ID = GeoWaveArr[GeoWaveID].pointsIdArray[rnd.Next(1, GeoWaveArr[GeoWaveID].pointsIdArray.Count() - 1)];


        //    Maingridindex = Convert.ToInt32(training_GridIndex_dt[partition_ID][dimIndex]);//this is dangerouse for Maingridindex > 2^32

        //    return true;
        //}        

        //private double getMaingridIndex(int startIndex, List<double> list, double value)
        //{
        //    for (int i = startIndex; i < list.Count; i++)
        //        if (list[i] > value)
        //            return (Convert.ToDouble(i - 1));
        //    return 0;
        //}

        //private int getIndexinGrid(double val, int dimIndex)
        //{
        //    int first = 0;
        //    int last = Form1.MainGrid[dimIndex].Count - 1;
        //    int mid=0;
        //    mid = (first + last) / 2;
        //    while (first < last)
        //    {
        //        if (Form1.MainGrid[dimIndex][mid] == val)
        //            return mid;
        //        else if (val < Form1.MainGrid[dimIndex][mid])
        //            first = mid + 1;
        //        else 
        //            last = mid - 1;
        //        mid = (first + last) / 2;
        //    }
        //    return mid;
        //}

        //private double evaluateErrorInPartition(int dimIndex, int index, GeoWave geoWave)
        //{
        //    //create two lists of ids
        //    List<int> LeftIdArr = new List<int>();
        //    List<int> RightIdArr = new List<int>();
        //    double partitionPoint = Form1.MainGrid[dimIndex][index];

        //    for (int i = 0; i < geoWave.pointsIdArray.Count; i++)
        //    {
        //        if (training_dt[geoWave.pointsIdArray[i]][dimIndex] <= partitionPoint)
        //            LeftIdArr.Add(geoWave.pointsIdArray[i]);
        //        else
        //            RightIdArr.Add(geoWave.pointsIdArray[i]);
        //    } 
        //    double errLeft = geoWave.calc_MeanValueReturnError(training_label, LeftIdArr);
        //    double errRight = geoWave.calc_MeanValueReturnError(training_label, RightIdArr);

        //    //if (index == 741 && dimIndex == 8)
        //    //{
        //    //    StreamWriter sw = new StreamWriter(@"C:\Users\Oren\Dropbox\ADA\tmp\right_points_old.txt", false);

        //    //    sw.WriteLine("right_points_old_ID right_points_old_val");
        //    //    for (int k = 0; k < RightIdArr.Count; k++)
        //    //    {
        //    //        sw.WriteLine(RightIdArr[k] + " " + training_dt[RightIdArr[k]][dimIndex]); 
        //    //    }

        //    //    StreamWriter sw2 = new StreamWriter(@"C:\Users\Oren\Dropbox\ADA\tmp\left_points_old.txt", false);

        //    //    sw2.WriteLine("left_points_old_ID left_points_old_val");
        //    //    for (int k = 0; k < LeftIdArr.Count; k++)
        //    //    {
        //    //        sw2.WriteLine(LeftIdArr[k] + " " + training_dt[LeftIdArr[k]][dimIndex]);
        //    //    }
        //    //    sw.Close();
        //    //        sw2.Close();
        //    //}

        //    return errLeft + errRight;

        //    ////find avarage in each side
        //    //double partitionPoint = Form1.MainGrid[dimIndex][index];
        //    //double avgLeft = 0;
        //    //double avgRight = 0;
        //    //double countLeft = 0;
        //    //double countRight = 0;
        //    //for (int i = 0; i < geoWave.pointsIdArray.Count; i++)
        //    //{
        //    //    if (training_dt[geoWave.pointsIdArray[i]][dimIndex] <= partitionPoint)
        //    //    {
        //    //        avgLeft += training_label[geoWave.pointsIdArray[i]][dimIndex];
        //    //        countLeft += 1;
        //    //    }
        //    //    else
        //    //    {
        //    //        avgRight += training_label[geoWave.pointsIdArray[i]][dimIndex];
        //    //        countRight += 1;
        //    //    }
        //    //}
        //    //if (countLeft > 0)
        //    //    avgLeft /= countLeft;
        //    //if (countRight > 0)
        //    //    avgRight /= countRight;

        //    ////find error in each side
        //    //double errLeft = 0;
        //    //double errRight = 0;
        //    //if(rc.partitionErrType == 2)
        //    //{
        //    //    for (int i = 0; i < geoWave.pointsIdArray.Count; i++)
        //    //    {
        //    //        if (training_dt[geoWave.pointsIdArray[i]][dimIndex] <= partitionPoint)
        //    //            errLeft += (training_label[geoWave.pointsIdArray[i]][dimIndex] - avgLeft) * (training_label[geoWave.pointsIdArray[i]][dimIndex] - avgLeft);
        //    //        else
        //    //            errRight += (training_label[geoWave.pointsIdArray[i]][dimIndex] - avgRight) * (training_label[geoWave.pointsIdArray[i]][dimIndex] - avgRight);
        //    //    }   
        //    //    Math.
        //    //}
        //    //else if (rc.partitionErrType == 2)
        //    //{
        //    //    for (int i = 0; i < geoWave.pointsIdArray.Count; i++)
        //    //    {
        //    //        if (training_dt[geoWave.pointsIdArray[i]][dimIndex] <= partitionPoint)
        //    //            errLeft += (training_label[geoWave.pointsIdArray[i]][dimIndex] - avgLeft) * (training_label[geoWave.pointsIdArray[i]][dimIndex] - avgLeft);
        //    //        else
        //    //            errRight += (training_label[geoWave.pointsIdArray[i]][dimIndex] - avgRight) * (training_label[geoWave.pointsIdArray[i]][dimIndex] - avgRight);
        //    //    }
        //    //}

        //    //return errLeft + errRight;
        //}

    }
}

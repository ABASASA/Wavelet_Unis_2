using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using LibOptimization.Optimization;
using LibOptimization.Util;

namespace DataSetsSparsity
{
    class decicionTree
    {
        private double[][] training_dt;
        private long[][] training_GridIndex_dt;
        private double[][] training_label;
        private bool[] Dime2Take;
        private int m_dimenstion;
        private double[] m_MeanPositionForSplit_5;

        public decicionTree( DB db, bool[] Dime2Take)
        {
            this.training_dt = db.training_dt;
            this.training_label = db.training_label;
            this.training_GridIndex_dt = db.DBtraining_GridIndex_dt;
            this.Dime2Take = Dime2Take;
            this.m_dimenstion = this.training_dt[0].Count();
            this.m_MeanPositionForSplit_5 = new double[m_dimenstion];
        }

        public List<GeoWave> getdecicionTree(List<int> trainingArr, int[][] boundingBox, int seed = -1)
        {
            //CREATE DECISION_GEOWAVEARR
            List<GeoWave> decision_GeoWaveArr = new List<GeoWave>();

            //SET ROOT WAVELETE
            GeoWave gwRoot = new GeoWave(training_dt[0].Count(), training_label[0].Count());

            //SET REGION POINTS IDS
            gwRoot.pointsIdArray = trainingArr;
            boundingBox.CopyTo(gwRoot.boubdingBox, 0);

            decision_GeoWaveArr.Add(gwRoot);
            DecomposeWaveletsByConsts(decision_GeoWaveArr, seed);

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
            GeoWaveArr[0].computeNormOfConsts(Convert.ToDouble(userConfig.partitionType));
            GeoWaveArr[0].level = 0;

            if (seed == -1)
                recursiveBSP_WaveletsByConsts(GeoWaveArr, 0);
            else recursiveBSP_WaveletsByConsts(GeoWaveArr, 0, seed);//0 is the root index
        }

        private void recursiveBSP_WaveletsByConsts(List<GeoWave> GeoWaveArr, int GeoWaveID, int seed=0)
        {
            //CALC APPROX_SOLUTION FOR GEO WAVE
            double Error = GeoWaveArr[GeoWaveID].calc_MeanValueReturnError(training_label, GeoWaveArr[GeoWaveID].pointsIdArray);
            if (Error < userConfig.approxThresh || GeoWaveArr[GeoWaveID].pointsIdArray.Count() <= userConfig.minNodeSize || userConfig.boundLevelDepth <=  GeoWaveArr[GeoWaveID].level)
                return;

            int dimIndex = -1;
            int Maingridindex = -1;

            bool IsPartitionOK = false;
            var ran1 = new Random(seed);
            var ran2 = new Random(GeoWaveID);
            int one = ran1.Next(0, int.MaxValue / 10);
            int two = ran2.Next(0, int.MaxValue / 10);
            bool[] Dim2TakeNode = getDim2Take(one + two);
            double[] hyperPlane = new double[this.m_dimenstion + 1];


            // ASAFAB -old implemention : IsPartitionOK = getBestPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, Error, Dim2TakeNode);
            IsPartitionOK = GetUnisotropicParition(GeoWaveArr, GeoWaveID, Error, out hyperPlane, Dim2TakeNode);
            dimIndex = 0;
            Maingridindex = 0;

            //MAKING SURE WE DON'T STOP BECAUSE OF SEARCHING THE WRONG FEATURES
            if (!IsPartitionOK)
            {
                for (int i = 0; i < Dim2TakeNode.Count(); i++)
                    Dim2TakeNode[i] = (Dim2TakeNode[i] == true) ? false : true;
                IsPartitionOK = getBestPartitionResult(ref dimIndex, ref Maingridindex, GeoWaveArr, GeoWaveID, Error, Dim2TakeNode);
                if (!IsPartitionOK)
                    return;
            }

            GeoWave child0 = new GeoWave(GeoWaveArr[GeoWaveID].boubdingBox, training_label[0].Count());
            GeoWave child1 = new GeoWave(GeoWaveArr[GeoWaveID].boubdingBox, training_label[0].Count());

            //set partition
            child0.boubdingBox[1][dimIndex] = Maingridindex;
            child1.boubdingBox[0][dimIndex] = Maingridindex;

            //DOCUMENT ON CHILDREN
            child0.dimIndex = dimIndex;
            child0.Maingridindex = Maingridindex;
            child1.dimIndex = dimIndex;
            child1.Maingridindex = Maingridindex;
            child0.MaingridValue = wf.Program.MainGrid[dimIndex][Maingridindex];
            child1.MaingridValue = wf.Program.MainGrid[dimIndex][Maingridindex];

            //DOCUMENT ON PARENT
            GeoWaveArr[GeoWaveID].dimIndexSplitter = dimIndex;
            GeoWaveArr[GeoWaveID].splitValue = wf.Program.MainGrid[dimIndex][Maingridindex];

            //calc norm
            //calc mean value

            if (wf.Program.IsBoxSingular(child0.boubdingBox, training_dt[0].Count()) || wf.Program.IsBoxSingular(child1.boubdingBox, training_dt[0].Count()))
                return;

            //SHOULD I VERIFY THAT THE CHILD IS NOT ITS PARENT ? (IN CASES WHERE CAN'T MODEFY THE PARTITION)

            //setChildrensPointsAndMeanValue(ref child0, ref child1, dimIndex, GeoWaveArr[GeoWaveID].pointsIdArray);
            setChildrensPointsAndMeanValueUnIsotrpi(ref child0, ref child1,
                             hyperPlane, GeoWaveArr[GeoWaveID].pointsIdArray);
            GeoWaveArr[GeoWaveID].hyperPlane = hyperPlane;
            GeoWaveArr[GeoWaveID].PositionMean = this.m_MeanPositionForSplit_5;

            //SET TWO CHILDS
            child0.parentID = child1.parentID = GeoWaveID;
            child0.child0 = child1.child0 = -1;
            child0.child1 = child1.child1 = -1;
            child0.level = child1.level = GeoWaveArr[GeoWaveID].level + 1;

            child0.computeNormOfConsts(GeoWaveArr[GeoWaveID], Convert.ToDouble(userConfig.partitionType));
            child1.computeNormOfConsts(GeoWaveArr[GeoWaveID], Convert.ToDouble(userConfig.partitionType));
            GeoWaveArr.Add(child0);
            GeoWaveArr.Add(child1);
            GeoWaveArr[GeoWaveID].child0 = GeoWaveArr.Count - 2;
            GeoWaveArr[GeoWaveID].child1 = GeoWaveArr.Count - 1;
            child0.dimIndex = dimIndex;
            child1.dimIndex = dimIndex;
            //ASAFAb - check loop

            if (child0.pointsIdArray.Count() == 0 || child1.pointsIdArray.Count() == 0){
                double a = 0;
            }
            Console.WriteLine("COunt1 : {0} Count2: {1}", child0.pointsIdArray.Count(), child1.pointsIdArray.Count());
            //RECURSION STEP !!!
            recursiveBSP_WaveletsByConsts(GeoWaveArr, GeoWaveArr[GeoWaveID].child0, seed);
            recursiveBSP_WaveletsByConsts(GeoWaveArr, GeoWaveArr[GeoWaveID].child1, seed);
        }

        private bool getBestPartitionResult(ref int dimIndex, ref int Maingridindex, List<GeoWave> GeoWaveArr, int GeoWaveID, double Error, bool[] Dims2Take)
        {
            double[][] error_dim_partition = new double[2][];//error, Maingridindex
            error_dim_partition[0] = new double[training_dt[0].Count()];
            error_dim_partition[1] = new double[training_dt[0].Count()];

            //PARALLEL RUN - SEARCHING BEST PARTITION IN ALL DIMS
            if (userConfig.useParallel)
            {
                Parallel.For(0, training_dt[0].Count(), i =>
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
                for (int i = 0; i < training_dt[0].Count(); i++)
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

            Maingridindex = Convert.ToInt32(error_dim_partition[1][dimIndex]);
            return true;
        }

        private double[] getBestPartitionLargeDB(int dimIndex, GeoWave geoWave)
        {
            double[] error_n_point = new double[2];//error index
            if (wf.Program.MainGrid[dimIndex].Count == 1)//empty feature
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

                if (lowest_err > tmp_err && training_dt[tmpIDs[tmpIDs.Count - i - 1]][dimIndex] != training_dt[tmpIDs[tmpIDs.Count - i - 2]][dimIndex]
                    && (i + 1) >= userConfig.minNodeSize && (i + userConfig.minNodeSize) < tmpIDs.Count)
                {
                    best_ID = tmpIDs[tmpIDs.Count - i - 1];
                    lowest_err = tmp_err;
                }
            }

            if (best_ID == -1)
            {
                error_n_point[0] = double.MaxValue;
                error_n_point[1] = double.MaxValue;
                return error_n_point;
            }

            error_n_point[0] = Math.Max(lowest_err, 0);
            error_n_point[1] = training_GridIndex_dt[best_ID][dimIndex];

            return error_n_point;
        }

        private bool GetGiniPartitionResult(ref int dimIndex, ref int Maingridindex, List<GeoWave> GeoWaveArr, int GeoWaveID, double Error, bool[] Dims2Take)
        {
            double[][] error_dim_partition = new double[2][];//information gain, Maingridindex
            error_dim_partition[0] = new double[training_dt[0].Count()];
            error_dim_partition[1] = new double[training_dt[0].Count()];

            //PARALLEL RUN - SEARCHING BEST PARTITION IN ALL DIMS
            if (userConfig.useParallel)
            {
                Parallel.For(0, training_dt[0].Count(), i =>
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
                for (int i = 0; i < training_dt[0].Count(); i++)
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
            if (wf.Program.MainGrid[dimIndex].Count == 1)//empty feature
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

                if (gain > bestGain && training_dt[tmpIDs[tmpIDs.Count - i - 1]][dimIndex] != training_dt[tmpIDs[tmpIDs.Count - i - 2]][dimIndex]
                    && (i + 1) >= userConfig.minNodeSize && (i + userConfig.minNodeSize) < tmpIDs.Count 
                    )
                {
                    best_ID = tmpIDs[tmpIDs.Count - i - 1];
                    bestGain = gain;
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
                dimIndex = rnd.Next(0, training_dt[0].Count()); // creates a number between 0 and GeoWaveArr[0].rc.dim 
                int partition_ID = GeoWaveArr[GeoWaveID].pointsIdArray[rnd.Next(1, GeoWaveArr[GeoWaveID].pointsIdArray.Count() - 1)];

                Maingridindex = Convert.ToInt32(training_GridIndex_dt[partition_ID][dimIndex]);//this is dangerouse for Maingridindex > 2^32
                
                return true;
            }

            return false;
        }

        private void setChildrensPointsAndMeanValue(ref GeoWave child0, ref GeoWave child1, int dimIndex, List<int> indexArr)
        {
            for (int i = 0; i < child0.MeanValue.Count(); i++)
            {
                child0.MeanValue[i] *=0;
                child1.MeanValue[i] *= 0;
            }

            //GO OVER ALL POINTS IN REGION
            for (int i = 0; i < indexArr.Count; i++)
            {
                if (training_dt[indexArr[i]][dimIndex] < wf.Program.MainGrid[dimIndex].ElementAt(child0.boubdingBox[1][dimIndex]))
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

            for (int i = 0; i < child0.MeanValue.Count(); i++)
            {
                if (child0.pointsIdArray.Count > 0)
                    child0.MeanValue[i] /= Convert.ToDouble(child0.pointsIdArray.Count);
                if (child1.pointsIdArray.Count > 0)
                    child1.MeanValue[i] /= Convert.ToDouble(child1.pointsIdArray.Count);
            }
        }

        private bool[] getDim2Take( int Seed)
        {
            bool[] Dim2Take = new bool[training_dt[0].Count()];

            var ran = new Random(Seed);
            for (int i = 0; i < userConfig.nFeatures; i++)
            {
                //Dim2Take[dimArr[i]] = true;
                int index = ran.Next(0, training_dt[0].Count());
                if (Dim2Take[index] == true)
                    i--;
                else
                    Dim2Take[index] = true;
            }
            return Dim2Take;
        }


        //ASAFAB - do the actull partition
        private void setChildrensPointsAndMeanValueUnIsotrpi(ref GeoWave child0, ref GeoWave child1,
            double[] hyperPlane, List<int> indexArr)
        {
            //ASAFAB - TODO check
            MultipleDoubleArray(ref child0.MeanValue, 0);
            MultipleDoubleArray(ref child1.MeanValue, 0);
            //child0.MeanValue.Multiply(0);
            //child1.MeanValue.Multiply(0);

            double[] mean0 = new double[training_label[0].Count()];
            double[] mean1 = new double[training_label[0].Count()];

            //GO OVER ALL POINTS IN REGION
            for (int i = 0; i < indexArr.Count; i++)
            {
                double score = 0;
                for (int positionSizeIndex = 0; positionSizeIndex < this.m_dimenstion; positionSizeIndex++)
                {
                    score += hyperPlane[positionSizeIndex] *
                        (training_dt[indexArr[i]][positionSizeIndex] - this.m_MeanPositionForSplit_5[positionSizeIndex]);
                }
                score += hyperPlane[this.m_dimenstion];
                if (score > 0)
                {
                    child0.pointsIdArray.Add(indexArr[i]);
                    for (int valueIndex = 0; valueIndex < training_label[0].Count(); valueIndex++)
                    {
                        mean0[valueIndex] += training_label[indexArr[i]][valueIndex];
                    }
                }
                else
                {
                    child1.pointsIdArray.Add(indexArr[i]);
                    for (int valueIndex = 0; valueIndex < training_label[0].Count(); valueIndex++)
                    {
                        mean1[valueIndex] += training_label[indexArr[i]][valueIndex];
                    }
                }
            }

            if (child0.pointsIdArray.Count > 0)
            {
                MultipleDoubleArray(ref mean0, 1 / Convert.ToDouble(child0.pointsIdArray.Count));

                child0.MeanValue = mean0;
            }
            if (child1.pointsIdArray.Count > 0)
            {
                MultipleDoubleArray(ref mean1, 1 / Convert.ToDouble(child1.pointsIdArray.Count));

                child1.MeanValue = mean1;
            }
            int num0 = child0.pointsIdArray.Count;
            int num1 = child1.pointsIdArray.Count;

        }

        public static int[] UseKMeans2(double[][] observations)
        {
            // Create a new K-Means algorithm
            //KMeans kmeans = new KMeans(k: 2);
            int[] clustering = KMeansDemo.Cluster(observations, 2); // this is it

            //int[] labels = kmeans.Compute(observations, 0.1);
            return clustering;
        }


        //ASAFABAS - organize data befoe optimizers & SVM (caluculate mean & substract by it)
        public double[][][] OrganizeData(List<GeoWave> GeoWaveArr, int GeoWaveID, bool[] Dim2TakeNode, int dimNumber)
        {
            List<int> dataIDInGwW = new List<int>(GeoWaveArr[GeoWaveID].pointsIdArray); // The index of the relevent data
            double[] meanPosition = new double[this.m_dimenstion]; 
            double[][] centeredTrainingData = new double[dataIDInGwW.Count()][];


            // Calculate mmean position of the relevant data (node data).
            for (int indexTmp = 0; indexTmp < dataIDInGwW.Count(); indexTmp++)
            {
                for (int j = 0; j < this.m_dimenstion; j++)
                {
                    meanPosition[j] += training_dt[dataIDInGwW[indexTmp]][j];
                }
            }

            //meanPosition = meanPosition.Divide((double)dataIDInGwW.Count());
            MultipleDoubleArray(ref meanPosition, 1 / (double)dataIDInGwW.Count());

            // combine positions and labels
            double[][][] dataForOptimizer = new double[2][][];
            dataForOptimizer[0] = new double[dataIDInGwW.Count()][];
            dataForOptimizer[1] = new double[dataIDInGwW.Count()][];
            // move the fetuare of the data to mean 
            for (int indexTmp = 0; indexTmp < dataIDInGwW.Count(); indexTmp++)
            {
                dataForOptimizer[0][indexTmp] = new double[dimNumber + 1];
                centeredTrainingData[indexTmp] = new double[dimNumber + 1];
                centeredTrainingData[indexTmp][dimNumber] = 1; // Placing 1 in the last index  -> transfet the point to higher dimnsion
                int dimCounter = 0;
                for (int j = 0; j < training_dt[0].Count(); j++)
                {
                    if (Dim2TakeNode[j])
                    {
                        centeredTrainingData[indexTmp][dimCounter] = training_dt[dataIDInGwW[indexTmp]][j] - meanPosition[j];
                        dataForOptimizer[0][indexTmp][dimCounter] = centeredTrainingData[indexTmp][dimCounter];
                        dimCounter += 1;
                    }
                }
                dataForOptimizer[0][indexTmp][dimNumber] = 1;

                dataForOptimizer[1][indexTmp] = new double[training_label[0].Count()];
                for (int indexLabelTmp = 0; indexLabelTmp < training_label[0].Count(); indexLabelTmp++)
                {
                    dataForOptimizer[1][indexTmp][indexLabelTmp] = training_label[dataIDInGwW[indexTmp]][indexLabelTmp];
                }
            }

            this.m_MeanPositionForSplit_5 = meanPosition;


            return dataForOptimizer;
        }
        #region SVM
        
        public bool GetUnisotropicParitionUsingSVM(List<GeoWave> GeoWaveArr,
            int GeoWaveID, double Error, out double[] hyperPlane, bool[] Dim2TakeNode)
        {

            int dimNumber = 0;
            for(int  i = 0; i < Dim2TakeNode.Count(); i ++)
            {
                dimNumber += Dim2TakeNode[i] ? 1 : 0;
            }
            double[][][] dataForOptimizer = OrganizeData(GeoWaveArr, GeoWaveID, Dim2TakeNode, dimNumber);

            //2Means
            int[] clusters = UseKMeans2(dataForOptimizer[1]);
            for (int passingIndex = 0; passingIndex < dataForOptimizer[1].Count(); passingIndex++)
            {
                double tmpValue = ((double)clusters[passingIndex] == 0) ? -1 : 1;
                dataForOptimizer[1][passingIndex] = new double[1] { tmpValue };
            }

            Random rnd = new Random();

            double[] strtingState = new double[dimNumber + 1];
            double[] endState = new double[dimNumber + 1];
            strtingState[rnd.Next(0, dimNumber - 1)] = 1;


            
            //try 2
            var func = new Optimization2MeansSVMFunctiton(dataForOptimizer);
            var opt = new LibOptimization.Optimization.clsOptNelderMead(func);

            opt.InitialPosition = strtingState;

                //Init
                opt.Init();
               // clsUtil.DebugValue(opt);
                //do optimization!
                int size = 500;
            bool flag = opt.DoIteration(size);

            //clsUtil.DebugValue(opt);
            double eval1 = opt.Result.Eval;
            double tmp = 0;
                while (false)
                {
                    size += 200;
                    opt.InitialPosition = strtingState;

                    //Init
                    opt.Init();
                    flag = opt.DoIteration(size);
                    tmp = opt.Result.Eval;
                    if (Math.Abs(tmp - eval1) < 0.4)
                    {
                        break;
                    }
                    eval1 = tmp;
            }


    endState = opt.Result.ToArray();
            hyperPlane = new double[this.m_dimenstion + 1];
            int counterDim = 0;
            for(int i = 0; i<Dim2TakeNode.Count(); i++)
            {
                if (Dim2TakeNode[i])
                {
                    hyperPlane[i] = endState[counterDim];
                    counterDim += 1;
                }
            }
            hyperPlane[m_dimenstion] = endState[dimNumber];
            //hyperPlane = endState;
            return true;


        }
       
        
        public static void SVMOptimizer(double[] x, ref double func, object obj)
        {
            // Cast bsck to tree
            double[][][] dataForOptimizer = (double[][][])obj;

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
            double lambda = 0.01;
            double norm = 0;
            for (int positionIndex = 0; positionIndex < x.Count() - 1; positionIndex++)
            {
                norm += x[positionIndex] * x[positionIndex];
            }
            norm = Math.Sqrt(norm);

            error /= dataForOptimizer[0].Count() + norm * lambda;

            if (valuesChild1.Count() == 0 || valuesChild0.Count() == 0)
            {
                Random rnd = new Random();

                error += rnd.Next(1000, 1200);
            }
            // List<int> originalIndexList = tree;
            //for (int tmpIndex = 0; t)
            func = error;

        }
        #endregion

        //ASAFABAS - add functanlity of mulipication on double[]
        public static void MultipleDoubleArray(ref double[] array, double mult)
        {
            for (int index = 0; index < array.Count(); index ++)
            {
                array[index] *= mult;
            }
        }

        // ASAFAB - first try to find optimal partiiton
        private bool GetUnisotropicParition(List<GeoWave> GeoWaveArr,
            int GeoWaveID, double Error, out double[] hyperPlane, bool[] Dim2TakeNode)
        {
            int dimNumber = 0;
            for(int  i = 0; i < Dim2TakeNode.Count(); i ++)
            {
                dimNumber += Dim2TakeNode[i] ? 1 : 0;
            }
            double[][][] dataForOptimizer = OrganizeData(GeoWaveArr, GeoWaveID, Dim2TakeNode, dimNumber);


            Random rnd = new Random();
            // ASAFAB - Initialize the optimizer
            double epsg = 0.000010100000;
            double epsf = 0.00000000000;
            double epsx = 0.000000;
            double diffstep = 1;
            int maxits = 1000;
            double[] strtingState = new double[dimNumber + 1];
            double[] endState = new double[dimNumber + 1];
            strtingState[rnd.Next(0, dimNumber - 1)] = 1;
            //   strtingState = strtingState.Add(0.1);
           // strtingState[m_dimenstion] = 0;

            // Optimizer 1
            /*
            alglib.minlbfgsstate state;
            alglib.minlbfgsreport rep;
            alglib.minlbfgscreatef(1, strtingState, diffstep, out state);
            alglib.minlbfgssetcond(state, epsg, epsf, epsx, maxits);
            //alglib.minlbfgsoptimize(state, function1_funcOptimizer1, null, (object)dataForOptimizer);
            //alglib.minlbfgsresults(state, out endState, out rep);
            */
            //try 2
            var func = new OptimizationNaiveFunction(dataForOptimizer);
            var opt = new LibOptimization.Optimization.clsOptNelderMead(func);

            opt.InitialPosition = strtingState;

            //Init
            opt.Init();
           // clsUtil.DebugValue(opt);
            //do optimization!
            int size = 500;
            bool flag = opt.DoIteration(size);

            //clsUtil.DebugValue(opt);
            double eval1 = opt.Result.Eval;
            double tmp = 0;
            while (false)
            {
                size += 200;
                opt.InitialPosition = strtingState;

                //Init
                opt.Init();
                flag = opt.DoIteration(size);
                tmp = opt.Result.Eval;
                if (Math.Abs(tmp - eval1) < 0.4)
                {
                    break;
                }
                eval1 = tmp;
            }


            endState = opt.Result.ToArray();
            hyperPlane = new double[this.m_dimenstion + 1];
            int counterDim = 0;
            for(int i = 0; i < Dim2TakeNode.Count(); i++)
            {
                if (Dim2TakeNode[i])
                {
                    hyperPlane[i] = endState[counterDim];
                    counterDim += 1;
                }
            }
            hyperPlane[m_dimenstion] = endState[dimNumber];
            //hyperPlane = endState;
            return true;


        }

        //ASAFAB - this is the function we want to minimize
        public static void function1_funcOptimizer1(double[] x, ref double func, object obj)
        {
            // Cast bsck to tree
            double[][][] dataForOptimizer = (double[][][])obj;

            // clculate the mean of each side
            double[] mean0 = new double[dataForOptimizer[1][0].Count()];
            double[] mean1 = new double[dataForOptimizer[1][0].Count()];

            double error0 = 0;
            double error1 = 0;

            List<double[]> valuesChild0 = new List<double[]>();
            List<double[]> valuesChild1 = new List<double[]>();

            for (int passingIndex = 0; passingIndex < dataForOptimizer[0].Count(); passingIndex++)
            {
                double score = 0;
                // compute score (up or down the hyperplane)
                for (int positionIndex = 0; positionIndex < dataForOptimizer[0][0].Count(); positionIndex++)
                {
                    score += x[positionIndex] * dataForOptimizer[0][passingIndex][positionIndex];
                }

                if (score > 0) // go to child 0 
                {
                    valuesChild0.Add(dataForOptimizer[1][passingIndex]);
                    for (int valueIndex = 0; valueIndex < dataForOptimizer[1][0].Count(); valueIndex++)
                    {
                        mean0[valueIndex] += dataForOptimizer[1][passingIndex][valueIndex];
                    }

                }
                else // go to child1
                {
                    valuesChild1.Add(dataForOptimizer[1][passingIndex]);
                    for (int valueIndex = 0; valueIndex < dataForOptimizer[1][0].Count(); valueIndex++)
                    {
                        mean1[valueIndex] += dataForOptimizer[1][passingIndex][valueIndex];
                    }
                }
            }
            // devide the mean (so ... it will be actully be mean)
            for (int valueIndex = 0; valueIndex < dataForOptimizer[1][0].Count(); valueIndex++)
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

            error0 = CalculateErrorOFChild(mean0, valuesChild0);
            error1 = CalculateErrorOFChild(mean1, valuesChild1);
            if (valuesChild1.Count() == 0 || valuesChild0.Count() == 0)
            {
                Random rnd = new Random();

                error1 += rnd.Next(100, 620);
            }
            // List<int> originalIndexList = tree;
            //for (int tmpIndex = 0; t)
            func = error0 + error1;

        }
        public static double CalculateErrorOFChild(double[] mean, List<double[]> valueChilds)
        {
            if (valueChilds.Count() == 0)
            {
                return 0;
            }
            double[] error = new double[valueChilds[0].Count()];
            //calulate the diffrance between mean and labels
            for (int passingIndex = 0; passingIndex < valueChilds.Count(); passingIndex++)
            {
                for (int valueIndex = 0; valueIndex < valueChilds[0].Count(); valueIndex++)
                {
                    error[valueIndex] = valueChilds[passingIndex][valueIndex] - mean[valueIndex];
                }
            }

            // calculate norm
            double normError = 0;
            for (int valueIndex = 0; valueIndex < valueChilds[0].Count(); valueIndex++)
            {
                normError += error[valueIndex] * error[valueIndex];
            }
            normError = Math.Sqrt(normError);
            return normError;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Accord.Math;
//using Amazon.DynamoDBv2;

// ReSharper disable CompareOfFloatsByEqualityOperator
// ReSharper disable RedundantAssignment

namespace DataScienceAnalysis
{
    public class DecicionTree
    {
        private readonly recordConfig _rc;
        private readonly double[][] _trainingDt;
        private readonly long[][] _trainingGridIndexDt;
        private readonly double[][] _trainingLabel;
        private readonly bool[] _dime2Take;
        public string debugAnalysisFolderName { set; get; }

        public enum SplitType { NotSplitted, MainAxes, Random, MainAxesRandDim, Gini, GiniRandDim, LocalPca,
        DiffMaps5Percent, DiffMaps1Percent, DiffMapsHalfPercent, Categorical};
        public DecicionTree(recordConfig rc, DB db)
        {
            _trainingDt = db.PCAtraining_dt;
            _trainingLabel = db.training_label;
            _trainingGridIndexDt = db.PCAtraining_GridIndex_dt;
            _rc = rc;
        }

        public DecicionTree(recordConfig rc, DB db, bool[] dime2Take)
        {
            _trainingDt = db.PCAtraining_dt;
            _trainingLabel = db.training_label;
            _trainingGridIndexDt = db.PCAtraining_GridIndex_dt;
            _rc = rc;
            _dime2Take = dime2Take;
        }
        public DecicionTree(recordConfig rc, double[][] trainingDt, double[][] trainingLabel)
        {
            _trainingDt = trainingDt;
            _trainingLabel = trainingLabel;
            _rc = rc;
        }

        public DecicionTree(recordConfig rc, double[][] trainingDt, double[][] trainingLabel, long[][] trainingGridIndexDt, bool[] dime2Take)
        {
            _trainingDt = trainingDt;
            _trainingLabel = trainingLabel;
            _rc = rc;
            _trainingGridIndexDt = trainingGridIndexDt;
            _dime2Take = dime2Take;
        }

        public List<GeoWave> getdecicionTree(List<int> trainingArr, int[][] boundingBox, int seed = -1)
        {
            //CREATE DECISION_GEOWAVEARR
            List<GeoWave> decision_GeoWaveArr = new List<GeoWave>();

            //SET ROOT WAVELETE
            GeoWave gwRoot = new GeoWave(_rc.dim, _rc.labelDim, _rc) {pointsIdArray = trainingArr};

        

            decision_GeoWaveArr.Add(gwRoot);
            decomposeWaveletsByConsts(decision_GeoWaveArr, seed);

            //consider next twofunctions ?????

            //SET ID
            for (int i = 0; i < decision_GeoWaveArr.Count; i++)
                decision_GeoWaveArr[i].ID = i;

            //get sorted list
            decision_GeoWaveArr = decision_GeoWaveArr.OrderByDescending(o => o.norm).ToList();

            return decision_GeoWaveArr;
        }

        public struct SplitProps
        {
            public SplitType type { get; set; }
            public int dimIndex { get; set; }
            public int splitId { get; set; }
            public double splitValue { get; set; }
            //id's sorted accorded to split transformation type
            public List<int> sortedIds { get; set; }
            public bool isPartitionOk { get; set; }
            public double error { get; set; }
        }
     

        public void decomposeWaveletsByConsts(List<GeoWave> geoWaveArr, int seed = -1)//SHOULD GET LIST WITH ROOT GEOWAVE
        {
            geoWaveArr[0].MeanValue = geoWaveArr[0].calc_MeanValue(_trainingLabel, geoWaveArr[0].pointsIdArray);
            geoWaveArr[0].computeNormOfConsts();
            geoWaveArr[0].level = 0;
            //single dim
            geoWaveArr[0].meanDiffFromParent = geoWaveArr[0].MeanValue[0];
            recursiveBSP_TransformedData(geoWaveArr, 0, new List<SplitType>() {SplitType.LocalPca,
                                                                              // SplitType.DiffMapsHalfPercent,
                                                                             //SplitType.DiffMaps1Percent,
                                                                             //SplitType.DiffMaps5Percent,
                                                                             //SplitType.Categorical,
                                                                              SplitType.MainAxes }); 
        }

      

        // Transformed data decomposition
        private void recursiveBSP_TransformedData(IList<GeoWave> geoWaveArr, int geoWaveId, List<SplitType> splitTypes )
        {
            GeoWave parentNode = geoWaveArr[geoWaveId];
            double error = parentNode.calc_MeanValueReturnError(_trainingLabel, parentNode.pointsIdArray);
            if (error < _rc.approxThresh ||
                parentNode.pointsIdArray.Count() <= _rc.minWaveSize ||
                _rc.boundDepthTree <= parentNode.level)
                return;
           
           
            List<SplitProps> resultSplitsProperties = (from splitType in splitTypes 
                                                       select getTransformedPartitionAllDim(parentNode, error, splitType)).ToList();

            resultSplitsProperties = resultSplitsProperties.Where(x => x.isPartitionOk).ToList();
            //not exist split that may help
            if (resultSplitsProperties.Count == 0)  return;
            SplitProps bestSplit = resultSplitsProperties.Aggregate((a, b) => (a.error < b.error) ? a : b);

           
            if (!bestSplit.isPartitionOk) return;
            parentNode.typeTransformed = bestSplit.type;
            GeoWave child0 = new GeoWave(_rc.dim, _rc.labelDim, _rc);
            GeoWave child1 = new GeoWave(_rc.dim, _rc.labelDim, _rc);
            child0.dimIndex = bestSplit.dimIndex;
            child1.dimIndex = bestSplit.dimIndex;
            List<int> sortedIds = bestSplit.sortedIds;
            int splitId = bestSplit.splitId;
            //set childs id's
            child0.pointsIdArray = sortedIds.GetRange(0, sortedIds.IndexOf(splitId));
            child1.pointsIdArray = sortedIds.GetRange(sortedIds.IndexOf(splitId), sortedIds.Count - child0.pointsIdArray.Count);
            // set upper split value only
            child0.upperSplitValue = bestSplit.splitValue;
            //set mean values
            setTransformedChildMeanValue(ref child0);
            setTransformedChildMeanValue(ref child1);
            //set parent id
            child0.parentID = geoWaveId;
            child1.parentID = geoWaveId;
            //set level
            child0.level = parentNode.level + 1;
            child1.level = parentNode.level + 1;
            //debug writelines
            Debug.WriteLine("************Parent Size:"+parentNode.pointsIdArray.Count);
            Debug.WriteLine("************Level:" + (parentNode.level + 1));
            Debug.WriteLine("************Type Splitted:" + bestSplit.type);
            Debug.WriteLine("***********************************************************");

            //!!! START DEBUG VISULIZE SPIRAL SPLIT 
   /*         double[][] child0Data = child0.pointsIdArray.Select(id => _trainingDt[id]).ToArray();
            double[][] child1Data = child1.pointsIdArray.Select(id => _trainingDt[id]).ToArray();
            double[][] child0responce = child0.pointsIdArray.Select(id => _trainingLabel[id]).ToArray();
            double[][] child1responce = child1.pointsIdArray.Select(id => _trainingLabel[id]).ToArray();
            int level = child0.level;
            PrintEngine.debugVisualizeSpiralSplit(child0Data, child1Data,
                child0responce, child1responce,
                level, parentNode.typeTransformed, debugAnalysisFolderName);*/
          
            //!!! END DEBUG VISUALIZE SPIRAL SPLIT'

            //compute norms
            child0.computeNormOfConsts(parentNode);
            child1.computeNormOfConsts(parentNode);

            child0.meanDiffFromParent = child0.MeanValue[0] - parentNode.MeanValue[0];
            child1.meanDiffFromParent = child1.MeanValue[0] - parentNode.MeanValue[0];

            geoWaveArr.Add(child0);
            geoWaveArr.Add(child1);
            parentNode.child0 = geoWaveArr.IndexOf(child0);
            parentNode.child1 = geoWaveArr.IndexOf(child1);

   
            //RECURSION STEP !!!
            recursiveBSP_TransformedData(geoWaveArr, parentNode.child0, splitTypes);
            recursiveBSP_TransformedData(geoWaveArr, parentNode.child1, splitTypes);

        }
      
      /*  private void recursiveBSP_WaveletsByConsts(List<GeoWave> geoWaveArr, int geoWaveId, int seed=0)
        {
            //CALC APPROX_SOLUTION FOR GEO WAVE
            double error = geoWaveArr[geoWaveId].calc_MeanValueReturnError(_trainingLabel, geoWaveArr[geoWaveId].pointsIdArray);
            if (error < _rc.approxThresh ||
                geoWaveArr[geoWaveId].pointsIdArray.Count() <= _rc.minWaveSize ||
                _rc.boundDepthTree <=  geoWaveArr[geoWaveId].level)
            return;

            int dimIndex = -1;
            int Maingridindex = -1;

            bool IsPartitionOK = false;
            switch (_rc.split_type)
            {
                case 0:
                   IsPartitionOK = getBestPartitionResult(ref dimIndex, ref Maingridindex, geoWaveArr, geoWaveId, error, _dime2Take);
                    break;
                case 1:
                    IsPartitionOK = getRandPartitionResult(ref dimIndex, ref Maingridindex, geoWaveArr, geoWaveId, error, seed);
                    break;
                case 2:
                {
                    Random ran1 = new Random(seed);
                    Random ran2 = new Random(geoWaveId);
                    int one = ran1.Next(0, int.MaxValue / 10);
                    int two = ran2.Next(0, int.MaxValue / 10);
                    bool[] Dim2TakeNode = getDim2Take(_rc, one + two);
                    IsPartitionOK = getBestPartitionResult(ref dimIndex, ref Maingridindex, geoWaveArr, geoWaveId, error, Dim2TakeNode);
                }
                    break;
                case 3:
                    IsPartitionOK = getGiniPartitionResult(ref dimIndex, ref Maingridindex, geoWaveArr, geoWaveId, error, _dime2Take);
                    break;
                case 4:
                {
                    Random ran1 = new Random(seed);
                    Random ran2 = new Random(geoWaveId);
                    int one = ran1.Next(0, int.MaxValue / 10);
                    int two = ran2.Next(0, int.MaxValue / 10);
                    bool[] Dim2TakeNode = getDim2Take(_rc, one + two);
                    IsPartitionOK = getGiniPartitionResult(ref dimIndex, ref Maingridindex, geoWaveArr, geoWaveId, error, Dim2TakeNode);
                }
                    break;
              

            }


          

            if (!IsPartitionOK)
                return;


            GeoWave child0 = new GeoWave(geoWaveArr[geoWaveId].boubdingBox, _trainingLabel[0].Count(), geoWaveArr[geoWaveId].rc);
            GeoWave child1 = new GeoWave(geoWaveArr[geoWaveId].boubdingBox, _trainingLabel[0].Count(), geoWaveArr[geoWaveId].rc);

            //set partition
            child0.boubdingBox[1][dimIndex] = Maingridindex;
            child1.boubdingBox[0][dimIndex] = Maingridindex;

            //DOCUMENT ON CHILDREN
            child0.dimIndex = dimIndex;
            child0.Maingridindex = Maingridindex;
            child1.dimIndex = dimIndex;
            child1.Maingridindex = Maingridindex;

            child0.MaingridValue = Form1.MainGrid[dimIndex][Maingridindex];
            child1.MaingridValue = Form1.MainGrid[dimIndex][Maingridindex];

            //calc norm
            //calc mean value

            if (Form1.isBoxSingular(child0.boubdingBox, _rc.dim) || Form1.isBoxSingular(child1.boubdingBox, _rc.dim))
                return;

            //SHOULD I VERIFY THAT THE CHILD IS NOT ITS PARENT ? (IN CASES WHERE CAN'T MODEFY THE PARTITION)

            setChildrensPointsAndMeanValue(ref child0, ref child1, dimIndex, geoWaveArr[geoWaveId].pointsIdArray);
            //SET TWO CHILDS
            child0.parentID = child1.parentID = geoWaveId;
            child0.child0 = child1.child0 = -1;
            child0.child1 = child1.child1 = -1;
            child0.level = child1.level = geoWaveArr[geoWaveId].level + 1;

            child0.computeNormOfConsts(geoWaveArr[geoWaveId]);
            child1.computeNormOfConsts(geoWaveArr[geoWaveId]);
            geoWaveArr.Add(child0);
            geoWaveArr.Add(child1);
            geoWaveArr[geoWaveId].child0 = geoWaveArr.Count - 2;
            geoWaveArr[geoWaveId].child1 = geoWaveArr.Count - 1;


          

            //RECURSION STEP !!!
            recursiveBSP_WaveletsByConsts(geoWaveArr, geoWaveArr[geoWaveId].child0, seed);
            recursiveBSP_WaveletsByConsts(geoWaveArr, geoWaveArr[geoWaveId].child1, seed);
        }
*/
        private SplitProps getTransformedPartitionAllDim( GeoWave parentNode, double error, SplitType splitType)
        {
            double[][] originalNodeData = parentNode.pointsIdArray.Select(id => _trainingDt[id]).ToArray();
            double[][] transformedData;
            //clean columns of categorical variables 2m0rr0w2
           // originalNodeData = Helpers.copyAndRemoveCategoricalColumns(originalNodeData, _rc);
            //result struct
            SplitProps resultProps = new SplitProps();
            switch (splitType)
            {
                    case SplitType.LocalPca:
                        DimReduction.constructNodePcaByOriginalData(originalNodeData, parentNode);
                        transformedData = parentNode.localPca.Transform(originalNodeData);
                        break;
                    case SplitType.DiffMaps5Percent:
                        if (originalNodeData.Count() <= _rc.dim)
                        {
                             resultProps.isPartitionOk = false;
                             return resultProps;
                        }
                        transformedData = DiffusionMaps.getTransformedMatrix(originalNodeData, 0.05);
                        break;
                    case SplitType.DiffMaps1Percent:
                        if (originalNodeData.Count() <= _rc.dim)
                        {
                            resultProps.isPartitionOk = false;
                            return resultProps;
                        }
                        transformedData = DiffusionMaps.getTransformedMatrix(originalNodeData, 0.01);
                        break;
                    case SplitType.DiffMapsHalfPercent:
                        if (originalNodeData.Count() <= _rc.dim)
                        {
                            resultProps.isPartitionOk = false;
                            return resultProps;
                        }
                        transformedData = DiffusionMaps.getTransformedMatrix(originalNodeData, 0.005);
                        break;
                    case SplitType.MainAxes:
                        transformedData = originalNodeData;
                        break;
                    case SplitType.Categorical:
                        transformedData = originalNodeData;
                        break;
                    default:
                        transformedData = null;
                        break;
            }

            if (transformedData == null)
            {
                //throw new Exception("******TRANSFORMATION ERROR!!!");
                resultProps.isPartitionOk = false;
                Debug.WriteLine("*********Failed transformation");
                Debug.WriteLine("*********Failed node size: "+parentNode.pointsIdArray.Count);
                return resultProps;
            }
            parentNode.transformedDim = transformedData.First().Length;
            //save dim of transformed data
            int transformedDim = parentNode.transformedDim;
            double[] errorEachDim = new double[transformedDim];
            int[] partitionIdEachDim = new int[transformedDim];
            // _rc.dim replaced by transformedDim dimention
            Helpers.applyFor(0, transformedDim, dim =>
            {

                errorEachDim[dim] = getTransformedDataPartitionSingleDim(dim, transformedData, parentNode, partitionIdEachDim);
            });
            int bestDim = Enumerable.Range(0, transformedDim)
               .Aggregate((a, b) => (errorEachDim[a] < errorEachDim[b]) ? a : b);
            resultProps.splitId = partitionIdEachDim[bestDim];
            //save id's order in transformed data at best dimention
            resultProps.sortedIds = new List<int>(parentNode.pointsIdArray); // will be sorted at best split dimention
            List<int> idsClone = new List<int>(resultProps.sortedIds); // id's in original position
            resultProps.sortedIds.Sort((c1, c2) =>
              transformedData[idsClone.IndexOf(c1)][bestDim].CompareTo(transformedData[idsClone.IndexOf(c2)][bestDim]));
            //save partition value  
            int originalSplitLocation = idsClone.IndexOf(resultProps.splitId);
            
            if (originalSplitLocation == -1)
            {
                resultProps.isPartitionOk = false;
                return resultProps;
            }
            resultProps.isPartitionOk = (errorEachDim[bestDim] < error);
            resultProps.splitValue =  transformedData[originalSplitLocation][bestDim];
            resultProps.error = errorEachDim[bestDim];
            resultProps.type = splitType;
            resultProps.dimIndex = bestDim;
            //shift dimention if it was not categorical split 2m0rr0w2
        /*    foreach (int categoricalInd in _rc.indOfCategorical)
            {
                resultProps.dimIndex = (resultProps.dimIndex == categoricalInd)
                    ? resultProps.dimIndex++
                    : resultProps.dimIndex;
            }*/
            return resultProps;
        }

        private double getCategoricalPartitionSingleDim(int dimIndex, double[][] transformedData,
            GeoWave geoWave, IList<int> partId)
        {
            List<int> tmpIDs = new List<int>(geoWave.pointsIdArray);
            int amountCategories = tmpIDs.Select(id => transformedData[id][dimIndex]).Distinct().Count();
            if (amountCategories == 1)//all values are the same 
            {
                return double.MaxValue;
            }
            double[] errorsByCategory = new double[amountCategories];
            int tmpInd = 0;
            foreach (double catValue in tmpIDs.Select(id=>transformedData[id][dimIndex]).Distinct())
            {
                List<int> lstCurrCat = tmpIDs.Where(id => transformedData[id][dimIndex] == catValue).ToList();
                List<int> lstOtherCat = tmpIDs.Where(id => transformedData[id][dimIndex] != catValue).ToList();
                double currAvg = lstCurrCat.Average();
                double otherAvg = lstOtherCat.Average();
                double currCatErro = lstCurrCat.Sum(id => (_trainingLabel[tmpIDs[id]][0] - currAvg) * (_trainingLabel[tmpIDs[id]][0] - currAvg));
                double otherCatErro = lstOtherCat.Sum(id => (_trainingLabel[tmpIDs[id]][0] - otherAvg) * (_trainingLabel[tmpIDs[id]][0] - otherAvg));
                errorsByCategory[tmpInd] = currCatErro/lstCurrCat.Count() +  otherCatErro/lstOtherCat.Count();
                tmpInd++;
            }
            return 0;
            // List<int> idsClone = tmpIDs.Where(id => )
        }
      
        private double getTransformedDataPartitionSingleDim(int dimIndex, double[][] transformedData,
                                            GeoWave geoWave, IList<int> partId)
        {
         
            List<int> tmpIDs = new List<int>(geoWave.pointsIdArray); // will be sorted
            List<int> idsClone = new List<int>(tmpIDs); // id's in original position
            tmpIDs.Sort((c1, c2) =>
                transformedData[idsClone.IndexOf(c1)][dimIndex].CompareTo(transformedData[idsClone.IndexOf(c2)][dimIndex]));
            double smalestDimValue = transformedData[idsClone.IndexOf(tmpIDs.First())][dimIndex];
            double biggestDimValue = transformedData[idsClone.IndexOf(tmpIDs.Last())][dimIndex];

            if (smalestDimValue == biggestDimValue)//all values are the same 
            {
                return double.MaxValue;
            }

            // TO DO: Check suitability with Oren's calculation
            int best_ID = -1;
            double lowest_err = double.MaxValue;
            double[] leftAvg = new double[geoWave.MeanValue.Count()];
            double[] rightAvg = new double[geoWave.MeanValue.Count()];
            double[] leftErr = geoWave.calc_MeanValueReturnError(_trainingLabel, geoWave.pointsIdArray, ref leftAvg);//CONTAINES ALL POINTS - AT THE BEGINING
            double[] rightErr = new double[geoWave.MeanValue.Count()];


            double N_points = Convert.ToDouble(tmpIDs.Count);

            int last = tmpIDs.Count - 1; //we dont calc the last (rightmost) boundary - it equal to the left most
            for (int i = 0; i < last; i++)
            {
                double tmp_err = 0;
                for (int j = 0; j < _rc.labelDim; j++)
                {
                    leftErr[j] = leftErr[j] - (N_points - i) * (_trainingLabel[tmpIDs[last - i]][j] - leftAvg[j]) * (_trainingLabel[tmpIDs[last - i]][j] - leftAvg[j]) / (N_points - i - 1);
                    leftAvg[j] = (N_points - i) * leftAvg[j] / (N_points - i - 1) - _trainingLabel[tmpIDs[last - i]][j] / (N_points - i - 1);
                    rightErr[j] = rightErr[j] + (_trainingLabel[tmpIDs[last- i]][j] - rightAvg[j]) * (_trainingLabel[tmpIDs[last- i]][j] - rightAvg[j]) * Convert.ToDouble(i) / Convert.ToDouble(i + 1);
                    rightAvg[j] = rightAvg[j] * Convert.ToDouble(i) / Convert.ToDouble(i + 1) + _trainingLabel[tmpIDs[last - i]][j] / Convert.ToDouble(i + 1);
                    tmp_err += leftErr[j] + rightErr[j];
                }
                //in case some points has the same values - we calc the avarage (relevant for splitting) only after all the points (with same values) had moved to the right
                //we don't alow "improving" the same split with two points with the same position (sort is not unique)
                if (!(lowest_err > tmp_err) ||
                    transformedData[idsClone.IndexOf(tmpIDs[last - i])][dimIndex] == transformedData[idsClone.IndexOf(tmpIDs[last - i - 1])][dimIndex] ||
                    (i + 1) < _rc.minWaveSize || (i + _rc.minWaveSize) >= N_points ||
                    Form1.trainNaTable.ContainsKey(new Tuple<int, int>(tmpIDs[last - i], dimIndex)))
                    continue;

                best_ID = tmpIDs[last-i];
                lowest_err = tmp_err;
            }
            partId[dimIndex] = best_ID;
            return Math.Max(lowest_err, 0); 
        }
       
     
        
        // ReSharper disable once RedundantAssignment
        private bool getBestPartitionResult(ref int dimIndex, ref int mainGridindex, List<GeoWave> geoWaveArr, int geoWaveId, double error, bool[] dims2Take)
        {
            double[][] error_dim_partition = new double[2][];//error, mainGridindex
            error_dim_partition[0] = new double[_rc.dim];
            error_dim_partition[1] = new double[_rc.dim];
            int[] partitionIdEachDim = new int[_rc.dim];
            GeoWave parent = geoWaveArr[geoWaveId];
            double[][] originalNodeData = parent.pointsIdArray.Select(id => _trainingDt[id]).ToArray();
            //PARALLEL RUN - SEARCHING BEST PARTITION IN ALL DIMS
            Helpers.applyFor(0, _rc.dim, i =>
            {
                //double[] tmpResult = getBestPartition(i, GeoWaveArr[GeoWaveID]);
                if (dims2Take[i])
                {
                    double transError = getTransformedDataPartitionSingleDim(i, originalNodeData, parent, partitionIdEachDim);
                    error_dim_partition[0][i] = transError;
                    if (partitionIdEachDim[i] != -1)
                        error_dim_partition[1][i] = _trainingGridIndexDt[partitionIdEachDim[i]][i];
                    else error_dim_partition[1][i] = Double.MaxValue;
                    

                }
                else
                {
                    error_dim_partition[0][i] = double.MaxValue;//error
                    error_dim_partition[1][i] = -1;//mainGridindex                    
                }
            });
          
            dimIndex = Enumerable.Range(0, error_dim_partition[0].Count())
                .Aggregate((a, b) => (error_dim_partition[0][a] < error_dim_partition[0][b]) ? a : b);

            if (error_dim_partition[0][dimIndex] >= error)
                return false;//if best partition doesn't help - return

            mainGridindex = Convert.ToInt32(error_dim_partition[1][dimIndex]);
            return true;
        }

        #region Gini partition
        // ReSharper disable once UnusedParameter.Local
        private bool getGiniPartitionResult(ref int dimIndex, ref int mainGridindex, List<GeoWave> geoWaveArr, int geoWaveId, double error, bool[] dims2Take)
        {
            double[][] error_dim_partition = new double[2][];//information gain, mainGridindex
            error_dim_partition[0] = new double[_rc.dim];
            error_dim_partition[1] = new double[_rc.dim];
            Helpers.applyFor(0, _rc.dim, i =>
            {
                if (dims2Take[i])
                {
                    double[] tmpResult = getGiniPartitionLargeDb(i, geoWaveArr[geoWaveId]);
                    error_dim_partition[0][i] = tmpResult[0];//information gain
                    error_dim_partition[1][i] = tmpResult[1];//mainGridindex                    
                }
                else
                {
                    error_dim_partition[0][i] = double.MinValue;//information gain
                    error_dim_partition[1][i] = -1;//mainGridindex                    
                }
            });


            dimIndex = Enumerable.Range(0, error_dim_partition[0].Count())
                .Aggregate((a, b) => (error_dim_partition[0][a] > error_dim_partition[0][b]) ? a : b); //maximal gain (>)

            if (error_dim_partition[0][dimIndex] <= 0)
                return false;//if best partition doesn't help - return

            mainGridindex = Convert.ToInt32(error_dim_partition[1][dimIndex]);
            return true;
        }

        private double[] getGiniPartitionLargeDb(int dimIndex, GeoWave geoWave)
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
            tmpIDs.Sort(delegate(int c1, int c2) { return _trainingDt[c1][dimIndex].CompareTo(_trainingDt[c2][dimIndex]); });

            if (_trainingDt[tmpIDs[0]][dimIndex] == _trainingDt[tmpIDs[tmpIDs.Count - 1]][dimIndex])//all values are the same 
            {
                error_n_point[0] = double.MinValue;//min gain
                error_n_point[1] = -1;
                return error_n_point;
            }

            Dictionary<double, double> leftcategories = new Dictionary<double, double>(); //double as counter to enable devision
            Dictionary<double, double> rightcategories = new Dictionary<double, double>(); //double as counter to enable devision
            for (int i = 0; i < tmpIDs.Count(); i++)
            {
                if (leftcategories.ContainsKey(_trainingLabel[tmpIDs[i]][0]))
                    leftcategories[_trainingLabel[tmpIDs[i]][0]] += 1;
                else
                    leftcategories.Add(_trainingLabel[tmpIDs[i]][0], 1);
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
                double rightMostLable = _trainingLabel[tmpIDs[tmpIDs.Count - i - 1]][0];

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
                if (gain > bestGain && _trainingDt[tmpIDs[tmpIDs.Count - i - 1]][dimIndex] != _trainingDt[tmpIDs[tmpIDs.Count - i - 2]][dimIndex]
                    && (i + 1) >= _rc.minWaveSize && (i + _rc.minWaveSize) < tmpIDs.Count 
                    && !Form1.trainNaTable.ContainsKey(new Tuple<int, int>(tmpIDs[tmpIDs.Count - i - 1], dimIndex)))
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
            error_n_point[1] = _trainingGridIndexDt[best_ID][dimIndex];

            return error_n_point;
        }

        private double calcGini(Dictionary<double, double> totalcategories, double npoints)
        {
            return totalcategories.Select((t, i) => (totalcategories.ElementAt(i).Value / npoints) * (1 - (totalcategories.ElementAt(i).Value / npoints))).Sum();
        }
        #endregion
        #region Random partition
        // ReSharper disable once UnusedParameter.Local
        private bool getRandPartitionResult(ref int dimIndex, ref int mainGridindex, List<GeoWave> geoWaveArr, int geoWaveId, double error, int seed=0)
        {
            Random rnd0 = new Random(seed);
            int seedIndex = rnd0.Next(0, Int16.MaxValue/2); 

            Random rnd = new Random(seedIndex + geoWaveId);

            int counter = 0;

            while(counter < 20)
            {
                counter++;
                dimIndex = rnd.Next(0, geoWaveArr[0].rc.dim); // creates a number between 0 and GeoWaveArr[0].rc.dim 
                int partition_ID = geoWaveArr[geoWaveId].pointsIdArray[rnd.Next(1, geoWaveArr[geoWaveId].pointsIdArray.Count() - 1)];

                mainGridindex = Convert.ToInt32(_trainingGridIndexDt[partition_ID][dimIndex]);//this is dangerouse for mainGridindex > 2^32
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


            //mainGridindex = Convert.ToInt32(training_GridIndex_dt[partition_ID][dimIndex]);//this is dangerouse for mainGridindex > 2^32

            return false;
        }

        #endregion


        private void setTransformedChildMeanValue(ref GeoWave child)
        {
            child.MeanValue.Multiply(0);
            foreach (int ID in child.pointsIdArray)
            {
                for (int j = 0; j < _rc.labelDim; j++)
                    child.MeanValue[j] += _trainingLabel[ID][j];
            }
            if (child.pointsIdArray.Count > 0)
                child.MeanValue = child.MeanValue.Divide(Convert.ToDouble(child.pointsIdArray.Count));
        }

        private void setChildrensPointsAndMeanValue(ref GeoWave child0, ref GeoWave child1, int dimIndex, List<int> indexArr)
        {
            child0.MeanValue.Multiply(0);
            child1.MeanValue.Multiply(0);

            //GO OVER ALL POINTS IN REGION
            for (int i = 0; i < indexArr.Count; i++)
            {
                if (_trainingDt[indexArr[i]][dimIndex] < Form1.MainGrid[dimIndex].ElementAt(child0.boubdingBox[1][dimIndex]))
                {
                    for (int j = 0; j < _trainingLabel[0].Count(); j++)
                        child0.MeanValue[j] += _trainingLabel[indexArr[i]][j];
                    child0.pointsIdArray.Add(indexArr[i]);
                }
                else
                {
                    for (int j = 0; j < _trainingLabel[0].Count(); j++)
                        child1.MeanValue[j] += _trainingLabel[indexArr[i]][j];
                    child1.pointsIdArray.Add(indexArr[i]);
                }
            }
            if(child0.pointsIdArray.Count > 0)
                child0.MeanValue = child0.MeanValue.Divide(Convert.ToDouble(child0.pointsIdArray.Count));
            if (child1.pointsIdArray.Count > 0)
                child1.MeanValue = child1.MeanValue.Divide(Convert.ToDouble(child1.pointsIdArray.Count));
        }

   


        private bool[] getDim2Take(recordConfig rc, int seed)
        {
            bool[] Dim2Take = new bool[rc.dim];

            var ran = new Random(seed);
            //List<int> dimArr = Enumerable.Range(0, rc.dim).OrderBy(x => ran.Next()).ToList().GetRange(0, rc.dim);
            //List<int> dimArr = Enumerable.Range(0, rc.dim).OrderBy(x => ran.Next()).ToList().GetRange(0, rc.dim);
            for (int i = 0; i < rc.NDimsinRF; i++)
            {
                //Dim2Take[dimArr[i]] = true;
                int index = ran.Next(0, rc.dim);
                if (Dim2Take[index])
                    i--;
                else
                    Dim2Take[index] = true;
            }
            return Dim2Take;
        }
    }
}

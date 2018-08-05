using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using System.Diagnostics;
using System.Globalization;
using Accord.Math;

//using Accord.MachineLearning;
//using Accord.MachineLearning.DecisionTrees;
//using Accord.Math;
//using Amazon.S3.IO;


namespace DataScienceAnalysis
{
    class Analizer
    {
        public readonly string _analysisFolderName;
        private List<List<double>> _mainGrid;
        private readonly DB _db;
        private readonly recordConfig _rc;
        public static double[][] modefied_Lables;
        public static double[][][] resultsByTree;
        public static double[] resultsByForest;

        public Analizer(string analysisFolderName, List<List<double>> mainGrid, DB db, recordConfig rc)
        {
            // TODO: Complete member initialization
            _analysisFolderName = analysisFolderName;
            _mainGrid = mainGrid;
            _rc = rc;

            _db = new DB();
            _db.PCAtraining_dt = (double[][]) db.PCAtraining_dt.Clone();
            _db.PCAtesting_dt = (double[][])db.PCAtesting_dt.Clone();
            _db.PCAvalidation_dt= (double[][])db.PCAvalidation_dt.Clone();
            _db.training_label= (double[][])db.training_label.Clone();
            _db.testing_label = (double[][])db.testing_label .Clone();
            _db.validation_label= (double[][])db.validation_label.Clone();
            _db.validation_dt = (double[][])db.validation_dt.Clone();
        }

        public void excludeFeatureFromDb(int indExclude)
        {
            for (int i = 0; i < _db.PCAtraining_dt.Count(); i++)
            {
                _db.PCAtraining_dt[i] = _db.PCAtraining_dt[i].Where((val,ind)=> ind!=indExclude).ToArray();
            }
            for (int i = 0; i < _db.PCAtesting_dt.Count(); i++)
            {
                _db.PCAtesting_dt[i] = _db.PCAtesting_dt[i].Where((val, ind) => ind != indExclude).ToArray();
            }
            for (int i = 0; i < _db.PCAvalidation_dt.Count(); i++)
            {
                _db.PCAvalidation_dt[i] = _db.PCAvalidation_dt[i].Where((val, ind) => ind != indExclude).ToArray();
            }
            _rc.dim--;
        }
       

        public void  analize(List<int> trainingArr, List<int> testingArr, int[][] boundingBox,
                                                                      ref int indOfWorse, ref double predResult)
        {
            //print data in R format at first cicle
            if (indOfWorse == -1)
            {
                double[][] trainRData = trainingArr.Select(id => _db.PCAtraining_dt[id]).ToArray();
                double[][] testRData = testingArr.Select(id => _db.PCAtesting_dt[id]).ToArray();
                double[] trainRLabel = trainingArr.Select(id => _db.training_label[id][0]).ToArray();
                double[] testRLabel = testingArr.Select(id => _db.testing_label[id][0]).ToArray();
                PrintEngine.printDataInRFormat(trainRData, testRData, trainRLabel, testRLabel, _rc, _analysisFolderName);
            }
            else
            {
                Debug.WriteLine("secon cicle");
            }
            
            #region one tree

            //CREATE DECISION TREES
            Stopwatch watch = Stopwatch.StartNew();

            //RAND DIM
            bool[] Dim2TakeOneTree = getDim2Take(_rc, 1);
            DecicionTree decTree = new DecicionTree(_rc, _db, Dim2TakeOneTree)
            {
                debugAnalysisFolderName = _analysisFolderName
            };

            //decicionTree decTree = new decicionTree(rc, db);
            List<GeoWave> decision_GeoWaveArr = decTree.getdecicionTree(trainingArr, boundingBox);
            watch.Stop();
            double[] toc_time = new double[1];
            toc_time[0] = watch.ElapsedMilliseconds;


            printErrorsOfTree(toc_time, _analysisFolderName + "\\time_to_generate_FullTree.txt");
         
            double[] nWaev = new double[1];
            nWaev[0] = decision_GeoWaveArr.Count;
            printErrorsOfTree(nWaev, _analysisFolderName + "\\NwaveletsInTree.txt");
            
            //test it
            List<GeoWave> final_GeoWaveArr = decision_GeoWaveArr.OrderByDescending(o => o.norm).ToList();//see if not sorted by norm already...

            //int arrSize = Convert.ToInt32(rc.test_error_size * final_GeoWaveArr.Count / rc.hopping_size);
            int testBegin = _rc.waveletsTestRange[0];
            //rc.test_error_size = percent of wavelets to estimate
            int estimateSizeArr = _rc.getEstimationArrSize(final_GeoWaveArr.Count);

            double[] errorTree = new double[estimateSizeArr];
            double[] decayOnTraining = new double[estimateSizeArr];
            double[] Nwavelets = new double[estimateSizeArr];

            if(Form1.u_config.estimateOneTreeCb == "1")
            {
                Helpers.applyFor(testBegin, estimateSizeArr, i =>
                {
                    double normTreshold = final_GeoWaveArr[i*_rc.hopping_size].norm;
                    errorTree[i] = testDecisionTree(testingArr, _db.PCAvalidation_dt, _db.validation_label,
                                decision_GeoWaveArr, normTreshold, _rc.NormLPType);
                    if (Form1.u_config.runOneTreeOnTtrainingCB == "1")
                        decayOnTraining[i] = testDecisionTree(trainingArr, _db.PCAtraining_dt,
                            _db.training_label, decision_GeoWaveArr, normTreshold, _rc.NormLPType);
                    Nwavelets[i] = i * _rc.hopping_size;
                });
                
               

                int minErr_index = Enumerable.Range(0, errorTree.Length).Aggregate((a, b) => (errorTree[a] < errorTree[b]) ? a : b); //minerror
                double lowest_Tree_error = testDecisionTree(testingArr, _db.PCAtesting_dt, _db.testing_label, decision_GeoWaveArr, final_GeoWaveArr[minErr_index * _rc.hopping_size].norm, _rc.NormLPType);
                printErrorsOfTree(lowest_Tree_error, minErr_index * _rc.hopping_size, _analysisFolderName + "\\bsp_tree_errors_by_wavelets_TestDB.txt");

                //PRINT ERRORS TO FILE...
                printErrorsOfTree(errorTree, Nwavelets, _analysisFolderName + "\\bsp_tree_errors_by_wavelets_ValidationDB.txt");
                if (Form1.u_config.runOneTreeOnTtrainingCB == "1")
                    printErrorsOfTree(decayOnTraining, Nwavelets, _analysisFolderName + "\\bsp_tree_errors_by_wavelets_trainingDB.txt");           
                        
            }
            
            #region prooning one tree            
            
            if(Form1.runProoning)
            {
                //TEST TREE WITH PROONING
                int topLevelBegin = _rc.pruningTestRange[0];
                int topLevel = _rc.waveletsTestRange[1] == 0 ? getTopLevel(decision_GeoWaveArr) : _rc.waveletsTestRange[1];

                //int topLevel = getTopLevel(decision_GeoWaveArr);
                double[] errorTreeProoning = new double[topLevel];
                double[] errorTreeProoningOnTraining = new double[topLevel];
                //double[] errorTreeProoningL1 = new double[topLevel];
                double[] NLevels = new double[topLevel];
                //double[] errorTreeProoningBER = new double[topLevel];

                Helpers.applyFor(topLevelBegin, topLevel, i =>
                {
                    errorTreeProoning[i] = testDecisionTreeWithProoning(testingArr, _db.PCAvalidation_dt, _db.validation_label, decision_GeoWaveArr, i + 1, _rc.NormLPType);
                    if (Form1.u_config.runOneTreeOnTtrainingCB == "1")
                        errorTreeProoningOnTraining[i] = testDecisionTreeWithProoning(_db.PCAtraining_dt, _db.training_label, decision_GeoWaveArr, i + 1, _rc.NormLPType);
                    //errorTreeProoningBER[i] = testDecisionTreeWithProoning(db.PCAtesting_dt, db.testing_label, decision_GeoWaveArr, i + 1, -2);
                    //errorTreeProoningL1[i] = testDecisionTreeWithProoning(db.PCAtesting_dt, db.testing_label, decision_GeoWaveArr, i + 1, 1);
                    NLevels[i] = i;// * rc.hopping_size;
                });
              


                int minErrPruning_index = Enumerable.Range(0, errorTreeProoning.Length).Aggregate((a, b) => (errorTreeProoning[a] < errorTreeProoning[b]) ? a : b); //minerror
                double lowest_TreePruning_error = testDecisionTree(testingArr,_db.PCAtesting_dt, _db.testing_label, decision_GeoWaveArr, minErrPruning_index + 1, _rc.NormLPType);
                printErrorsOfTree(lowest_TreePruning_error, minErrPruning_index, _analysisFolderName + "\\bsp_tree_errors_by_waveletsPruning_TestDB.txt");
                
                //PRINT ERRORS TO FILE...
                printErrorsOfTree(errorTreeProoning, NLevels, _analysisFolderName + "\\bsp_tree_errors_by_prooning_Validation.txt");
                if (Form1.u_config.runOneTreeOnTtrainingCB == "1")
                    printErrorsOfTree(errorTreeProoningOnTraining, NLevels, _analysisFolderName + "\\bsp_tree_errors_by_prooning_training.txt");                
                //printErrorsOfTree(errorTreeProoningBER, NLevels, analysisFolderName + "\\bsp_tree_errors_by_prooningBER.txt");
                //printErrorsOfTree(errorTreeProoningL1, NLevels, analysisFolderName + "\\bsp_tree_errors_by_prooningL1.txt");
             #endregion

            }

            #endregion

           #region RF tree 

            int tmp_N_rows = Convert.ToInt32(trainingArr.Count * _rc.rfBaggingPercent);
            //List<int>[] trainingArrRF_indecesList = new List<int>[tmp_N_rows];
            List<int>[] trainingArrRF_indecesList = new List<int>[_rc.rfNum];

            int debugReadyTreesCounter = 0;
            

            if (Form1.runRf)
            {
                //create RF
                List<GeoWave>[] RFdecTreeArr = new List<GeoWave>[_rc.rfNum];

                List<GeoWave>[] arr = RFdecTreeArr;
               //debug parralel 5
                for (int i = 0; i < _rc.rfNum; i ++)
                {
                    //DiffusionMaps.reCreateMatlabCom();
                    //int upper = Math.Min(s + 2, _rc.rfNum);
                    //Helpers.applyFor(s, upper, i =>
                    //{
                        List<int> trainingArrRF = Form1.u_config.BaggingWithRepCB == "1" ? baggingBreiman(trainingArr, i) :
                            bagging(trainingArr, _rc.rfBaggingPercent, i);

                        trainingArrRF_indecesList[i] = trainingArrRF;
                        bool[] Dim2Take = getDim2Take(_rc, i);
                        DecicionTree decTreeRF = new DecicionTree(_rc, _db, Dim2Take);
                        arr[i] = decTreeRF.getdecicionTree(trainingArrRF, boundingBox, i);
                        debugReadyTreesCounter++;
                        Debug.WriteLine("*******Tress alredy ready:" + debugReadyTreesCounter);
                   // });
                }
         
                

                PrintEngine.printAverageWavetsCount(RFdecTreeArr, _analysisFolderName);
                //sparse the forest to have max "1000" wavelets in each tree
                if (Form1.u_config.sparseRfCB == "1" && Form1.u_config.sparseRfTB != "")
                {
                    int NwaveletsTmp;
                    if(int.TryParse(Form1.u_config.sparseRfTB, out NwaveletsTmp))
                        RFdecTreeArr = getsparseRf(RFdecTreeArr, NwaveletsTmp);
                }

                PrintEngine.printForestProperties(RFdecTreeArr, _analysisFolderName);
              /*  if (_rc.split_type == 5)
                {
                    PrintEngine.printSplitByComponentHistogram(RFdecTreeArr, _analysisFolderName);
                    PrintEngine.printMainWithLocalPCAoriginalFeatureHist(RFdecTreeArr, _analysisFolderName);
                }*/
               
                //DEBUG 2m0rr0w2 F.I. used for PCA + MAIN splits only

                
                if (Form1.u_config.saveTressCB == "1")
                {
                    if (Directory.Exists(_analysisFolderName + "\\archive"))
                    {
                        Directory.CreateDirectory(_analysisFolderName + "\\archive");
                    }
                    for (int i = 0; i < RFdecTreeArr.Count(); i++)
                    {
                        PrintEngine.printWaveletsProperties(RFdecTreeArr[i], _analysisFolderName + "\\archive\\waveletsPropertiesTree_" + i.ToString() + ".txt");
                    }                
                }
    
        

                if (Form1.u_config.estimateRFwaveletsCB == "1")
                {

                    double[] normsOfTrees = new double[RFdecTreeArr.Count()];//[N trees][testingArr]
                    resultsByTree = new double[RFdecTreeArr.Count()][][];//[N trees][testingArr][value]
                    Helpers.applyFor(0, RFdecTreeArr.Count(), i =>
                    {
                        RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                        resultsByTree[i] = new double[testingArr.Count()][];
                        for (int j = 0; j < testingArr.Count(); j++)
                            resultsByTree[i][j] = new double[_rc.labelDim];
                        normsOfTrees[i] = -1;// ITS SET ON THE FLY
                    });

                   

                    //NO PRUNING
                    /*double[] errorRyByForest = testDecisionTreeManyRFnew(testingArr, _db.PCAvalidation_dt,
                        _db.validation_label, RFdecTreeArr, 0.0, _rc.NormLPType);*/

                    List<double> NwaveletsInRF = new List<double>();
                    for (int i = 0; i < RFdecTreeArr.Count(); i++)
                        NwaveletsInRF.Add(RFdecTreeArr[i].Count());

                   // PrintEngine.printList(errorRyByForest.ToList(), _analysisFolderName + "\\errorRyByForest.txt");
                    PrintEngine.printList(NwaveletsInRF, _analysisFolderName + "\\NwaveletsInRF.txt");

                    //WAVELETS ANALYSIS:
                    List<double[]> NormMultyArr = new List<double[]>();
                    for (int i = 0; i < RFdecTreeArr.Count(); i++)
                        for (int j = 0; j < RFdecTreeArr[i].Count; j++)
                        {
                            double[] pair = new double[3];
                            pair[0] = RFdecTreeArr[i][j].norm;
                            pair[1] = i; //tree
                            pair[2] = Double.Parse(((int)RFdecTreeArr[i][j].typeTransformed).ToString());  //node type
                            NormMultyArr.Add(pair);
                        }

                    NormMultyArr = NormMultyArr.OrderByDescending(t => t[0]).ToList();
                    int NerrorRyByWavelets = NormMultyArr.Count();
                    //double[] errorRyByWavelets = new double[NerrorRyByWavelets];
                    List<double> errorRyByWavelets = new List<double>();
                    //double[] errorRyByForest = new double[RFdecTreeArr.Count()];
                    //int N_treesInUse = 0;
                    //bool newTree = false;
                    List<double> NwaveletsInWaveletByWavelet = new List<double>();

                    //SET GLOBAL PARAMETER MODEFIED_LABLES (TO IMPROV EPERFORMANCE)
                    resultsByForest = new double[testingArr.Count()];//[N trees][testingArr]
                    modefied_Lables = new double[testingArr.Count()][];

                    for (int i = 0; i < testingArr.Count(); i++)
                        modefied_Lables[i] = new double[_db.validation_label[0].Count()];

                    //start eperiment of path creation
                    List<PointPaths> _testPointsPathes = createPathesByForest(RFdecTreeArr, testingArr, _db.PCAvalidation_dt);
                    for (int i = 0; i < NerrorRyByWavelets; i++)
                    {
                        if (i < 100)
                            i += 1;
                        else if ( i < 500)
                            i += 24;
                        else if (i < 1000)
                            i += 49;
                        else if (i < 5000)
                            i += 99;
                        else if (i < 10000)
                            i += 199;
                        else if (i < 100000)
                            i += 499;
                        else if (i < 200000)
                            i += 999;
                        else if (i < 300000)
                            i += 4999;
                        else
                            i += 10000; //first 100 wavelets

                        if (i >= NerrorRyByWavelets)
                            continue;
                        double nwErr = errorByPathAndNormTreshold(_testPointsPathes, RFdecTreeArr,
                                   _db.validation_label, NormMultyArr[i][0]);

                        //end test new method
                        errorRyByWavelets.Add(nwErr);
                       
                        NwaveletsInWaveletByWavelet.Add(i+1);
                        Debug.WriteLine("Estimated waveletets: " + i);

                    }


                    predResult = errorRyByWavelets.Min();
                    //PRINT
                    PrintEngine.printMainWithLocalPCAoriginalFeatureHist(RFdecTreeArr, _analysisFolderName, ref indOfWorse, predResult, _rc);
                    PrintEngine.printAllWaveletsNormsAndTypes(NormMultyArr, _analysisFolderName);
                    PrintEngine.printList(errorRyByWavelets, _analysisFolderName + "\\errorRyByWavelets.txt");
                    PrintEngine.printList(NwaveletsInWaveletByWavelet, _analysisFolderName + "\\NwaveletsInWaveletByWavelet.txt");
                         
                }

                if (Form1.runRFProoning)
                {
                    int topLevel = int.MaxValue;
                    for(int k=0; k < RFdecTreeArr.Count(); k++)
                    {
                        int tmp =getTopLevel(RFdecTreeArr[k]);
                        if (tmp < topLevel)
                            topLevel = tmp;
                    }

                    int topLevelBeginRF = _rc.RFpruningTestRange[0];
                    topLevel = _rc.RFpruningTestRange[1] == 0 ? topLevel : _rc.RFpruningTestRange[1];

                    double[] errorRFProoning = new double[topLevel];
                    double[] NwaveletsRFProoning = new double[topLevel];
                    double[][] errorManyRFProoning = new double[topLevel][];
                    double[][] errorManyRFProoningNoVoting = new double[topLevel][];
                    for (int i = 0; i < topLevel; i++)
                    {
                        errorManyRFProoning[i] = new double[RFdecTreeArr.Count()];
                        errorManyRFProoningNoVoting[i] = new double[RFdecTreeArr.Count()];
                    }
                    Helpers.applyFor(0, topLevel, i =>
                        {
                            //errorRFProoning[i] = testDecisionTreeRF(db.PCAvalidation_dt, db.validation_label, RFdecTreeArr, i + 1, 2);
                            errorManyRFProoning[i] = testDecisionTreeManyRF(testingArr,_db.PCAvalidation_dt, _db.validation_label, RFdecTreeArr, i + 1, _rc.NormLPType);
                            if (Form1.u_config.estimateRFnoVotingCB == "1")
                                errorManyRFProoningNoVoting[i] = testDecisionTreeManyRFNoVoting(testingArr, _db.PCAvalidation_dt, _db.validation_label, RFdecTreeArr, i + 1, _rc.NormLPType);
                            NwaveletsRFProoning[i] = i+1;//  / rc.rfNum - if we want to devide by the number of trees to get the degree 
                        });    

               

                    int minErrPruningRF_index = Enumerable.Range(0, errorRFProoning.Length).Aggregate((a, b) => (errorRFProoning[a] < errorRFProoning[b]) ? a : b); //minerror
                    double lowest_TreePruningRF_error = testDecisionTreeRF(testingArr,_db.PCAtesting_dt, _db.testing_label, RFdecTreeArr, minErrPruningRF_index + 1, 2);
                    printErrorsOfTree(lowest_TreePruningRF_error, minErrPruningRF_index * _rc.hopping_size, _analysisFolderName + "\\bsp_tree_errors_by_Pruning_RF_TestDB.txt");
                
                    //printErrorsOfTree(errorRFProoning, NwaveletsRFProoning, analysisFolderName + "\\RF_errors_by_Pruning.txt");
                    PrintEngine.printtable(errorManyRFProoning, _analysisFolderName + "\\cumulative_RF_errors_by_Pruning_validationDB.txt");

                    if (Form1.u_config.estimateRFnoVotingCB == "1")
                        PrintEngine.printtable(errorManyRFProoningNoVoting, _analysisFolderName + "\\independent_errors_of_rf_treees_no_voting_Pruning_validationDB.txt");                                                                             
                    //printErrorsOfTree(NwaveletsRFProoning, analysisFolderName + "\\Num_of_ManyRF_levels_validationDB.txt");              
                }
            }

            #endregion

            #region Boosting tree

            if (Form1.runBoosting)
            {
                //BOOST
                List<GeoWave>[] BoostTreeArr = new List<GeoWave>[_rc.boostNum];
                double[][] boostedLabels = new double[_db.training_label.Count()][];
                for (int i = 0; i < _db.training_label.Count(); i++)
                {
                    boostedLabels[i] = new double[_db.training_label[0].Count()];
                    for (int j = 0; j < _db.training_label[0].Count(); j++)
                        boostedLabels[i][j] = _db.training_label[i][j];
                }

                //
                bool[] Dim2Take = getDim2Take(_rc, 0);//should take all


                //Array.Copy(db.training_label, 0, boostedLabels, 0, db.training_label.Length); - bad copy - by reference
                double[] best_norms = new double[_rc.boostNum];
                int[] best_indeces = new int[_rc.boostNum];
                for (int i = 0; i < _rc.boostNum; i++)
                {
                    DecicionTree decTreeBoost = new DecicionTree(_rc, _db.PCAtraining_dt, boostedLabels, _db.PCAtraining_GridIndex_dt,Dim2Take);
                    if (i == 0 && decision_GeoWaveArr.Count > 0)
                        BoostTreeArr[i] = decision_GeoWaveArr;//take tree from first creation of "BSP" tree
                    else
                        BoostTreeArr[i] = decTreeBoost.getdecicionTree(trainingArr, boundingBox);

                    
                    //KFUNC
                    //best_indeces[i] = getGWIndexByKfunc(BoostTreeArr[i], rc, db.PCAtraining_dt, boostedLabels, ref best_norms[i]);
                    best_indeces[i] = getGwIndexByKfuncLessAcurate(BoostTreeArr[i], _rc, _db.PCAtraining_dt, boostedLabels, ref best_norms[i], testingArr);
                    best_norms[i] = BoostTreeArr[i][best_indeces[i]].norm;

                    boostedLabels = getResidualLabelsInBoosting(BoostTreeArr[i], _db.PCAtraining_dt, boostedLabels, best_norms[i]);
                    //Form1.printtable(boostedLabels, analysisFolderName + "\\BoostingLabels_" + i.ToString() + "_tree.txt");
                    _rc.boostlamda_0 = _rc.boostlamda_0 * 0.5;

                    //dbg
                    //Form1.printConstWavelets2File(BoostTreeArr[i], analysisFolderName + "\\BoostingdecTreeArr_" + i.ToString() + "_tree.txt");//dbg
                }

                double[] tmpArr = new double[BoostTreeArr.Count()];
                for (int i = 0; i < BoostTreeArr.Count(); i++)
                {
                    tmpArr[i] = Convert.ToDouble(best_indeces[i]);
                }
                printErrorsOfTree(tmpArr, _analysisFolderName + "\\num_wavelets_in_boosting.txt");
                printErrorsOfTree(best_norms, _analysisFolderName + "\\threshold_norms_of_wavelets_in_boosting.txt");

                //TEST IT
                List<double> NormArrBoosting = new List<double>();
                for (int i = 0; i < BoostTreeArr.Count(); i++)
                    for (int j = 0; j < BoostTreeArr[i].Count; j++)
                        if (BoostTreeArr[i][j].norm >= best_norms[i])
                            NormArrBoosting.Add(BoostTreeArr[i][j].norm);
                NormArrBoosting = NormArrBoosting.OrderByDescending(o => o).ToList();

                int arrSizeBoost = Convert.ToInt32(_rc.test_error_size * NormArrBoosting.Count / _rc.hopping_size);
                double[] errorBoosting = new double[arrSizeBoost];
                //double[] missLabelsBoosting = new double[arrSizeBoost];
                double[] missLabelsBoostingBER = new double[arrSizeBoost];
                double[] NwaveletsBoosting = new double[arrSizeBoost];

                Helpers.applyFor(0, arrSizeBoost, i =>
                {
                    errorBoosting[i] = testDecisionTreeBoosting(_db.PCAtesting_dt, _db.testing_label, BoostTreeArr, NormArrBoosting[i * _rc.hopping_size], 2, best_norms);
                    //missLabelsBoosting[i] = testDecisionTreeBoosting(db.PCAtesting_dt, db.testing_label, BoostTreeArr, NormArr[i * rc.hopping_size], 0, best_norms);
                    //missLabelsBoostingBER[i] = testDecisionTreeBoosting(db.PCAtesting_dt, db.testing_label, BoostTreeArr, NormArrBoosting[i * rc.hopping_size], -2, best_norms);
                    NwaveletsBoosting[i] = i * _rc.hopping_size;
                });

            

                printErrorsOfTree(errorBoosting, NwaveletsBoosting, _analysisFolderName + "\\Boosting_errors_by_wavelets.txt");
                //printErrorsOfTree(missLabelsBoostingBER, NwaveletsBoosting, analysisFolderName + "\\Boosting_BER_by_wavelets.txt");
                //printErrorsOfTree(missLabelsBoosting, NwaveletsBoosting, analysisFolderName + "\\Boosting_missLabe_by_wavelets.txt");                        
            }
            #endregion
            
            #region Prooning Boosting tree
            
            if (Form1.runBoostingProoning)
            {
                //BOOST
                List<GeoWave>[] BoostTreeArrPooning = new List<GeoWave>[_rc.boostNum];
                double[][] boostedLabelsPooning = new double[_db.training_label.Count()][];
                for (int i = 0; i < _db.training_label.Count(); i++)
                {
                    boostedLabelsPooning[i] = new double[_db.training_label[0].Count()];
                    for (int j = 0; j < _db.training_label[0].Count(); j++)
                        boostedLabelsPooning[i][j] = _db.training_label[i][j];
                }
                
                bool[] Dim2Take = getDim2Take(_rc, 0);//should take all

                //Array.Copy(db.training_label, 0, boostedLabels, 0, db.training_label.Length); - bad copy - by reference
                int[] best_level = new int[_rc.boostNum];
                int[] best_indecesProoning = new int[_rc.boostNum];
                for (int i = 0; i < _rc.boostNum; i++)
                {
                    DecicionTree decTreeBoost = new DecicionTree(_rc, _db.PCAtraining_dt, boostedLabelsPooning, _db.PCAtraining_GridIndex_dt, Dim2Take);
                    if (i == 0 && decision_GeoWaveArr.Count > 0)
                        BoostTreeArrPooning[i] = decision_GeoWaveArr;//take tree from first creation of "BSP" tree
                    else
                        BoostTreeArrPooning[i] = decTreeBoost.getdecicionTree(trainingArr, boundingBox);

                    //KFUNC
                    //best_indeces[i] = getGWIndexByKfunc(BoostTreeArr[i], rc, db.PCAtraining_dt, boostedLabels, ref best_norms[i]);
                    //best_level[i] = getGWIndexByKfuncLessAcuratePooning(BoostTreeArrPooning[i], rc, db.PCAtraining_dt, boostedLabelsPooning);

                    best_level[i] = Convert.ToInt32(_rc.boostProoning_0);

                    boostedLabelsPooning = getResidualLabelsInBoostingProoning(BoostTreeArrPooning[i], _db.PCAtraining_dt, boostedLabelsPooning, best_level[i]);

                    //dbg
                    //Form1.printtable(boostedLabelsPooning, analysisFolderName + "\\BoostingPruningLabels_" + i.ToString() + "_tree.txt");
                    //Form1.printConstWavelets2File(BoostTreeArrPooning[i], analysisFolderName + "\\Boosting_Prooning_decTreeArr_" + i.ToString() + "_tree.txt");//dbg
                }

                double[] tmpArr = new double[BoostTreeArrPooning.Count()];
                for (int i = 0; i < BoostTreeArrPooning.Count(); i++)
                {
                    tmpArr[i] = Convert.ToDouble(best_level[i]);
                }
                printErrorsOfTree(tmpArr, _analysisFolderName + "\\tree_levels_in_boosting.txt");

                //TEST IT
                double[] errorBoostingProoning = new double[_rc.boostNum];//error size in each boosting step

                testDecisionTreeBoostingProoning(_db.PCAtesting_dt, _db.testing_label, BoostTreeArrPooning, best_level, 2, errorBoostingProoning);
                printErrorsOfTree(errorBoostingProoning, tmpArr, _analysisFolderName + "\\Boosting_Prooning_errors_by_levels.txt");

                //double[] errorBoostingProoningBER = new double[rc.boostNum];//error size in each boosting step
                //testDecisionTreeBoostingProoning(db.PCAtesting_dt, db.testing_label, BoostTreeArrPooning, best_level, -2, errorBoostingProoningBER);
                //printErrorsOfTree(errorBoostingProoningBER, tmpArr, analysisFolderName + "\\Boosting_Prooning_BER_by_levels.txt");    
            }
            #endregion

            #region Boosting tree LearningRate

            if (Form1.runBoostingLearningRate)
            {
                //BOOST
                //need to modefy to work with testingArr in training and testing

                List<GeoWave>[] BoostTreeArrLearningRate = new List<GeoWave>[_rc.boostNumLearningRate];
                double[] BoostArrLearningRateNorms = new double[_rc.boostNumLearningRate];
                double[][] boostedLabelsLearningRate = new double[_db.training_label.Count()][];//trainingArr.Count
                for (int i = 0; i < _db.training_label.Count(); i++)//trainingArr.Count
                {
                    boostedLabelsLearningRate[i] = new double[_db.training_label[0].Count()];
                    for (int j = 0; j < _db.training_label[0].Count(); j++)
                        boostedLabelsLearningRate[i][j] = _db.training_label[i][j];//trainingArr[i][j]
                }

                bool[] Dim2Take = getDim2Take(_rc, 0);//should take all

                //Array.Copy(db.training_label, 0, boostedLabels, 0, db.training_label.Length); - bad copy - by reference
                int[] best_level = new int[_rc.boostNumLearningRate];
                int[] best_indecesProoning = new int[_rc.boostNumLearningRate];
                for (int i = 0; i < _rc.boostNumLearningRate; i++)
                {
                    DecicionTree decTreeBoost = new DecicionTree(_rc, _db.PCAtraining_dt, boostedLabelsLearningRate, _db.PCAtraining_GridIndex_dt, Dim2Take);
                    if (i == 0 && decision_GeoWaveArr.Count > 0)
                        BoostTreeArrLearningRate[i] = decision_GeoWaveArr;//take tree from first creation of "BSP" tree
                    else
                        BoostTreeArrLearningRate[i] = decTreeBoost.getdecicionTree(trainingArr, boundingBox);

                    //KFUNC
                    //best_indeces[i] = getGWIndexByKfunc(BoostTreeArr[i], rc, db.PCAtraining_dt, boostedLabels, ref best_norms[i]);
                    //best_level[i] = getGWIndexByKfuncLessAcuratePooning(BoostTreeArrPooning[i], rc, db.PCAtraining_dt, boostedLabelsPooning);

                    //best_level[i] = Convert.ToInt32(rc.boostProoning_0);

                    //List<GeoWave> tmp_GeoWaveArr = BoostTreeArrLearningRate[i].OrderByDescending(o => o.norm).ToList();//see if not sorted by norm already...
                    if (BoostTreeArrLearningRate[i].Count > _rc.NwaveletsBoosting)
                    {
                        BoostArrLearningRateNorms[i] = BoostTreeArrLearningRate[i][_rc.NwaveletsBoosting].norm;
                        boostedLabelsLearningRate = getResidualLabelsInBoosting(BoostTreeArrLearningRate[i], _db.PCAtraining_dt, boostedLabelsLearningRate, BoostArrLearningRateNorms[i]);
                    }
                    else
                    {
                        BoostArrLearningRateNorms[i] = BoostTreeArrLearningRate[i][BoostTreeArrLearningRate[i].Count - 1].norm;
                        boostedLabelsLearningRate = getResidualLabelsInBoosting(BoostTreeArrLearningRate[i], _db.PCAtraining_dt, boostedLabelsLearningRate, BoostArrLearningRateNorms[i]);                    
                    }


                    //dbg
                    //Form1.printtable(boostedLabelsLearningRate, analysisFolderName + "\\BoostingLearningRateLabels_" + i.ToString() + "_tree.txt");
                    //Form1.printConstWavelets2File(BoostTreeArrLearningRate[i], analysisFolderName + "\\Boosting_LearningRate_decTreeArr_" + i.ToString() + "_tree.txt");//dbg
                }

                double[] BoostTreeArrLearningRateErrors = new double[_rc.boostNumLearningRate];
                for (int i = 0; i < _rc.boostNumLearningRate; i++ )
                    BoostTreeArrLearningRateErrors[i] = testDecisionTreeBoostingLearningRate(testingArr, _db.PCAtesting_dt, _db.testing_label, BoostTreeArrLearningRate, 2, BoostArrLearningRateNorms, i + 1);

                printErrorsOfTree(BoostTreeArrLearningRateErrors, _analysisFolderName + "\\BoostTreeArrLearningRateError.txt");

                ////TEST IT
                //double[] errorBoostingProoning = new double[rc.boostNum];//error size in each boosting step

                //testDecisionTreeBoostingProoning(db.PCAtesting_dt, db.testing_label, BoostTreeArrLearningRate, best_level, 2, errorBoostingProoning);
                //printErrorsOfTree(errorBoostingProoning, tmpArr, analysisFolderName + "\\Boosting_LearningRate_errors_by_levels.txt");

                //double[] errorBoostingProoningBER = new double[rc.boostNum];//error size in each boosting step
                //testDecisionTreeBoostingProoning(db.PCAtesting_dt, db.testing_label, BoostTreeArrPooning, best_level, -2, errorBoostingProoningBER);
                //printErrorsOfTree(errorBoostingProoningBER, tmpArr, analysisFolderName + "\\Boosting_Prooning_BER_by_levels.txt");    
            }
            #endregion
            #region Exculede One feachure end save predict result 
            
            #endregion
        }

        private double calcResultByTree(List<int> testingArr, double[][] Data_table, double[][] Data_Lables,
                                                     List<GeoWave>[] RFdecTreeArrById, double[] normsOfTrees)
        {
            double[] errArr = new double[testingArr.Count()];

            Parallel.For(0, testingArr.Count(), i =>
            {
                //GO OVER ALL TREES
                double[] tmpVal = {0};
                Parallel.For(0, RFdecTreeArrById.Count(), j =>
                {
                    double[] val = askTreeMeanVal(Data_table[testingArr[i]], RFdecTreeArrById[j], normsOfTrees[j]);
                    tmpVal[0] += val[0];
                });
                tmpVal[0] /= RFdecTreeArrById.Count();
                errArr[i] = (tmpVal[0] - Data_Lables[testingArr[i]][0]) * (tmpVal[0] - Data_Lables[testingArr[i]][0]);
            });

            double errorL2 = 0;
            for (int i = 0; i < errArr.Count(); i++)
                errorL2 += errArr[i];
            errorL2 = Math.Sqrt(errorL2 / Convert.ToDouble(testingArr.Count()));
            return errorL2;
        }

        public static void preparenormsOfTrees(ref double[] normsOfTrees, int index2Start, IList<double[]> normMultyArr)
        {
            bool[] wasSet = new bool[normsOfTrees.Count()];
            int totalSet = 0;
            while (totalSet != normsOfTrees.Count())
            {
                int ID = (int)normMultyArr[index2Start][1];
                if (wasSet[ID])
                {
                    index2Start--;
                    continue;
                }

                normsOfTrees[ID] = normMultyArr[index2Start][0];
                wasSet[ID] = true;
                totalSet++;
                index2Start--;
            }
        }

        private double testerrorRyByWavelets(List<int> testingArr, double[][] Data_table, double[][] Data_Lables,
                                           List<GeoWave>[] RFdecTreeArrById, double[] normsOfTrees,
                                           ref int N_treesInUse, ref int indexeTreeChanged, bool newTree, int NormLPType)
        {
            //GO TO THE TREE THAT HAD BEEN CHANGED (ONE MORE WAVELET) AND RE-CALCULATE IT 
            int tmpIndexeTreeChanged = indexeTreeChanged;
           // double oldVal = 0;
           // double newVal = 0;
           // double weightOldGroup = Convert.ToDouble(N_treesInUse - 1) / (Convert.ToDouble(N_treesInUse));
           // double weightChangedTree = Convert.ToDouble(1) / (Convert.ToDouble(N_treesInUse));

            Helpers.applyFor(0, testingArr.Count(), i =>
            {
                double[] testPoint = Data_table[testingArr[i]];
                List<GeoWave> treeByIndex = RFdecTreeArrById[tmpIndexeTreeChanged];
                double normOfTheTreeByIndex = normsOfTrees[tmpIndexeTreeChanged];
                modefied_Lables[i] = askTreeMeanVal(testPoint, treeByIndex, normOfTheTreeByIndex);
                // - GET OLD VAL BEFORE CHANGE (COULD BE ZERO) HERE I DON'T SUPPORT MULTY LABELING - CAN BE EASLY CHANGED
                //oldVal = resultsByTree[tmpIndexeTreeChanged][i][0];
                resultsByTree[tmpIndexeTreeChanged][i] = modefied_Lables[i];
                //newVal = resultsByTree[tmpIndexeTreeChanged][i][0];
                ////SET TOTAL
                //if (newTree)
                //    resultsByForest[i] = weightChangedTree * (newVal - oldVal) + weightOldGroup * resultsByForest[i];
                //else
                //    resultsByForest[i] += weightChangedTree * (newVal - oldVal);
                int total = 0;
                double avg = 0;
                for (int j = 0; j < normsOfTrees.Count(); j++)
                {
                    if (normsOfTrees[j] == -1)
                        continue;
                    total++;
                    avg += resultsByTree[tmpIndexeTreeChanged][i][0];
                }
                resultsByForest[i] = avg / total;
            });

    

            double errorL2 = 0;
            for (int i = 0; i < testingArr.Count(); i++)
                errorL2 += (resultsByForest[i] - Data_Lables[testingArr[i]][0]) * (resultsByForest[i] - Data_Lables[testingArr[i]][0]);
            errorL2 = Math.Sqrt(errorL2 / Convert.ToDouble(testingArr.Count()));
            return errorL2;

            //if (newTree)
            //{
            //    weightChangedTree = Convert.ToDouble(N_treesInUse-1) / (Convert.ToDouble(N_treesInUse) + 1);
            //    weightOldGroup = Convert.ToDouble(1) / (Convert.ToDouble(N_treesInUse) + 1);
            //    //N_treesInUse++;
            //}

        }

        private List<int> getListFromFile(string fileName)
        {
            List<int> Arr = new List<int>();
            StreamReader sr = new StreamReader(File.OpenRead(fileName));

            while (!sr.EndOfStream )
            {
                Arr.Add(int.Parse(sr.ReadLine()));
            }
            sr.Close();
            return Arr;
        }


        private bool[] getDim2Take(recordConfig rc, int seed)
        {
            bool[] Dim2Take = new bool[rc.dim];

            Random ran = new Random(seed);
          
           // int varToUse = rc.NDimsinRF;
            //DEBUG
            int varToUse = rc.dim;
            for (int i = 0; i < varToUse; i++)
            {
                int index = ran.Next(0, rc.dim);
                if (Dim2Take[index] == true)
                    i--;
                else
                    Dim2Take[index] = true;
            }

            return Dim2Take;
        }

        private int getTopLevel(List<GeoWave> decisionGeoWaveArr)
        {
            int topLevel = 0;
            for (int i = 0; i < decisionGeoWaveArr.Count; i++)
                if (decisionGeoWaveArr[i].level > topLevel)
                    topLevel = decisionGeoWaveArr[i].level;
            return topLevel;
        }

        private List<int> bagging(List<int> trainingArr, double percent, int seed)//percent in [0,1]
        {
            //List<int> baggedArr = new List<int>();
            int N_rows = Convert.ToInt32(trainingArr.Count * percent);
            //int Seed = (int)DateTime.Now.Ticks;
            var ran = new Random(seed);
//            return Enumerable.Range(0, trainingArr.Count).OrderBy(x => ran.Next()).ToList().GetRange(0, N_rows);
            return trainingArr.OrderBy(x => ran.Next()).ToList().GetRange(0, N_rows);
        }

        private List<int> baggingBreiman(List<int> trainingArr, int seed)//percent in [0,1]
        {
            bool[] isSet = new bool[trainingArr.Count];
            List<int> baggedArr = new List<int>();
            var ran = new Random(seed);
            for (int i = 0; i < trainingArr.Count; i++)
            {
                int j = ran.Next(0, trainingArr.Count);
                if (isSet[j] == false)
                    baggedArr.Add(trainingArr[j]);
                isSet[j] = true;
            }          
            return baggedArr;
        }

        private int getGwIndexByKfunc(List<GeoWave> tmpTreeOrderedByNorm,
                                         recordConfig rc,
                                         double[][] trainingData, 
                                         double[][] trainingLabel,
                                         ref double bestNorm,
                                         List<int> testingArr)
        {
            //double[] best_index_norm = new double[2];//returned value ...
            int NumOfSkips = Convert.ToInt16(1 / rc.NskipsinKfunc);
            int skipSize = Convert.ToInt16(Math.Floor(rc.NskipsinKfunc * tmpTreeOrderedByNorm.Count));

            if (skipSize * NumOfSkips > tmpTreeOrderedByNorm.Count)
                MessageBox.Show(@"skipping made us go out of range - shuold not get here");

            double[] errArr = new double[NumOfSkips-1];

            ////DO THE HOPPING/SKIPPING
            Helpers.applyFor(1, NumOfSkips, i =>
            {
                double thresholdNorm = tmpTreeOrderedByNorm[i * skipSize].norm;
                double Tgt_approx_error = testDecisionTree(testingArr, trainingData, trainingLabel, tmpTreeOrderedByNorm, thresholdNorm, rc.boostNormTarget);
                double geowave_total_norm = getgeowaveNorm(tmpTreeOrderedByNorm, i * skipSize, rc.boostNormsecond, rc.boostTau);
                errArr[i - 1] = Tgt_approx_error + (rc.boostlamda_0 * geowave_total_norm);
            });
            

            int best_index = Enumerable.Range(0, errArr.Length).Aggregate((a, b) => (errArr[a] < errArr[b]) ? a : b); //minerror

            int first_index, last_index;
            if (best_index == 0)
            {
                first_index = 0;
                last_index = Math.Min(2 * skipSize, tmpTreeOrderedByNorm.Count);
            }
            else if (best_index == (NumOfSkips - 2))
            {
                first_index = Math.Max((best_index) * skipSize, 0);
                last_index = tmpTreeOrderedByNorm.Count;
            }
            else
            {
                first_index = Math.Max((best_index) * skipSize, 0);
                last_index = Math.Min((best_index + 2) * skipSize, tmpTreeOrderedByNorm.Count);
            }

            errArr = new double[last_index - first_index];

            //SEARCH IN THE BOUNDING 
            Helpers.applyFor(first_index, last_index, i =>
            {
                double thresholdNorm = tmpTreeOrderedByNorm[i].norm;
                double Tgt_approx_error = testDecisionTree(testingArr, trainingData, trainingLabel, tmpTreeOrderedByNorm, thresholdNorm, rc.boostNormTarget);
                double geowave_total_norm = getgeowaveNorm(tmpTreeOrderedByNorm, i, rc.boostNormsecond, rc.boostTau);
                errArr[i - first_index] = Tgt_approx_error + (rc.boostlamda_0 * geowave_total_norm);
            });
            

            best_index = Enumerable.Range(0, errArr.Length).Aggregate((a, b) => (errArr[a] < errArr[b]) ? a : b); //minerror
            bestNorm = tmpTreeOrderedByNorm[first_index + best_index].norm;

            return (first_index+ best_index);//indicates the number of waveletes to take (calced in order by ID)
        }

        private int getGwIndexByKfuncLessAcurate(List<GeoWave> tmpTreeOrderedByNorm,
                                         recordConfig rc,
                                         double[][] trainingData,
                                         double[][] trainingLabel,
                                         ref double bestNorm,
                                         List<int> testingArr)
        {
            //double[] best_index_norm = new double[2];//returned value ...
            int skipSize  = Convert.ToInt16(1 / rc.NskipsinKfunc);
            int NumOfSkips  = Convert.ToInt16(Math.Floor(rc.NskipsinKfunc * tmpTreeOrderedByNorm.Count));

            if (skipSize * NumOfSkips > tmpTreeOrderedByNorm.Count)
                MessageBox.Show(@"skipping made us go out of range - shuold not get here");

            double[] errArr = new double[NumOfSkips];

            ////DO THE HOPPING/SKIPPING
            Helpers.applyFor(0, NumOfSkips, i =>
            {
                double thresholdNorm = tmpTreeOrderedByNorm[i * skipSize].norm;
                double Tgt_approx_error = testDecisionTree(testingArr, trainingData, trainingLabel, tmpTreeOrderedByNorm, thresholdNorm, rc.boostNormTarget);
                double geowave_total_norm = getgeowaveNorm(tmpTreeOrderedByNorm, i * skipSize, rc.boostNormsecond, rc.boostTau);
                if (rc.boostNormsecond == 0)
                    geowave_total_norm += 1;
                errArr[i] = Tgt_approx_error + (rc.boostlamda_0 * geowave_total_norm);
            });
            int best_index = Enumerable.Range(0, errArr.Length).Aggregate((a, b) => (errArr[a] < errArr[b]) ? a : b); //minerror

            return best_index * skipSize;
        }

        private int getGwIndexByKfuncLessAcuratePooning(List<GeoWave> boostedTreeArrPooning, recordConfig rc, double[][] trainingDt, double[][] boostedLabelsPooning,List<int> testingArr)
        {
            int topLevel = getTopLevel(boostedTreeArrPooning);
            double[] errorTreeProoning = new double[topLevel];
            double[] NLevels = new double[topLevel];
            double[] errArr = new double[topLevel];
            Helpers.applyFor(0, topLevel, i =>
            {
                errorTreeProoning[i] = testDecisionTreeWithProoning(testingArr, _db.PCAtesting_dt, _db.testing_label, boostedTreeArrPooning, i + 1, 2);
                NLevels[i] = i;// * rc.hopping_size;
                errArr[i] = errorTreeProoning[i] + (NLevels[i] * rc.boostProoning_0);
            });
            int best_level = Enumerable.Range(0, errArr.Length).Aggregate((a, b) => (errArr[a] < errArr[b]) ? a : b); //minerror

            return best_level;
        }

        private void adjustlabels2Simplex4(double[][] estimatedLabels)
        {             
            double[] dist = new double[4];
            double[][] Data_Lables = new double[4][] ;
            for(int i=0; i<4;i++)
                Data_Lables[i] = new double[3];

            Data_Lables[0][0] = 0;      Data_Lables[0][1] = 0;                   Data_Lables[0][2] = 0; //0 0 0
            Data_Lables[1][0] = 1;      Data_Lables[1][1] = 0;                   Data_Lables[1][2] = 0; //1 0 0 
            Data_Lables[2][0] = 0.5;    Data_Lables[2][1] = Math.Sqrt(3)/2.0;    Data_Lables[2][2] = 0;//0.5 sqrt(3)/2 0
            Data_Lables[3][0] = 0.5;    Data_Lables[3][1] = Math.Sqrt(3)/6.0;    Data_Lables[3][2] = Math.Sqrt(3)/6.0; //0.5 sqrt(3)/6 sqrt(3)/6

            for (int i = 0; i < estimatedLabels.Count(); i++)
            {
                for(int k=0; k < 4; k++)
                    dist[k] = normPoint3D(estimatedLabels[i], Data_Lables[k]);
                int minIndex = Array.IndexOf(dist, Enumerable.Min(dist));
                if (minIndex == 0)
                    {estimatedLabels[i][0] = 0; estimatedLabels[i][1] = 0; estimatedLabels[i][2] = 0;}
                else if (minIndex == 1)
                    { estimatedLabels[i][0] = 1; estimatedLabels[i][1] = 0; estimatedLabels[i][2] = 0; }
                else if (minIndex == 2)
                    { estimatedLabels[i][0] = 0.5; estimatedLabels[i][1] = Math.Sqrt(3) / 2.0; estimatedLabels[i][2] = 0; }
                else
                    { estimatedLabels[i][0] = 0.5; estimatedLabels[i][1] = Math.Sqrt(3) / 6.0; estimatedLabels[i][2] = Math.Sqrt(3) / 6.0; }
            } 
        }

        private double normPoint3D(double[] p, double[] p2)
        {
            double norm = 0;
            for (int i = 0; i < p.Count(); i++)
                norm += (p[i] - p2[i]) * (p[i] - p2[i]);
            return norm;
        }

        //old version no testarr

        private double testDecisionTree(double[][] dataTable, double[][] dataLables, List<GeoWave> treeOrderedById, double normThreshold, int normLpType)
        {
            treeOrderedById = treeOrderedById.OrderBy(o => o.ID).ToList();

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[dataLables.Count()][];
            for (int i = 0; i < dataLables.Count(); i++)
                estimatedLabels[i] = new double[dataLables[0].Count()];
            Helpers.applyFor(0, dataLables.Count(), i =>
            {
                estimatedLabels[i] = askTreeMeanVal(dataTable[i], treeOrderedById, normThreshold);
            });
            double error = 0;
            if (normLpType == 2)
            {
                for (int j = 0; j < dataLables[0].Count(); j++)
                    for (int i = 0; i < dataLables.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - dataLables[i][j]) * (estimatedLabels[i][j] - dataLables[i][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(dataLables.Count()));
            }
            else if (normLpType == 1)
            {
                for (int j = 0; j < dataLables[0].Count(); j++)
                    for (int i = 0; i < dataLables.Count(); i++)
                    {
                        error += Math.Abs(estimatedLabels[i][j] - dataLables[i][j]);
                    }
            }
            else if (normLpType == -1)//max
            {
                List<double> errList = new List<double>();
                double tmp = 0;
                for (int i = 0; i < dataLables.Count(); i++)
                {
                    tmp = 0;
                    for (int j = 0; j < dataLables[0].Count(); j++)
                    {
                        tmp += Math.Abs(estimatedLabels[i][j] - dataLables[i][j]);
                    }
                    errList.Add(tmp);
                }
                error = errList.Max();
            }
            else if (normLpType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < dataLables.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * dataLables[i][0]) <= 0)
                        error += 1;
                }
            }
            else if (normLpType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2Simplex4(estimatedLabels);

                for (int i = 0; i < dataLables.Count(); i++)
                {
                    if (0.00001 < normPoint3D(estimatedLabels[i], dataLables[i]))
                        error += 1;
                }
            }
            else if (normLpType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < dataLables.Count(); i++)
                {
                    if (dataLables[i][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (dataLables[i][0] == -1)
                    {
                        NclassB += 1;
                        if (estimatedLabels[i][0] >= 0)
                            NMissclassB += 1;
                    }
                }
                error = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
            }

            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double testDecisionTreeWithProoning(double[][] dataTable, double[][] dataLables, List<GeoWave> treeOrderedById, int topLevel, int normLpType)
        {
            treeOrderedById = treeOrderedById.OrderBy(o => o.ID).ToList();

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[dataLables.Count()][];
            for (int i = 0; i < dataLables.Count(); i++)
                estimatedLabels[i] = new double[dataLables[0].Count()];

            Helpers.applyFor(0, dataLables.Count(), i =>
            {
                estimatedLabels[i] = askTreeMeanValAtLevel(dataTable[i], treeOrderedById, topLevel);
            });
           

            double error = 0;
            if (normLpType == 2)
            {
                for (int j = 0; j < dataLables[0].Count(); j++)
                    for (int i = 0; i < dataLables.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - dataLables[i][j]) * (estimatedLabels[i][j] - dataLables[i][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(dataLables.Count()));
            }
            else if (normLpType == 1)
            {
                for (int j = 0; j < dataLables[0].Count(); j++)
                    for (int i = 0; i < dataLables.Count(); i++)
                    {
                        error += Math.Abs(estimatedLabels[i][j] - dataLables[i][j]);
                    }
            }
            else if (normLpType == -1)//max
            {
                List<double> errList = new List<double>();
                double tmp = 0;
                for (int i = 0; i < dataLables.Count(); i++)
                {
                    tmp = 0;
                    for (int j = 0; j < dataLables[0].Count(); j++)
                    {
                        tmp += Math.Abs(estimatedLabels[i][j] - dataLables[i][j]);
                    }
                    errList.Add(tmp);
                }
                error = errList.Max();
            }
            else if (normLpType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < dataLables.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * dataLables[i][0]) <= 0)
                        error += 1;
                }
            }
            else if (normLpType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2Simplex4(estimatedLabels);

                for (int i = 0; i < dataLables.Count(); i++)
                {
                    if (0.00001 < normPoint3D(estimatedLabels[i], dataLables[i]))
                        error += 1;
                }
            }
            else if (normLpType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < dataLables.Count(); i++)
                {
                    if (dataLables[i][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (dataLables[i][0] == -1)
                    {
                        NclassB += 1;
                        if (estimatedLabels[i][0] >= 0)
                            NMissclassB += 1;
                    }
                }
                error = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
            }

            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double testDecisionTreeRF(double[][] dataTable, double[][] dataLables, List<GeoWave>[] rFdecTreeArr, double normThreshold, int normLpType)
        {
            Helpers.applyFor(0, rFdecTreeArr.Count(), i =>
            {
                rFdecTreeArr[i] = rFdecTreeArr[i].OrderBy(o => o.ID).ToList();
            });
           

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[dataLables.Count()][];
            for (int i = 0; i < dataLables.Count(); i++)
                estimatedLabels[i] = new double[dataLables[0].Count()];

            Helpers.applyFor(0, dataLables.Count(), i =>
            {
                //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                double[] point = new double[_rc.dim];//Data_table[0].Count()
                //Data_table.CopyTo(point, i);
                for (int j = 0; j < _rc.dim; j++)//Data_table[0].Count()
                    point[j] = double.Parse(dataTable[i][j].ToString());
                double[][] tmpLabel = new double[rFdecTreeArr.Count()][];
                Helpers.applyFor(0, rFdecTreeArr.Count(), j =>
                {
                    tmpLabel[j] = askTreeMeanVal(point, rFdecTreeArr[j], normThreshold);
                });

                for (int j = 0; j < dataLables[0].Count(); j++)
                    for (int k = 0; k < rFdecTreeArr.Count(); k++)
                        estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(rFdecTreeArr.Count());
            });

           

            double error = 0;
            if (normLpType == 2)
            {
                for (int j = 0; j < dataLables[0].Count(); j++)
                    for (int i = 0; i < dataLables.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - dataLables[i][j]) * (estimatedLabels[i][j] - dataLables[i][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(dataLables.Count()));
            }
            else if (normLpType == 1)//L1
            {
                for (int j = 0; j < dataLables[0].Count(); j++)
                    for (int i = 0; i < dataLables.Count(); i++)
                    {
                        error += Math.Abs(estimatedLabels[i][j] - dataLables[i][j]);
                    }
            }
            else if (normLpType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < dataLables.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * dataLables[i][0]) <= 0)
                        error += 1;
                }
            }
            else if (normLpType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2Simplex4(estimatedLabels);

                for (int i = 0; i < dataLables.Count(); i++)
                {
                    if (0.00001 < normPoint3D(estimatedLabels[i], dataLables[i]))
                        error += 1;
                }
            }
            else if (normLpType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < dataLables.Count(); i++)
                {
                    if (dataLables[i][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (dataLables[i][0] == -1)
                    {
                        NclassB += 1;
                        if (estimatedLabels[i][0] >= 0)
                            NMissclassB += 1;
                    }
                }
                error = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
            }

            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double testDecisionTreeManyRFNormNbound(double[][] dataTable, double[][] dataLables, List<GeoWave>[] rFdecTreeArr, double normThreshold, int boundLevel, int normLpType)
        {
            Helpers.applyFor(0, rFdecTreeArr.Count(), i =>
            {
                rFdecTreeArr[i] = rFdecTreeArr[i].OrderBy(o => o.ID).ToList();
            });
           

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[dataLables.Count()][];
            for (int i = 0; i < dataLables.Count(); i++)
                estimatedLabels[i] = new double[dataLables[0].Count()];

            Helpers.applyFor(0, dataLables.Count(), i =>
            {
                //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                double[] point = new double[_rc.dim];//Data_table[0].Count()
                //Data_table.CopyTo(point, i);
                for (int j = 0; j < _rc.dim; j++)//Data_table[0].Count()
                    point[j] = double.Parse(dataTable[i][j].ToString());
                double[][] tmpLabel = new double[rFdecTreeArr.Count()][];
                Helpers.applyFor(0, rFdecTreeArr.Count(), j =>
                {
                    tmpLabel[j] = askTreeMeanValBoundLevel(point, rFdecTreeArr[j], normThreshold, boundLevel);
                });

                for (int j = 0; j < dataLables[0].Count(); j++)
                    for (int k = 0; k < rFdecTreeArr.Count(); k++)
                        estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(rFdecTreeArr.Count());
            });

           

            double error = 0;
            if (normLpType == 2)
            {
                for (int j = 0; j < dataLables[0].Count(); j++)
                    for (int i = 0; i < dataLables.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - dataLables[i][j]) * (estimatedLabels[i][j] - dataLables[i][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(dataLables.Count()));
            }
            else if (normLpType == 1)//L1
            {
                for (int j = 0; j < dataLables[0].Count(); j++)
                    for (int i = 0; i < dataLables.Count(); i++)
                    {
                        error += Math.Abs(estimatedLabels[i][j] - dataLables[i][j]);
                    }
            }
            else if (normLpType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < dataLables.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * dataLables[i][0]) <= 0)
                        error += 1;
                }
            }
            else if (normLpType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2Simplex4(estimatedLabels);

                for (int i = 0; i < dataLables.Count(); i++)
                {
                    if (0.00001 < normPoint3D(estimatedLabels[i], dataLables[i]))
                        error += 1;
                }
            }
            else if (normLpType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < dataLables.Count(); i++)
                {
                    if (dataLables[i][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (dataLables[i][0] == -1)
                    {
                        NclassB += 1;
                        if (estimatedLabels[i][0] >= 0)
                            NMissclassB += 1;
                    }
                }
                error = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
            }

            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double[] testDecisionTreeManyRF(double[][] dataTable, double[][] dataLables, List<GeoWave>[] rFdecTreeArr, double normThreshold, int normLpType)
        {
            Helpers.applyFor(0, rFdecTreeArr.Count(), i =>
            {
                rFdecTreeArr[i] = rFdecTreeArr[i].OrderBy(o => o.ID).ToList();
            });
            

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[rFdecTreeArr.Count()][][];//num of trees, label index, label values (or value in most cases)
            for (int i = 0; i < rFdecTreeArr.Count(); i++)
            {
                estimatedLabels[i] = new double[dataLables.Count()][];
                for (int j = 0; j < dataLables.Count(); j++)
                    estimatedLabels[i][j] = new double[dataLables[0].Count()];
            }

            Helpers.applyFor(0, dataLables.Count(), i =>
            {
                //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                double[] point = new double[_rc.dim];//Data_table[0].Count()
                //Data_table.CopyTo(point, i);
                for (int j = 0; j < _rc.dim; j++)//Data_table[0].Count()
                    point[j] = double.Parse(dataTable[i][j].ToString());
                double[][] tmpLabel = new double[rFdecTreeArr.Count()][];
                Helpers.applyFor(0, rFdecTreeArr.Count(), j =>
                {
                    tmpLabel[j] = askTreeMeanVal(point, rFdecTreeArr[j], normThreshold);
                });

                for (int j = 0; j < dataLables[0].Count(); j++)
                {
                    estimatedLabels[0][i][j] = tmpLabel[0][j];
                    for (int k = 1; k < rFdecTreeArr.Count(); k++)
                        estimatedLabels[k][i][j] = (Convert.ToDouble(k) / (Convert.ToDouble(k) + 1)) * estimatedLabels[k - 1][i][j] + (1 / (Convert.ToDouble(k) + 1)) * tmpLabel[k][j];
                }
            });

            double[] error = new double[rFdecTreeArr.Count()];
            switch (normLpType)
            {
                case 2:
                    for (int k = 0; k < rFdecTreeArr.Count(); k++)
                    {
                        for (int j = 0; j < dataLables[0].Count(); j++)
                            for (int i = 0; i < dataLables.Count(); i++)
                                error[k] += (estimatedLabels[k][i][j] - dataLables[i][j]) * (estimatedLabels[k][i][j] - dataLables[i][j]);
                        error[k] = Math.Sqrt(error[k] / Convert.ToDouble(dataLables.Count()));
                    }
                    break;
                case 1:
                    for (int k = 0; k < rFdecTreeArr.Count(); k++)
                    {
                        for (int j = 0; j < dataLables[0].Count(); j++)
                            for (int i = 0; i < dataLables.Count(); i++)
                                error[k] += Math.Abs(estimatedLabels[k][i][j] - dataLables[i][j]);
                    }
                    break;
                default:
                    if (normLpType == -2 && estimatedLabels[0][0].Count() == 1)//+-1 labels + BER
                    {
                        double threshVal = 0.5 * (Form1.upper_label + Form1.lower_label);
                        for (int k = 0; k < rFdecTreeArr.Count(); k++)
                        {
                            double NclassA = 0;
                            double NclassB = 0;
                            double NMissclassA = 0;
                            double NMissclassB = 0;

                            for (int j = 0; j < dataLables[0].Count(); j++)
                                for (int i = 0; i < dataLables.Count(); i++)
                                {
                                    if (dataLables[i][j] == Form1.upper_label)
                                    {
                                        NclassA += 1;
                                        if (estimatedLabels[k][i][j] <= threshVal)
                                            NMissclassA += 1;
                                    }
                                    if (dataLables[i][j] == Form1.lower_label)
                                    {
                                        NclassB += 1;
                                        if (estimatedLabels[k][i][j] >= threshVal)
                                            NMissclassB += 1;
                                    }
                                }
                            error[k] = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
                        }
                    }
                    break;
            }
            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double[] testDecisionTreeManyRFNoVoting(double[][] dataTable, double[][] dataLables, List<GeoWave>[] rFdecTreeArr, double normThreshold, int normLpType)
        {
            Helpers.applyFor(0, rFdecTreeArr.Count(), i =>
            {
                rFdecTreeArr[i] = rFdecTreeArr[i].OrderBy(o => o.ID).ToList();
            });
            

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[rFdecTreeArr.Count()][][];//num of trees, label index, label values (or value in most cases)
            for (int i = 0; i < rFdecTreeArr.Count(); i++)
            {
                estimatedLabels[i] = new double[dataLables.Count()][];
                for (int j = 0; j < dataLables.Count(); j++)
                    estimatedLabels[i][j] = new double[dataLables[0].Count()];
            }
            Helpers.applyFor(0, dataLables.Count(), i =>
            {
                //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                double[] point = new double[_rc.dim];//Data_table[0].Count()
                //Data_table.CopyTo(point, i);
                for (int j = 0; j < _rc.dim; j++)//Data_table[0].Count()
                    point[j] = double.Parse(dataTable[i][j].ToString());
                double[][] tmpLabel = new double[rFdecTreeArr.Count()][];
                Helpers.applyFor(0, rFdecTreeArr.Count(), j =>
                {
                    tmpLabel[j] = askTreeMeanVal(point, rFdecTreeArr[j], normThreshold);
                });

                for (int j = 0; j < dataLables[0].Count(); j++)
                {
                    for (int k = 0; k < rFdecTreeArr.Count(); k++)
                        estimatedLabels[k][i][j] = tmpLabel[k][j];
                }
            });

           

            double[] error = new double[rFdecTreeArr.Count()];
            switch (normLpType)
            {
                case 2:
                    for (int k = 0; k < rFdecTreeArr.Count(); k++)
                    {
                        for (int j = 0; j < dataLables[0].Count(); j++)
                            for (int i = 0; i < dataLables.Count(); i++)
                                error[k] += (estimatedLabels[k][i][j] - dataLables[i][j]) * (estimatedLabels[k][i][j] - dataLables[i][j]);
                        error[k] = Math.Sqrt(error[k] / Convert.ToDouble(dataLables.Count()));
                    }
                    break;
                case 1:
                    for (int k = 0; k < rFdecTreeArr.Count(); k++)
                    {
                        for (int j = 0; j < dataLables[0].Count(); j++)
                            for (int i = 0; i < dataLables.Count(); i++)
                                error[k] += Math.Abs(estimatedLabels[k][i][j] - dataLables[i][j]);
                    }
                    break;
                default:
                    if (normLpType == -2 && estimatedLabels[0][0].Count() == 1)//+-1 labels + BER
                    {
                        double threshVal = 0.5 * (Form1.upper_label + Form1.lower_label);
                        for (int k = 0; k < rFdecTreeArr.Count(); k++)
                        {
                            double NclassA = 0;
                            double NclassB = 0;
                            double NMissclassA = 0;
                            double NMissclassB = 0;

                            for (int j = 0; j < dataLables[0].Count(); j++)
                                for (int i = 0; i < dataLables.Count(); i++)
                                {
                                    if (dataLables[i][j] == Form1.upper_label)
                                    {
                                        NclassA += 1;
                                        if (estimatedLabels[k][i][j] <= threshVal)
                                            NMissclassA += 1;
                                    }
                                    if (dataLables[i][j] == Form1.lower_label)
                                    {
                                        NclassB += 1;
                                        if (estimatedLabels[k][i][j] >= threshVal)
                                            NMissclassB += 1;
                                    }
                                }
                            error[k] = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
                        }
                    }
                    break;
            }
            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double[] testDecisionTreeManyRFNoVoting(List<int>[] arrRfIndecesList, double[][] dataTable, double[][] dataLables, List<GeoWave>[] rFdecTreeArr, double normThreshold, int normLpType)
        {
            Helpers.applyFor(0, rFdecTreeArr.Count(), i =>
            {
                rFdecTreeArr[i] = rFdecTreeArr[i].OrderBy(o => o.ID).ToList();
            });
           

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[rFdecTreeArr.Count()][][];//num of trees, label index, label values (or value in most cases)
            for (int i = 0; i < rFdecTreeArr.Count(); i++)
            {
                estimatedLabels[i] = new double[arrRfIndecesList[i].Count()][];
                for (int j = 0; j < arrRfIndecesList[i].Count(); j++)
                    estimatedLabels[i][j] = new double[dataLables[0].Count()];
            }
            Helpers.applyFor(0, rFdecTreeArr.Count(), i =>
            {
                //for each tree go over all points
                for (int j = 0; j < arrRfIndecesList[i].Count(); j++)
                {
                    double[] point = new double[_rc.dim];
                    for (int t = 0; t < _rc.dim; t++)
                        point[t] = double.Parse(dataTable[arrRfIndecesList[i][j]][t].ToString());
                    double[] tmpLabel = askTreeMeanVal(point, rFdecTreeArr[i], normThreshold);
                    for (int t = 0; t < dataLables[0].Count(); t++)
                    {
                        estimatedLabels[i][j][t] = tmpLabel[t];
                    }
                }
            });

            

            double[] error = new double[rFdecTreeArr.Count()];
            switch (normLpType)
            {
                case 2:
                    for (int k = 0; k < rFdecTreeArr.Count(); k++)
                    {
                        for (int j = 0; j < dataLables[0].Count(); j++)
                            for (int i = 0; i < arrRfIndecesList[k].Count(); i++)//each tree may have diffrent label size
                                error[k] += (estimatedLabels[k][i][j] - dataLables[arrRfIndecesList[k][i]][j]) * (estimatedLabels[k][i][j] - dataLables[arrRfIndecesList[k][i]][j]);
                        error[k] = Math.Sqrt(error[k] / Convert.ToDouble(arrRfIndecesList[k].Count()));
                    }
                    break;
                case 1:
                    for (int k = 0; k < rFdecTreeArr.Count(); k++)
                    {
                        for (int j = 0; j < dataLables[0].Count(); j++)
                            for (int i = 0; i < arrRfIndecesList[k].Count(); i++)
                                error[k] += Math.Abs(estimatedLabels[k][i][j] - dataLables[arrRfIndecesList[k][i]][j]);
                    }
                    break;
            }
            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double[] testDecisionTreeManyRFbyIndex(double[][] dataTable, double[][] dataLables, List<GeoWave>[] rFdecTreeArr, int indexThreshold, int normLpType)
        {
            List<GeoWave>[] RFdecTreeArrById = new List<GeoWave>[rFdecTreeArr.Count()];
            Helpers.applyFor(0, rFdecTreeArr.Count(), i =>
            {
                RFdecTreeArrById[i] = rFdecTreeArr[i].OrderBy(o => o.ID).ToList();
            });
           
            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[rFdecTreeArr.Count()][][];//num of trees, label index, label values (or value in most cases)
            for (int i = 0; i < rFdecTreeArr.Count(); i++)
            {
                estimatedLabels[i] = new double[dataLables.Count()][];
                for (int j = 0; j < dataLables.Count(); j++)
                    estimatedLabels[i][j] = new double[dataLables[0].Count()];
            }
            Helpers.applyFor(0, dataLables.Count(), i =>
            {
                //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                double[] point = new double[_rc.dim];//Data_table[0].Count()
                //Data_table.CopyTo(point, i);
                for (int j = 0; j < _rc.dim; j++)//Data_table[0].Count()
                    point[j] = double.Parse(dataTable[i][j].ToString());
                double[][] tmpLabel = new double[rFdecTreeArr.Count()][];
                Helpers.applyFor(0, rFdecTreeArr.Count(), j =>
                {
                    tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArrById[j], rFdecTreeArr[j][indexThreshold].norm);
                });

                for (int j = 0; j < dataLables[0].Count(); j++)
                {
                    estimatedLabels[0][i][j] = tmpLabel[0][j];
                    for (int k = 1; k < rFdecTreeArr.Count(); k++)
                        estimatedLabels[k][i][j] = (Convert.ToDouble(k) / (Convert.ToDouble(k) + 1)) * estimatedLabels[k - 1][i][j] + (1 / (Convert.ToDouble(k) + 1)) * tmpLabel[k][j];
                }
            });

          

            double[] error = new double[rFdecTreeArr.Count()];
            if (normLpType == 2)
            {
                for (int k = 0; k < rFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < dataLables[0].Count(); j++)
                        for (int i = 0; i < dataLables.Count(); i++)
                            error[k] += (estimatedLabels[k][i][j] - dataLables[i][j]) * (estimatedLabels[k][i][j] - dataLables[i][j]);
                    error[k] = Math.Sqrt(error[k] / Convert.ToDouble(dataLables.Count()));
                }
            }
            if (normLpType == 1)
            {
                for (int k = 0; k < rFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < dataLables[0].Count(); j++)
                        for (int i = 0; i < dataLables.Count(); i++)
                            error[k] += Math.Abs(estimatedLabels[k][i][j] - dataLables[i][j]);
                }
            }
            else if (normLpType == -2 && estimatedLabels[0][0].Count() == 1)//+-1 labels + BER
            {
                double threshVal = 0.5 * (Form1.upper_label + Form1.lower_label);
                for (int k = 0; k < rFdecTreeArr.Count(); k++)
                {
                    double NclassA = 0;
                    double NclassB = 0;
                    double NMissclassA = 0;
                    double NMissclassB = 0;

                    for (int j = 0; j < dataLables[0].Count(); j++)
                        for (int i = 0; i < dataLables.Count(); i++)
                        {
                            if (dataLables[i][j] == Form1.upper_label)
                            {
                                NclassA += 1;
                                if (estimatedLabels[k][i][j] <= threshVal)
                                    NMissclassA += 1;
                            }
                            if (dataLables[i][j] == Form1.lower_label)
                            {
                                NclassB += 1;
                                if (estimatedLabels[k][i][j] >= threshVal)
                                    NMissclassB += 1;
                            }
                        }
                    error[k] = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
                }
            }
            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double testDecisionTreeRF(double[][] dataTable, double[][] dataLables, List<GeoWave>[] rFdecTreeArr, int topLevel, int normLpType)
        {
            Helpers.applyFor(0, rFdecTreeArr.Count(), i =>
            {
                rFdecTreeArr[i] = rFdecTreeArr[i].OrderBy(o => o.ID).ToList();
            });
           

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[dataLables.Count()][];
            for (int i = 0; i < dataLables.Count(); i++)
                estimatedLabels[i] = new double[dataLables[0].Count()];

            Helpers.applyFor(0, dataLables.Count(), i =>
            {
                //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                double[] point = new double[_rc.dim];//Data_table[0].Count()
                //Data_table.CopyTo(point, i);
                for (int j = 0; j < _rc.dim; j++)//Data_table[0].Count()
                    point[j] = double.Parse(dataTable[i][j].ToString());
                double[][] tmpLabel = new double[rFdecTreeArr.Count()][];
                Helpers.applyFor(0, rFdecTreeArr.Count(), j =>
                {
                    tmpLabel[j] = askTreeMeanValAtLevel(point, rFdecTreeArr[j], topLevel);
                });

                for (int j = 0; j < dataLables[0].Count(); j++)
                    for (int k = 0; k < rFdecTreeArr.Count(); k++)
                        estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(rFdecTreeArr.Count());
            });

           

            double error = 0;
            if (normLpType == 2)
            {
                for (int j = 0; j < dataLables[0].Count(); j++)
                    for (int i = 0; i < dataLables.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - dataLables[i][j]) * (estimatedLabels[i][j] - dataLables[i][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(dataLables.Count()));
            }
            if (normLpType == 1)
            {
                for (int j = 0; j < dataLables[0].Count(); j++)
                    for (int i = 0; i < dataLables.Count(); i++)
                    {
                        error += Math.Abs(estimatedLabels[i][j] - dataLables[i][j]);
                    }
            }
            else if (normLpType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < dataLables.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * dataLables[i][0]) <= 0)
                        error += 1;
                }
            }
            else if (normLpType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2Simplex4(estimatedLabels);

                for (int i = 0; i < dataLables.Count(); i++)
                {
                    if (0.00001 < normPoint3D(estimatedLabels[i], dataLables[i]))
                        error += 1;
                }
            }
            else if (normLpType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < dataLables.Count(); i++)
                {
                    if (dataLables[i][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (dataLables[i][0] == -1)
                    {
                        NclassB += 1;
                        if (estimatedLabels[i][0] >= 0)
                            NMissclassB += 1;
                    }
                }
                error = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
            }

            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double[] testDecisionTreeManyRF(double[][] dataTable, double[][] dataLables, List<GeoWave>[] rFdecTreeArr, int topLevel, int normLpType)
        {
            Helpers.applyFor(0, rFdecTreeArr.Count(), i =>
                {
                    rFdecTreeArr[i] = rFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
           

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[rFdecTreeArr.Count()][][];//num of trees, label index, label values (or value in most cases)
            for (int i = 0; i < rFdecTreeArr.Count(); i++)
            {
                estimatedLabels[i] = new double[dataLables.Count()][];
                for (int j = 0; j < dataLables.Count(); j++)
                    estimatedLabels[i][j] = new double[dataLables[0].Count()];
            }

            Helpers.applyFor(0, dataLables.Count(), i =>
            {
                //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                double[] point = new double[_rc.dim];//Data_table[0].Count()
                //Data_table.CopyTo(point, i);
                for (int j = 0; j < _rc.dim; j++)//Data_table[0].Count()
                    point[j] = double.Parse(dataTable[i][j].ToString());
                double[][] tmpLabel = new double[rFdecTreeArr.Count()][];
                Helpers.applyFor(0, rFdecTreeArr.Count(), j =>
                {
                    tmpLabel[j] = askTreeMeanValAtLevel(point, rFdecTreeArr[j], topLevel);
                });

                for (int j = 0; j < dataLables[0].Count(); j++)
                {
                    estimatedLabels[0][i][j] = tmpLabel[0][j];
                    for (int k = 1; k < rFdecTreeArr.Count(); k++)
                        estimatedLabels[k][i][j] = (Convert.ToDouble(k) / (Convert.ToDouble(k) + 1)) * estimatedLabels[k - 1][i][j] + (1 / (Convert.ToDouble(k) + 1)) * tmpLabel[k][j];
                }
            });

           

            double[] error = new double[rFdecTreeArr.Count()];
            switch (normLpType)
            {
                case 2:
                    for (int k = 0; k < rFdecTreeArr.Count(); k++)
                    {
                        for (int j = 0; j < dataLables[0].Count(); j++)
                            for (int i = 0; i < dataLables.Count(); i++)
                                error[k] += (estimatedLabels[k][i][j] - dataLables[i][j]) * (estimatedLabels[k][i][j] - dataLables[i][j]);
                        error[k] = Math.Sqrt(error[k] / Convert.ToDouble(dataLables.Count()));
                    }
                    break;
                case 1:
                    for (int k = 0; k < rFdecTreeArr.Count(); k++)
                    {
                        for (int j = 0; j < dataLables[0].Count(); j++)
                            for (int i = 0; i < dataLables.Count(); i++)
                                error[k] += Math.Abs(estimatedLabels[k][i][j] - dataLables[i][j]);
                    }
                    break;
                default:
                    if (normLpType == -2 && estimatedLabels[0][0].Count() == 1)//+-1 labels + BER
                    {
                        double threshVal = 0.5 * (Form1.upper_label + Form1.lower_label);
                        for (int k = 0; k < rFdecTreeArr.Count(); k++)
                        {
                            double NclassA = 0;
                            double NclassB = 0;
                            double NMissclassA = 0;
                            double NMissclassB = 0;

                            for (int j = 0; j < dataLables[0].Count(); j++)
                                for (int i = 0; i < dataLables.Count(); i++)
                                {
                                    if (dataLables[i][j] == Form1.upper_label)
                                    {
                                        NclassA += 1;
                                        if (estimatedLabels[k][i][j] <= threshVal)
                                            NMissclassA += 1;
                                    }
                                    if (dataLables[i][j] == Form1.lower_label)
                                    {
                                        NclassB += 1;
                                        if (estimatedLabels[k][i][j] >= threshVal)
                                            NMissclassB += 1;
                                    }
                                }
                            error[k] = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
                        }
                    }
                    break;
            }
            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double[] testDecisionTreeManyRFNoVoting(double[][] dataTable, double[][] dataLables, List<GeoWave>[] rFdecTreeArr, int topLevel, int normLpType)
        {
            Helpers.applyFor(0, rFdecTreeArr.Count(), i =>
            {
                rFdecTreeArr[i] = rFdecTreeArr[i].OrderBy(o => o.ID).ToList();
            });
            
            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[rFdecTreeArr.Count()][][];//num of trees, label index, label values (or value in most cases)
            for (int i = 0; i < rFdecTreeArr.Count(); i++)
            {
                estimatedLabels[i] = new double[dataLables.Count()][];
                for (int j = 0; j < dataLables.Count(); j++)
                    estimatedLabels[i][j] = new double[dataLables[0].Count()];
            }
            Helpers.applyFor(0, dataLables.Count(), i =>
            {
                //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                double[] point = new double[_rc.dim];//Data_table[0].Count()
                //Data_table.CopyTo(point, i);
                for (int j = 0; j < _rc.dim; j++)//Data_table[0].Count()
                    point[j] = double.Parse(dataTable[i][j].ToString());
                double[][] tmpLabel = new double[rFdecTreeArr.Count()][];
                Helpers.applyFor(0, rFdecTreeArr.Count(), j =>
                {
                    tmpLabel[j] = askTreeMeanValAtLevel(point, rFdecTreeArr[j], topLevel);
                });

                for (int j = 0; j < dataLables[0].Count(); j++)
                {
                    for (int k = 0; k < rFdecTreeArr.Count(); k++)
                        estimatedLabels[k][i][j] = tmpLabel[k][j];
                }
            });
          

            double[] error = new double[rFdecTreeArr.Count()];
            if (normLpType == 2)
            {
                for (int k = 0; k < rFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < dataLables[0].Count(); j++)
                        for (int i = 0; i < dataLables.Count(); i++)
                            error[k] += (estimatedLabels[k][i][j] - dataLables[i][j]) * (estimatedLabels[k][i][j] - dataLables[i][j]);
                    error[k] = Math.Sqrt(error[k] / Convert.ToDouble(dataLables.Count()));
                }
            }
            if (normLpType == 1)
            {
                for (int k = 0; k < rFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < dataLables[0].Count(); j++)
                        for (int i = 0; i < dataLables.Count(); i++)
                            error[k] += Math.Abs(estimatedLabels[k][i][j] - dataLables[i][j]);
                }
            }
            else if (normLpType == -2 && estimatedLabels[0][0].Count() == 1)//+-1 labels + BER
            {
                double threshVal = 0.5 * (Form1.upper_label + Form1.lower_label);
                for (int k = 0; k < rFdecTreeArr.Count(); k++)
                {
                    double NclassA = 0;
                    double NclassB = 0;
                    double NMissclassA = 0;
                    double NMissclassB = 0;

                    for (int j = 0; j < dataLables[0].Count(); j++)
                        for (int i = 0; i < dataLables.Count(); i++)
                        {
                            if (dataLables[i][j] == Form1.upper_label)
                            {
                                NclassA += 1;
                                if (estimatedLabels[k][i][j] <= threshVal)
                                    NMissclassA += 1;
                            }
                            if (dataLables[i][j] == Form1.lower_label)
                            {
                                NclassB += 1;
                                if (estimatedLabels[k][i][j] >= threshVal)
                                    NMissclassB += 1;
                            }
                        }
                    error[k] = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
                }
            }
            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double testDecisionTreeBoosting(double[][] testingDt, double[][] testingLabel, List<GeoWave>[] boostTreeArr, double normThreshold, int normLpType, double[] maxNorms)
        {
            Helpers.applyFor(0, boostTreeArr.Count(), i =>
            {
                boostTreeArr[i] = boostTreeArr[i].OrderBy(o => o.ID).ToList();
            });
           
            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[testingLabel.Count()][];
            for (int i = 0; i < testingLabel.Count(); i++)
                estimatedLabels[i] = new double[testingLabel[0].Count()];

            Helpers.applyFor(0, testingLabel.Count(), i =>
            {
                //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                double[] point = new double[_rc.dim];//testing_dt[0].Count()
                //Data_table.CopyTo(point, i);
                for (int j = 0; j < _rc.dim; j++)//testing_dt[0].Count()
                    point[j] = double.Parse(testingDt[i][j].ToString(CultureInfo.InvariantCulture));
                double[][] tmpLabel = new double[boostTreeArr.Count()][];
                Helpers.applyFor(0, boostTreeArr.Count(), j =>
                {
                    if (normThreshold < maxNorms[j])
                        tmpLabel[j] = askTreeMeanVal(point, boostTreeArr[j], maxNorms[j]);
                    else
                        tmpLabel[j] = askTreeMeanVal(point, boostTreeArr[j], normThreshold);
                });

                for (int j = 0; j < testingLabel[0].Count(); j++)
                    for (int k = 0; k < boostTreeArr.Count(); k++)
                        estimatedLabels[i][j] += tmpLabel[k][j];
            });

            double error = 0;
            if (normLpType == 2)
            {
                for (int j = 0; j < testingLabel[0].Count(); j++)
                    for (int i = 0; i < testingLabel.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - testingLabel[i][j]) * (estimatedLabels[i][j] - testingLabel[i][j]);
                    }
                error = Math.Sqrt(error);
            }
            else if (normLpType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < testingLabel.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * testingLabel[i][0]) <= 0)
                        error += 1;
                }
            }
            else if (normLpType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2Simplex4(estimatedLabels);

                for (int i = 0; i < testingLabel.Count(); i++)
                {
                    if (0.00001 < normPoint3D(estimatedLabels[i], testingLabel[i]))
                        error += 1;
                }
            }
            else if (normLpType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < testingLabel.Count(); i++)
                {
                    if (testingLabel[i][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (testingLabel[i][0] == -1)
                    {
                        NclassB += 1;
                        if (estimatedLabels[i][0] >= 0)
                            NMissclassB += 1;
                    }
                }
                error = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
            }
            return error;
        }

        private double testDecisionTreeBoostingLearningRate(double[][] testingDt, double[][] testingLabel, List<GeoWave>[] boostTreeArr, int normLpType, double[] maxNorms, int ntrees)
        {
            Helpers.applyFor(0, ntrees, i =>
            {
                boostTreeArr[i] = boostTreeArr[i].OrderBy(o => o.ID).ToList();
            });
           

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[testingLabel.Count()][];
            for (int i = 0; i < testingLabel.Count(); i++)
                estimatedLabels[i] = new double[testingLabel[0].Count()];

            Helpers.applyFor(0, testingLabel.Count(), i =>
            {
                //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                double[] point = new double[_rc.dim];//testing_dt[0].Count()
                //Data_table.CopyTo(point, i);
                for (int j = 0; j < _rc.dim; j++)//testing_dt[0].Count()
                    point[j] = double.Parse(testingDt[i][j].ToString());
                double[][] tmpLabel = new double[boostTreeArr.Count()][];
                Helpers.applyFor(0, ntrees, j =>
                {
                    tmpLabel[j] = askTreeMeanVal(point, boostTreeArr[j], maxNorms[j]);
                });

                for (int j = 0; j < testingLabel[0].Count(); j++)
                    for (int k = 0; k < ntrees; k++)
                        estimatedLabels[i][j] += tmpLabel[k][j];

            });

          

            double error = 0;
            if (normLpType == 2)
            {
                for (int j = 0; j < testingLabel[0].Count(); j++)
                    for (int i = 0; i < testingLabel.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - testingLabel[i][j]) * (estimatedLabels[i][j] - testingLabel[i][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(estimatedLabels.Count()));
            }
            else if (normLpType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < testingLabel.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * testingLabel[i][0]) <= 0)
                        error += 1;
                }
            }
            else if (normLpType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2Simplex4(estimatedLabels);

                for (int i = 0; i < testingLabel.Count(); i++)
                {
                    if (0.00001 < normPoint3D(estimatedLabels[i], testingLabel[i]))
                        error += 1;
                }
            }
            else if (normLpType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < testingLabel.Count(); i++)
                {
                    if (testingLabel[i][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (testingLabel[i][0] == -1)
                    {
                        NclassB += 1;
                        if (estimatedLabels[i][0] >= 0)
                            NMissclassB += 1;
                    }
                }
                error = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
            }
            return error;
        }

        private void testDecisionTreeBoostingProoning(double[][] testingDt, double[][] testingLabel, List<GeoWave>[] boostTreeArrPooning, int[] bestLevel, int normLpType, double[] error)
        {
            Helpers.applyFor(0, boostTreeArrPooning.Count(), i =>
            {
                boostTreeArrPooning[i] = boostTreeArrPooning[i].OrderBy(o => o.ID).ToList();
            });
           

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[testingLabel.Count()][][];
            for (int i = 0; i < testingLabel.Count(); i++)
            {
                estimatedLabels[i] = new double[_rc.boostNum][];
                for (int j = 0; j < _rc.boostNum; j++)
                    estimatedLabels[i][j] = new double[testingLabel[0].Count()];
            }


            if (Form1.rumPrallel)
            {
                Parallel.For(0, testingLabel.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[_rc.dim];//testing_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < _rc.dim; j++)//testing_dt[0].Count()
                        point[j] = double.Parse(testingDt[i][j].ToString());
                    double[][] tmpLabel = new double[boostTreeArrPooning.Count()][];
                    Parallel.For(0, boostTreeArrPooning.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, boostTreeArrPooning[j], bestLevel[j]);
                    });

                    for (int j = 0; j < testingLabel[0].Count(); j++)
                    {
                        double tmp = 0;
                        for (int k = 0; k < boostTreeArrPooning.Count(); k++)
                        {
                            estimatedLabels[i][k][j] = tmp + tmpLabel[k][j];
                            tmp += tmpLabel[k][j];
                        }
                    }
                });
            }
            else
            {
                for (int i = 0; i < testingLabel.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[_rc.dim];//testing_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < _rc.dim; j++)//testing_dt[0].Count()
                        point[j] = double.Parse(testingDt[i][j].ToString());
                    double[][] tmpLabel = new double[boostTreeArrPooning.Count()][];
                    for (int j = 0; j < boostTreeArrPooning.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, boostTreeArrPooning[j], bestLevel[j]);
                    }

                    for (int j = 0; j < testingLabel[0].Count(); j++)
                    {
                        double tmp = 0;
                        for (int k = 0; k < boostTreeArrPooning.Count(); k++)
                        {
                            estimatedLabels[i][k][j] = tmp + tmpLabel[k][j];
                            tmp += tmpLabel[k][j];
                        }
                    }
                }
            }

            if (normLpType == 2)
            {
                for (int k = 0; k < _rc.boostNum; k++)
                {
                    for (int j = 0; j < testingLabel[0].Count(); j++)
                        for (int i = 0; i < testingLabel.Count(); i++)
                            error[k] += (estimatedLabels[i][k][j] - testingLabel[i][j]) * (estimatedLabels[i][k][j] - testingLabel[i][j]);
                    error[k] = Math.Sqrt(error[k]);
                }

            }
            else if (normLpType == 0 && estimatedLabels[0][0].Count() == 1)//+-1 labels
            {
                for (int k = 0; k < _rc.boostNum; k++)
                {
                    for (int i = 0; i < testingLabel.Count(); i++)
                    {
                        if ((estimatedLabels[i][k][0] * testingLabel[i][0]) <= 0)
                            error[k] += 1;
                    }
                }
            }
            else if (normLpType == -2 && estimatedLabels[0][0].Count() == 1)//+-1 labels + BER
            {
                double[] NclassA = new double[_rc.boostNum];
                double[] NclassB = new double[_rc.boostNum];
                double[] NMissclassA = new double[_rc.boostNum];
                double[] NMissclassB = new double[_rc.boostNum];

                for (int k = 0; k < _rc.boostNum; k++)
                {
                    for (int i = 0; i < testingLabel.Count(); i++)
                    {
                        if (testingLabel[i][0] == 1)
                        {
                            NclassA[k] += 1;
                            if (estimatedLabels[i][k][0] <= 0)
                                NMissclassA[k] += 1;
                        }
                        if (testingLabel[i][0] == -1)
                        {
                            NclassB[k] += 1;
                            if (estimatedLabels[i][k][0] >= 0)
                                NMissclassB[k] += 1;
                        }
                    }
                    error[k] = 0.5 * ((NMissclassA[k] / NclassA[k]) + (NMissclassB[k] / NclassB[k]));
                }
            }
            //else if (NormLPType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            //{
            //    //adjust labels to simplex
            //    adjustlabels2simplex4(estimatedLabels);

            //    for (int i = 0; i < Data_Lables.Count(); i++)
            //    {
            //        if (0.00001 < normPoint3d(estimatedLabels[i], Data_Lables[i]))
            //            error += 1;
            //    }
            //}

        }


        //end old version no testarr

        private double testDecisionTree(List<int> testingArr, double[][] dataTable, double[][] dataLables, List<GeoWave> treeOrderedById, double normThreshold, int normLpType)
        {
            treeOrderedById = treeOrderedById.OrderBy(o => o.ID).ToList();

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[testingArr.Count()][];
            for (int i = 0; i < testingArr.Count(); i++)
                estimatedLabels[i] = new double[dataLables[0].Count()];
            Helpers.applyFor(0, testingArr.Count(), i =>
            {
                estimatedLabels[i] = askTreeMeanVal(dataTable[testingArr[i]], treeOrderedById, normThreshold);
            });
            


            double error = 0;
            switch (normLpType)
            {
                case 2:
                    for (int j = 0; j < dataLables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                        {
                            error += (estimatedLabels[i][j] - dataLables[testingArr[i]][j]) * (estimatedLabels[i][j] - dataLables[testingArr[i]][j]);
                        }
                    error = Math.Sqrt(error / Convert.ToDouble(testingArr.Count()));
                    break;
                case 1:
                    for (int j = 0; j < dataLables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                        {
                            error += Math.Abs(estimatedLabels[i][j] - dataLables[testingArr[i]][j]);
                        }
                    break;
                case -1:
                    List<double> errList = new List<double>();
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        double tmp = 0;
                        for (int j = 0; j < dataLables[0].Count(); j++)
                        {
                            tmp += Math.Abs(estimatedLabels[i][j] - dataLables[testingArr[i]][j]);
                        }
                        errList.Add(tmp);
                    }
                    error = errList.Max();
                    break;
                default:
                    if (normLpType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
                    {
                        for (int i = 0; i < testingArr.Count(); i++)
                        {
                            if ((estimatedLabels[i][0] * dataLables[testingArr[i]][0]) <= 0)
                                error += 1;
                        }
                    }
                    else if (normLpType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
                    {
                        //adjust labels to simplex
                        adjustlabels2Simplex4(estimatedLabels);

                        for (int i = 0; i < testingArr.Count(); i++)
                        {
                            if (0.00001 < normPoint3D(estimatedLabels[i], dataLables[testingArr[i]]))
                                error += 1;
                        }
                    }
                    else if (normLpType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
                    {
                        double NclassA = 0;
                        double NclassB = 0;
                        double NMissclassA = 0;
                        double NMissclassB = 0;

                        for (int i = 0; i < testingArr.Count(); i++)
                        {
                            if (dataLables[testingArr[i]][0] == 1)
                            {
                                NclassA += 1;
                                if (estimatedLabels[i][0] <= 0)
                                    NMissclassA += 1;
                            }
                            if (dataLables[testingArr[i]][0] != -1) continue;
                            NclassB += 1;
                            if (estimatedLabels[i][0] >= 0)
                                NMissclassB += 1;
                        }
                        error = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
                    }
                    break;
            }

            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double testDecisionTreeWithProoning(List<int> testingArr, double[][] dataTable, double[][] dataLables, List<GeoWave> treeOrderedById, int topLevel, int normLpType)
        {
            treeOrderedById = treeOrderedById.OrderBy(o => o.ID).ToList();

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[testingArr.Count()][];
            for (int i = 0; i < testingArr.Count(); i++)
                estimatedLabels[i] = new double[dataLables[0].Count()];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, testingArr.Count(), i =>
                {
                    estimatedLabels[i] = askTreeMeanValAtLevel(dataTable[testingArr[i]], treeOrderedById, topLevel);
                });
            }
            else
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    estimatedLabels[i] = askTreeMeanValAtLevel(dataTable[testingArr[i]], treeOrderedById, topLevel);
                }
            }

            double error = 0;
            if (normLpType == 2)
            {
                for (int j = 0; j < dataLables[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - dataLables[testingArr[i]][j]) * (estimatedLabels[i][j] - dataLables[testingArr[i]][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(testingArr.Count()));
            }
            else if (normLpType == 1)
            {
                for (int j = 0; j < dataLables[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        error += Math.Abs(estimatedLabels[i][j] - dataLables[testingArr[i]][j]);
                    }
            }
            else if (normLpType == -1)//max
            {
                List<double> errList = new List<double>();
                double tmp = 0;
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    tmp = 0;
                    for (int j = 0; j < dataLables[0].Count(); j++)
                    {
                        tmp += Math.Abs(estimatedLabels[i][j] - dataLables[testingArr[i]][j]);
                    }
                    errList.Add(tmp);
                }
                error = errList.Max();
            }
            else if (normLpType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * dataLables[testingArr[i]][0]) <= 0)
                        error += 1;
                }
            }
            else if (normLpType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2Simplex4(estimatedLabels);

                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if (0.00001 < normPoint3D(estimatedLabels[i], dataLables[testingArr[i]]))
                        error += 1;
                }
            }
            else if (normLpType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if (dataLables[testingArr[i]][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (dataLables[testingArr[i]][0] == -1)
                    {
                        NclassB += 1;
                        if (estimatedLabels[i][0] >= 0)
                            NMissclassB += 1;
                    }
                }
                error = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
            }

            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double testDecisionTreeRF(List<int> testingArr, double[][] dataTable, double[][] dataLables, List<GeoWave>[] rFdecTreeArr , double normThreshold, int normLpType)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, rFdecTreeArr.Count(), i =>
                {
                    rFdecTreeArr[i] = rFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < rFdecTreeArr.Count(); i++)
                {
                    rFdecTreeArr[i] = rFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }                         

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[testingArr.Count()][];
            for (int i = 0; i < testingArr.Count(); i++)
                estimatedLabels[i] = new double[dataLables[0].Count()];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, testingArr.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[_rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < _rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(dataTable[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[rFdecTreeArr.Count()][];
                    Parallel.For(0, rFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanVal(point, rFdecTreeArr[j], normThreshold);
                    });

                    for (int j = 0; j < dataLables[0].Count(); j++)
                        for (int k = 0; k < rFdecTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(rFdecTreeArr.Count());
                });
            }
            else
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[_rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < _rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(dataTable[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[rFdecTreeArr.Count()][];

                    for (int j = 0; j < rFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanVal(point, rFdecTreeArr[j], normThreshold);
                    }

                    for (int j = 0; j < dataLables[0].Count(); j++)
                        for (int k = 0; k < rFdecTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(rFdecTreeArr.Count());
                }
            } 

            double error = 0;
            if (normLpType == 2)
            {
                for (int j = 0; j < dataLables[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - dataLables[testingArr[i]][j]) * (estimatedLabels[i][j] - dataLables[testingArr[i]][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(testingArr.Count()));
            }
            else if (normLpType == 1 )//L1
            {
                for (int j = 0; j < dataLables[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        error += Math.Abs(estimatedLabels[i][j] - dataLables[testingArr[i]][j]);
                    }
            }
            else if (normLpType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * dataLables[testingArr[i]][0]) <= 0)
                        error += 1;
                }
            }
            else if (normLpType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2Simplex4(estimatedLabels);

                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if (0.00001 < normPoint3D(estimatedLabels[i], dataLables[testingArr[i]]))
                        error += 1;
                }
            }
            else if (normLpType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if (dataLables[testingArr[i]][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (dataLables[testingArr[i]][0] == -1)
                    {
                        NclassB += 1;
                        if (estimatedLabels[i][0] >= 0)
                            NMissclassB += 1;
                    }
                }
                error = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
            }

            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double testDecisionTreeManyRFNormNbound(List<int> testingArr, double[][] dataTable, double[][] dataLables, List<GeoWave>[] rFdecTreeArr, double normThreshold, int boundLevel, int normLpType)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, rFdecTreeArr.Count(), i =>
                {
                    rFdecTreeArr[i] = rFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < rFdecTreeArr.Count(); i++)
                {
                    rFdecTreeArr[i] = rFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }                         

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[testingArr.Count()][];
            for (int i = 0; i < testingArr.Count(); i++)
                estimatedLabels[i] = new double[dataLables[0].Count()];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, testingArr.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[_rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < _rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(dataTable[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[rFdecTreeArr.Count()][];
                    Parallel.For(0, rFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanValBoundLevel(point, rFdecTreeArr[j], normThreshold,boundLevel);
                    });

                    for (int j = 0; j < dataLables[0].Count(); j++)
                        for (int k = 0; k < rFdecTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(rFdecTreeArr.Count());
                });
            }
            else
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[_rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < _rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(dataTable[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[rFdecTreeArr.Count()][];

                    for (int j = 0; j < rFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanValBoundLevel(point, rFdecTreeArr[j], normThreshold, boundLevel);
                    }

                    for (int j = 0; j < dataLables[0].Count(); j++)
                        for (int k = 0; k < rFdecTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(rFdecTreeArr.Count());
                }
            } 

            double error = 0;
            if (normLpType == 2)
            {
                for (int j = 0; j < dataLables[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - dataLables[testingArr[i]][j]) * (estimatedLabels[i][j] - dataLables[testingArr[i]][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(testingArr.Count()));
            }
            else if (normLpType == 1)//L1
            {
                for (int j = 0; j < dataLables[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        error += Math.Abs(estimatedLabels[i][j] - dataLables[testingArr[i]][j]);
                    }
            }
            else if (normLpType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * dataLables[testingArr[i]][0]) <= 0)
                        error += 1;
                }
            }
            else if (normLpType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2Simplex4(estimatedLabels);

                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if (0.00001 < normPoint3D(estimatedLabels[i], dataLables[testingArr[i]]))
                        error += 1;
                }
            }
            else if (normLpType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if (dataLables[testingArr[i]][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (dataLables[testingArr[i]][0] == -1)
                    {
                        NclassB += 1;
                        if (estimatedLabels[i][0] >= 0)
                            NMissclassB += 1;
                    }
                }
                error = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
            }

            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double[] testDecisionTreeManyRFnew(List<int> testingArr, double[][] dataTable, 
                double[][] dataLables, List<GeoWave>[] rFdecTreeArr, double normThreshold, int normLpType)
        {
            int rfSize = rFdecTreeArr.Count(); // count trees in RF
            int testSize = testingArr.Count(); //count test points
            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[rfSize][][];//[trees in estimation][ label_index ][ label_value ] 
            for (int treeID = 0; treeID < rfSize; treeID++)
            {
                estimatedLabels[treeID] = new double[testSize][];
                for (int labelID = 0; labelID < testSize; labelID++)
                {
                    estimatedLabels[treeID][labelID] = new double[_rc.labelDim];
                }
                   
            }
            Helpers.applyFor(0, testSize, i =>
            {
                double[] point = new double[_rc.dim];
                int testPointId = testingArr[i];
                for (int j = 0; j < _rc.dim; j++)
                {
                    point[j] = double.Parse(dataTable[testPointId][j].ToString(CultureInfo.InvariantCulture));
                }

                double[][] tmpLabel = new double[rfSize][];
                Helpers.applyFor(0, rfSize, treeId =>
                {
                    tmpLabel[treeId] = askTreeMeanVal(point, rFdecTreeArr[treeId], normThreshold);
                });

                for (int j = 0; j < _rc.labelDim; j++)
                {
                    estimatedLabels[0][i][j] = tmpLabel[0][j];
                    //avg estimation by k trees
                    for (int k = 1; k < rfSize; k++)
                    {
                        estimatedLabels[k][i][j] = (Convert.ToDouble(k) / (Convert.ToDouble(k) + 1)) *
                           estimatedLabels[k - 1][i][j] + (1 / (Convert.ToDouble(k) + 1)) * tmpLabel[k][j];
                    }
                       
                }
            });

          

            double[] error = new double[rfSize];
            switch (normLpType)
            {
                case 2:
                    for (int treeId = 0; treeId < rfSize; treeId++)
                    {
                        for (int labelDim = 0; labelDim < _rc.labelDim; labelDim++)
                            for (int pointId = 0; pointId < testSize; pointId++)
                                error[treeId] += (estimatedLabels[treeId][pointId][labelDim] - dataLables[testingArr[pointId]][labelDim]) *
                                    (estimatedLabels[treeId][pointId][labelDim] - dataLables[testingArr[pointId]][labelDim]);
                        error[treeId] = Math.Sqrt(error[treeId] / Convert.ToDouble(testSize));
                    }
                    break;
                case 1:
                    for (int k = 0; k < rFdecTreeArr.Count(); k++)
                    {
                        for (int j = 0; j < dataLables[0].Count(); j++)
                            for (int i = 0; i < testingArr.Count(); i++)
                                error[k] += Math.Abs(estimatedLabels[k][i][j] - dataLables[testingArr[i]][j]);
                    }
                    break;
                default:
                    if (normLpType == -2 && estimatedLabels[0][0].Count() == 1)//+-1 labels + BER
                    {
                        double threshVal = 0.5 * (Form1.upper_label + Form1.lower_label);
                        for (int k = 0; k < rFdecTreeArr.Count(); k++)
                        {
                            double NclassA = 0;
                            double NclassB = 0;
                            double NMissclassA = 0;
                            double NMissclassB = 0;

                            for (int j = 0; j < dataLables[0].Count(); j++)
                                for (int i = 0; i < testingArr.Count(); i++)
                                {
                                    if (dataLables[testingArr[i]][j] == Form1.upper_label)
                                    {
                                        NclassA += 1;
                                        if (estimatedLabels[k][i][j] <= threshVal)
                                            NMissclassA += 1;
                                    }
                                    if (dataLables[testingArr[i]][j] == Form1.lower_label)
                                    {
                                        NclassB += 1;
                                        if (estimatedLabels[k][i][j] >= threshVal)
                                            NMissclassB += 1;
                                    }
                                }
                            error[k] = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
                        }
                    }
                    break;
            }
            return error;

        }

        private double[] testDecisionTreeManyRFNoVoting(List<int> testingArr, double[][] dataTable, double[][] dataLables, List<GeoWave>[] rFdecTreeArr, double normThreshold, int normLpType)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, rFdecTreeArr.Count(), i =>
                {
                    rFdecTreeArr[i] = rFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < rFdecTreeArr.Count(); i++)
                {
                    rFdecTreeArr[i] = rFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[rFdecTreeArr.Count()][][];//num of trees, label index, label values (or value in most cases)
            for (int i = 0; i < rFdecTreeArr.Count(); i++)
            {
                estimatedLabels[i] = new double[testingArr.Count()][];
                for (int j = 0; j < testingArr.Count(); j++)
                    estimatedLabels[i][j] = new double[dataLables[0].Count()];
            }

            if (Form1.rumPrallel)
            {
                Parallel.For(0, testingArr.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[_rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < _rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(dataTable[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[rFdecTreeArr.Count()][];
                    Parallel.For(0, rFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanVal(point, rFdecTreeArr[j], normThreshold);
                    });

                    for (int j = 0; j < dataLables[0].Count(); j++)
                    {
                        for (int k = 0; k < rFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = tmpLabel[k][j];
                    }
                });
            }
            else
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[_rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < _rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(dataTable[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[rFdecTreeArr.Count()][];

                    for (int j = 0; j < rFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanVal(point, rFdecTreeArr[j], normThreshold);
                    }

                    for (int j = 0; j < dataLables[0].Count(); j++)
                    {
                        for (int k = 0; k < rFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = tmpLabel[k][j];
                    }
                }
            }

            double[] error = new double[rFdecTreeArr.Count()];
            if (normLpType == 2)
            {
                for (int k = 0; k < rFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < dataLables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                            error[k] += (estimatedLabels[k][i][j] - dataLables[testingArr[i]][j]) * (estimatedLabels[k][i][j] - dataLables[testingArr[i]][j]);
                    error[k] = Math.Sqrt(error[k] / Convert.ToDouble(testingArr.Count()));
                }
            }
            else if (normLpType == 1)
            {
                for (int k = 0; k < rFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < dataLables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                            error[k] += Math.Abs(estimatedLabels[k][i][j] - dataLables[testingArr[i]][j]);
                }
            }
            else if (normLpType == -2 && estimatedLabels[0][0].Count() == 1)//+-1 labels + BER
            {
                double threshVal = 0.5 * (Form1.upper_label + Form1.lower_label);
                for (int k = 0; k < rFdecTreeArr.Count(); k++)
                {
                    double NclassA = 0;
                    double NclassB = 0;
                    double NMissclassA = 0;
                    double NMissclassB = 0;

                    for (int j = 0; j < dataLables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                        {
                            if (dataLables[testingArr[i]][j] == Form1.upper_label)
                            {
                                NclassA += 1;
                                if (estimatedLabels[k][i][j] <= threshVal)
                                    NMissclassA += 1;
                            }
                            if (dataLables[testingArr[i]][j] == Form1.lower_label)
                            {
                                NclassB += 1;
                                if (estimatedLabels[k][i][j] >= threshVal)
                                    NMissclassB += 1;
                            }
                        }
                    error[k] = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
                }
            }
            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double[] testDecisionTreeManyRFbyIndex(List<int> testingArr, double[][] dataTable, double[][] dataLables, List<GeoWave>[] rFdecTreeArr, int indexThreshold, int normLpType)
        {
            List<GeoWave>[] RFdecTreeArrById = new List<GeoWave>[rFdecTreeArr.Count()];
            if (Form1.rumPrallel)
            {
                Parallel.For(0, rFdecTreeArr.Count(), i =>
                {
                    RFdecTreeArrById[i] = rFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < rFdecTreeArr.Count(); i++)
                {
                    RFdecTreeArrById[i] = rFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[rFdecTreeArr.Count()][][];//num of trees, label index, label values (or value in most cases)
            for (int i = 0; i < rFdecTreeArr.Count(); i++)
            {
                estimatedLabels[i] = new double[testingArr.Count()][];
                for (int j = 0; j < testingArr.Count(); j++)
                    estimatedLabels[i][j] = new double[dataLables[0].Count()];
            }

            if (Form1.rumPrallel)
            {
                Parallel.For(0, dataLables.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[_rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < _rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(dataTable[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[rFdecTreeArr.Count()][];
                    Parallel.For(0, rFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArrById[j], rFdecTreeArr[j][indexThreshold].norm);
                    });

                    for (int j = 0; j < dataLables[0].Count(); j++)
                    {
                        estimatedLabels[0][i][j] = tmpLabel[0][j];
                        for (int k = 1; k < rFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = (Convert.ToDouble(k) / (Convert.ToDouble(k) + 1)) * estimatedLabels[k - 1][i][j] + (1 / (Convert.ToDouble(k) + 1)) * tmpLabel[k][j];              
                    }
                });
            }
            else
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[_rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < _rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(dataTable[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[rFdecTreeArr.Count()][];

                    for (int j = 0; j < rFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArrById[j], rFdecTreeArr[j][indexThreshold].norm);
                    }


                    for (int j = 0; j < dataLables[0].Count(); j++)
                    {
                        estimatedLabels[0][i][j] = tmpLabel[0][j];
                        for (int k = 1; k < rFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = (Convert.ToDouble(k) / (Convert.ToDouble(k) + 1)) * estimatedLabels[k - 1][i][j] + (1 / (Convert.ToDouble(k) + 1)) * tmpLabel[k][j];    
                    }
                }
            }

            double[] error = new double[rFdecTreeArr.Count()];
            if (normLpType == 2)
            {
                for (int k = 0; k < rFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < dataLables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                            error[k] += (estimatedLabels[k][i][j] - dataLables[testingArr[i]][j]) * (estimatedLabels[k][i][j] - dataLables[testingArr[i]][j]);
                    error[k] = Math.Sqrt(error[k] / Convert.ToDouble(testingArr.Count()));
                }
            }
            if (normLpType == 1)
            {
                for (int k = 0; k < rFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < dataLables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                            error[k] += Math.Abs(estimatedLabels[k][i][j] - dataLables[testingArr[i]][j]);
                }
            }
            else if (normLpType == -2 && estimatedLabels[0][0].Count() == 1)//+-1 labels + BER
            {
                double threshVal = 0.5 * (Form1.upper_label + Form1.lower_label);
                for (int k = 0; k < rFdecTreeArr.Count(); k++)
                {
                    double NclassA = 0;
                    double NclassB = 0;
                    double NMissclassA = 0;
                    double NMissclassB = 0;

                    for (int j = 0; j < dataLables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                        {
                            if (dataLables[testingArr[i]][j] == Form1.upper_label)
                            {
                                NclassA += 1;
                                if (estimatedLabels[k][i][j] <= threshVal)
                                    NMissclassA += 1;
                            }
                            if (dataLables[testingArr[i]][j] == Form1.lower_label)
                            {
                                NclassB += 1;
                                if (estimatedLabels[k][i][j] >= threshVal)
                                    NMissclassB += 1;
                            }
                        }
                    error[k] = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
                }
            }
            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }
        
        private double testDecisionTreeRF(List<int> testingArr, double[][] dataTable, double[][] dataLables, List<GeoWave>[] rFdecTreeArr, int topLevel, int normLpType)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, rFdecTreeArr.Count(), i =>
                {
                    rFdecTreeArr[i] = rFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < rFdecTreeArr.Count(); i++)
                {
                    rFdecTreeArr[i] = rFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[testingArr.Count()][];
            for (int i = 0; i < testingArr.Count(); i++)
                estimatedLabels[i] = new double[dataLables[0].Count()];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, testingArr.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[_rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < _rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(dataTable[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[rFdecTreeArr.Count()][];
                    Parallel.For(0, rFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, rFdecTreeArr[j], topLevel);
                    });

                    for (int j = 0; j < dataLables[0].Count(); j++)
                        for (int k = 0; k < rFdecTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(rFdecTreeArr.Count());
                });
            }
            else
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[_rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < _rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(dataTable[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[rFdecTreeArr.Count()][];

                    for (int j = 0; j < rFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, rFdecTreeArr[j], topLevel);
                    }

                    for (int j = 0; j < dataLables[0].Count(); j++)
                        for (int k = 0; k < rFdecTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(rFdecTreeArr.Count());
                }
            }

            double error = 0;
            if (normLpType == 2)
            {
                for (int j = 0; j < dataLables[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - dataLables[testingArr[i]][j]) * (estimatedLabels[i][j] - dataLables[testingArr[i]][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(testingArr.Count()));
            }
            if (normLpType == 1)
            {
                for (int j = 0; j < dataLables[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        error += Math.Abs(estimatedLabels[i][j] - dataLables[testingArr[i]][j]);
                    }
            }
            else if (normLpType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * dataLables[testingArr[i]][0]) <= 0)
                        error += 1;
                }
            }
            else if (normLpType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2Simplex4(estimatedLabels);

                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if (0.00001 < normPoint3D(estimatedLabels[i], dataLables[testingArr[i]]))
                        error += 1;
                }
            }
            else if (normLpType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if (dataLables[testingArr[i]][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (dataLables[testingArr[i]][0] == -1)
                    {
                        NclassB += 1;
                        if (estimatedLabels[i][0] >= 0)
                            NMissclassB += 1;
                    }
                }
                error = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
            }

            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double[] testDecisionTreeManyRF(List<int> testingArr, double[][] dataTable, double[][] dataLables, List<GeoWave>[] rFdecTreeArr, int topLevel, int normLpType)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, rFdecTreeArr.Count(), i =>
                {
                    rFdecTreeArr[i] = rFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < rFdecTreeArr.Count(); i++)
                {
                    rFdecTreeArr[i] = rFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[rFdecTreeArr.Count()][][];//num of trees, label index, label values (or value in most cases)
            for (int i = 0; i < rFdecTreeArr.Count(); i++)
            {
                estimatedLabels[i] = new double[testingArr.Count()][];
                for (int j = 0; j < testingArr.Count(); j++)
                    estimatedLabels[i][j] = new double[dataLables[0].Count()];
            }

            if (Form1.rumPrallel)
            {
                Parallel.For(0, testingArr.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[_rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < _rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(dataTable[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[rFdecTreeArr.Count()][];
                    Parallel.For(0, rFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, rFdecTreeArr[j], topLevel);
                    });

                    for (int j = 0; j < dataLables[0].Count(); j++)
                    {
                        estimatedLabels[0][i][j] = tmpLabel[0][j];
                        for (int k = 1; k < rFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = (Convert.ToDouble(k) / (Convert.ToDouble(k) + 1)) * estimatedLabels[k - 1][i][j] + (1 / (Convert.ToDouble(k) + 1)) * tmpLabel[k][j];    
                    }
                });
            }
            else
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[_rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < _rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(dataTable[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[rFdecTreeArr.Count()][];

                    for (int j = 0; j < rFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, rFdecTreeArr[j], topLevel);
                    }

                    for (int j = 0; j < dataLables[0].Count(); j++)
                    {
                        estimatedLabels[0][i][j] = tmpLabel[0][j];
                        for (int k = 1; k < rFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = (Convert.ToDouble(k) / (Convert.ToDouble(k) + 1)) * estimatedLabels[k - 1][i][j] + (1 / (Convert.ToDouble(k) + 1)) * tmpLabel[k][j];    
                    }
                }
            }

            double[] error = new double[rFdecTreeArr.Count()];
            if (normLpType == 2)
            {
                for (int k = 0; k < rFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < dataLables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                            error[k] += (estimatedLabels[k][i][j] - dataLables[testingArr[i]][j]) * (estimatedLabels[k][i][j] - dataLables[testingArr[i]][j]);
                    error[k] = Math.Sqrt(error[k] / Convert.ToDouble(testingArr.Count()));
                }
            }
            else if (normLpType == 1)
            {
                for (int k = 0; k < rFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < dataLables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                            error[k] += Math.Abs(estimatedLabels[k][i][j] - dataLables[testingArr[i]][j]);
                }
            }
            else if (normLpType == -2 && estimatedLabels[0][0].Count() == 1)//+-1 labels + BER
            {
                double threshVal = 0.5 * (Form1.upper_label + Form1.lower_label);
                for (int k = 0; k < rFdecTreeArr.Count(); k++)
                {
                    double NclassA = 0;
                    double NclassB = 0;
                    double NMissclassA = 0;
                    double NMissclassB = 0;
                    
                    for (int j = 0; j < dataLables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                        {
                            if (dataLables[testingArr[i]][j] == Form1.upper_label)
                            {
                                NclassA += 1;
                                if (estimatedLabels[k][i][j] <= threshVal)
                                    NMissclassA += 1;
                            }
                            if (dataLables[testingArr[i]][j] == Form1.lower_label)
                            {
                                NclassB += 1;
                                if (estimatedLabels[k][i][j] >= threshVal)
                                    NMissclassB += 1;
                            }                        
                        }
                    error[k] = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
                }                
            }
            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double[] testDecisionTreeManyRFNoVoting(List<int> testingArr, double[][] dataTable, double[][] dataLables, List<GeoWave>[] rFdecTreeArr, int topLevel, int normLpType)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, rFdecTreeArr.Count(), i =>
                {
                    rFdecTreeArr[i] = rFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < rFdecTreeArr.Count(); i++)
                {
                    rFdecTreeArr[i] = rFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[rFdecTreeArr.Count()][][];//num of trees, label index, label values (or value in most cases)
            for (int i = 0; i < rFdecTreeArr.Count(); i++)
            {
                estimatedLabels[i] = new double[testingArr.Count()][];
                for (int j = 0; j < testingArr.Count(); j++)
                    estimatedLabels[i][j] = new double[dataLables[0].Count()];
            }

            if (Form1.rumPrallel)
            {
                Parallel.For(0, testingArr.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[_rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < _rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(dataTable[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[rFdecTreeArr.Count()][];
                    Parallel.For(0, rFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, rFdecTreeArr[j], topLevel);
                    });

                    for (int j = 0; j < dataLables[0].Count(); j++)
                    {
                        for (int k = 0; k < rFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = tmpLabel[k][j];
                    }
                });
            }
            else
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[_rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < _rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(dataTable[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[rFdecTreeArr.Count()][];

                    for (int j = 0; j < rFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, rFdecTreeArr[j], topLevel);
                    }

                    for (int j = 0; j < dataLables[0].Count(); j++)
                    {
                        for (int k = 0; k < rFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = tmpLabel[k][j];
                    }
                }
            }

            double[] error = new double[rFdecTreeArr.Count()];
            if (normLpType == 2)
            {
                for (int k = 0; k < rFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < dataLables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                            error[k] += (estimatedLabels[k][i][j] - dataLables[testingArr[i]][j]) * (estimatedLabels[k][i][j] - dataLables[testingArr[i]][j]);
                    error[k] = Math.Sqrt(error[k] / Convert.ToDouble(testingArr.Count()));
                }
            }
            if (normLpType == 1)
            {
                for (int k = 0; k < rFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < dataLables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                            error[k] += Math.Abs(estimatedLabels[k][i][j] - dataLables[testingArr[i]][j]);
                }
            }
            else if (normLpType == -2 && estimatedLabels[0][0].Count() == 1)//+-1 labels + BER
            {
                double threshVal = 0.5 * (Form1.upper_label + Form1.lower_label);
                for (int k = 0; k < rFdecTreeArr.Count(); k++)
                {
                    double NclassA = 0;
                    double NclassB = 0;
                    double NMissclassA = 0;
                    double NMissclassB = 0;

                    for (int j = 0; j < dataLables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                        {
                            if (dataLables[testingArr[i]][j] == Form1.upper_label)
                            {
                                NclassA += 1;
                                if (estimatedLabels[k][i][j] <= threshVal)
                                    NMissclassA += 1;
                            }
                            if (dataLables[testingArr[i]][j] == Form1.lower_label)
                            {
                                NclassB += 1;
                                if (estimatedLabels[k][i][j] >= threshVal)
                                    NMissclassB += 1;
                            }
                        }
                    error[k] = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
                }
            }
            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private void testDecisionTreeBoostingProoning(List<int> testingArr, double[][] testingDt, double[][] testingLabel, List<GeoWave>[] boostTreeArrPooning, int[] bestLevel, int normLpType, double[] error)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, boostTreeArrPooning.Count(), i =>
                {
                    boostTreeArrPooning[i] = boostTreeArrPooning[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < boostTreeArrPooning.Count(); i++)
                {
                    boostTreeArrPooning[i] = boostTreeArrPooning[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[testingArr.Count()][][];//testing_label
            for (int i = 0; i < testingArr.Count(); i++)//testing_label
            {
                estimatedLabels[i] = new double[_rc.boostNum][];
                for (int j = 0; j < _rc.boostNum; j++)
                    estimatedLabels[i][j] = new double[testingLabel[0].Count()];
            }


            if (Form1.rumPrallel)
            {
                Parallel.For(0, testingArr.Count(), i =>     //testing_label
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[_rc.dim];//testing_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < _rc.dim; j++)//testing_dt[0].Count()
                        point[j] = double.Parse(testingDt[testingArr[i]][j].ToString());//[i][j]
                    double[][] tmpLabel = new double[boostTreeArrPooning.Count()][];
                    Parallel.For(0, boostTreeArrPooning.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, boostTreeArrPooning[j], bestLevel[j]);
                    });

                    for (int j = 0; j < testingLabel[0].Count(); j++)
                    {
                        double tmp = 0;
                        for (int k = 0; k < boostTreeArrPooning.Count(); k++)
                        {
                            estimatedLabels[i][k][j] = tmp + tmpLabel[k][j];
                            tmp += tmpLabel[k][j];
                        }
                    }
                });
            }
            else
            {
                for (int i = 0; i < testingArr.Count(); i++)//testing_label
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[_rc.dim];//testing_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < _rc.dim; j++)//testing_dt[0].Count()
                        point[j] = double.Parse(testingDt[testingArr[i]][j].ToString());//[i][j]
                    double[][] tmpLabel = new double[boostTreeArrPooning.Count()][];
                    for (int j = 0; j < boostTreeArrPooning.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, boostTreeArrPooning[j], bestLevel[j]);
                    }

                    for (int j = 0; j < testingLabel[0].Count(); j++)
                    {
                        double tmp = 0;
                        for (int k = 0; k < boostTreeArrPooning.Count(); k++)
                        {
                            estimatedLabels[i][k][j] = tmp + tmpLabel[k][j];
                            tmp += tmpLabel[k][j];
                        }
                    }
                }
            }

            if (normLpType == 2)
            {
                for (int k = 0; k < _rc.boostNum; k++)
                {
                    for (int j = 0; j < testingLabel[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)//testing_label
                            error[k] += (estimatedLabels[i][k][j] - testingLabel[testingArr[i]][j]) * (estimatedLabels[i][k][j] - testingLabel[testingArr[i]][j]);//[i][j]
                    error[k] = Math.Sqrt(error[k]);
                }

            }
            else if (normLpType == 0 && estimatedLabels[0][0].Count() == 1)//+-1 labels
            {
                for (int k = 0; k < _rc.boostNum; k++)
                {
                    for (int i = 0; i < testingArr.Count(); i++)//testing_label
                    {
                        if ((estimatedLabels[i][k][0] * testingLabel[testingArr[i]][0]) <= 0)//[i][0]
                            error[k] += 1;
                    }
                }
            }
            else if (normLpType == -2 && estimatedLabels[0][0].Count() == 1)//+-1 labels + BER
            {
                double[] NclassA = new double[_rc.boostNum];
                double[] NclassB = new double[_rc.boostNum];
                double[] NMissclassA = new double[_rc.boostNum];
                double[] NMissclassB = new double[_rc.boostNum];

                for (int k = 0; k < _rc.boostNum; k++)
                {
                    for (int i = 0; i < testingLabel.Count(); i++)
                    {
                        if (testingLabel[testingArr[i]][0] == 1)//[i][0]
                        {
                            NclassA[k] += 1;
                            if (estimatedLabels[i][k][0] <= 0)
                                NMissclassA[k] += 1;
                        }
                        if (testingLabel[testingArr[i]][0] == -1)//[i][0]
                        {
                            NclassB[k] += 1;
                            if (estimatedLabels[i][k][0] >= 0)
                                NMissclassB[k] += 1;
                        }
                    }
                    error[k] = 0.5 * ((NMissclassA[k] / NclassA[k]) + (NMissclassB[k] / NclassB[k]));
                }
            }
            //else if (NormLPType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            //{
            //    //adjust labels to simplex
            //    adjustlabels2simplex4(estimatedLabels);

            //    for (int i = 0; i < Data_Lables.Count(); i++)
            //    {
            //        if (0.00001 < normPoint3d(estimatedLabels[i], Data_Lables[i]))
            //            error += 1;
            //    }
            //}

        }

        private double testDecisionTreeBoostingLearningRate(List<int> testingArr, double[][] testingDt, double[][] testingLabel, List<GeoWave>[] boostTreeArr, int normLpType, double[] maxNorms, int ntrees)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, ntrees, i =>
                {
                    boostTreeArr[i] = boostTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < ntrees; i++)
                {
                    boostTreeArr[i] = boostTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[testingArr.Count()][];
            for (int i = 0; i < testingArr.Count(); i++)
                estimatedLabels[i] = new double[testingLabel[0].Count()];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, testingArr.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[_rc.dim];//testing_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < _rc.dim; j++)//testing_dt[0].Count()
                        point[j] = double.Parse(testingDt[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[boostTreeArr.Count()][];
                    Parallel.For(0, ntrees, j =>
                    {
                        tmpLabel[j] = askTreeMeanVal(point, boostTreeArr[j], maxNorms[j]);
                    });

                    for (int j = 0; j < testingLabel[0].Count(); j++)
                        for (int k = 0; k < ntrees; k++)
                            estimatedLabels[i][j] += tmpLabel[k][j];

                });
            }
            else
            {
                for (int i = 0; i < testingLabel.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[_rc.dim];//testing_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < _rc.dim; j++)//testing_dt[0].Count()
                        point[j] = double.Parse(testingDt[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[boostTreeArr.Count()][];
                    for (int j = 0; j < ntrees; j++)
                    {
                        tmpLabel[j] = askTreeMeanVal(point, boostTreeArr[j], maxNorms[j]);
                    }

                    for (int j = 0; j < testingLabel[0].Count(); j++)
                        for (int k = 0; k < ntrees; k++)
                            estimatedLabels[i][j] += tmpLabel[k][j];
                }
            }

            double error = 0;
            if (normLpType == 2)
            {
                for (int j = 0; j < testingLabel[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - testingLabel[testingArr[i]][j]) * (estimatedLabels[i][j] - testingLabel[testingArr[i]][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(estimatedLabels.Count()));
            }
            else if (normLpType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * testingLabel[testingArr[i]][0]) <= 0)
                        error += 1;
                }
            }
            //else if (NormLPType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            //{
            //    //adjust labels to simplex
            //    adjustlabels2simplex4(estimatedLabels);

            //    for (int i = 0; i < testing_label.Count(); i++)
            //    {
            //        if (0.00001 < normPoint3d(estimatedLabels[i], testing_label[i]))
            //            error += 1;
            //    }
            //}
            else if (normLpType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if (testingLabel[testingArr[i]][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (testingLabel[testingArr[i]][0] == -1)
                    {
                        NclassB += 1;
                        if (estimatedLabels[i][0] >= 0)
                            NMissclassB += 1;
                    }
                }
                error = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
            }
            return error;
        }        
        
        private List<GeoWave>[] getsparseRf(List<GeoWave>[] rFdecTreeArr, int nwavelets)
        {
            List<GeoWave>[] sparseRF = new List<GeoWave>[rFdecTreeArr.Count()];
            bool[][] wasElementSet = new bool[rFdecTreeArr.Count()][];
            List<GeoWave>[] IDRFdecTreeArr = new List<GeoWave>[rFdecTreeArr.Count()];

            
                //FOR EACH i TREE 
                Helpers.applyFor(0, rFdecTreeArr.Count(), i =>
                {
                    IDRFdecTreeArr[i] = rFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                    wasElementSet[i] = new bool[rFdecTreeArr[i].Count];
                    sparseRF[i] = new List<GeoWave>();

                    //SET WAVELETS
                    int Loops = (rFdecTreeArr[i].Count > nwavelets) ? nwavelets : rFdecTreeArr[i].Count;//set loops = min(Nwavelets, RFdecTreeArr[j].Count); 
                    for (int j = 0; j < Loops; j++)//each wavelets (up till Loops)
                    {
                        //COPY WAVELET - IF WAS NOT COPIED BEFORE
                        if (wasElementSet[i][rFdecTreeArr[i][j].ID] == false)
                        {
                            sparseRF[i].Add(rFdecTreeArr[i][j]);
                            wasElementSet[i][rFdecTreeArr[i][j].ID] = true;
                        }

                        int parentID = rFdecTreeArr[i][j].parentID;
                        while (parentID != -1)
                        {
                            //COPY PARENT WAVELET - IF WAS NOT COPIED BEFORE
                            if (wasElementSet[i][parentID] == false)
                            {
                                sparseRF[i].Add(IDRFdecTreeArr[i][parentID]);
                                wasElementSet[i][parentID] = true;
                            }
                            parentID = IDRFdecTreeArr[i][parentID].parentID;
                        }
                    }

                    //SORT
                    sparseRF[i] = sparseRF[i].OrderByDescending(o => o.norm).ToList();

                    Dictionary<int, int> IDmap = new Dictionary<int, int>();//old ID, new ID

                    //NULLIFY CHILDREN OF REMOVED WAVELETS
                    for (int j = 0; j < sparseRF[i].Count; j++)
                    {
                        IDmap.Add(sparseRF[i][j].ID, j);
                        sparseRF[i][j].ID = j;
                        
                        if (sparseRF[i][j].child0 != -1 && wasElementSet[i][sparseRF[i][j].child0] == false)
                            sparseRF[i][j].child0 = -1;
                        if (sparseRF[i][j].child1 != -1 && wasElementSet[i][sparseRF[i][j].child1] == false)
                            sparseRF[i][j].child1 = -1;
                    }

                    //SET NEW ID
                    for (int j = 0; j < sparseRF[i].Count; j++)
                    {
                        int newID;
                        if (IDmap.TryGetValue(sparseRF[i][j].child0, out newID))
                            sparseRF[i][j].child0 = newID;
                        if (IDmap.TryGetValue(sparseRF[i][j].child1, out newID))
                            sparseRF[i][j].child1 = newID;
                    }

                });
         
            return sparseRF;
        }        
        
        private double getgeowaveNorm(List<GeoWave> tmpTreeOrderedByNorm, int nwavelets, int normSecond, int orderTau)
        {
            double norm = 0;
            if (normSecond == 0)
                return 1.0 * nwavelets;// I dont add +1 because if I root estimation I want to give norm 0
            else if (orderTau == 1)
            {
                for (int i = 0; i <= nwavelets; i++)
                    norm += tmpTreeOrderedByNorm[i].norm;
                return norm;            
            }
            else if (orderTau == 2)
            {
                for (int i = 0; i <= nwavelets; i++)
                    norm += (tmpTreeOrderedByNorm[i].norm) * (tmpTreeOrderedByNorm[i].norm);
                return Math.Sqrt(norm);
            }
            else
            {
                for (int i = 0; i <= nwavelets; i++)
                    norm += Math.Pow(tmpTreeOrderedByNorm[i].norm, orderTau);
                return Math.Pow(norm, 1 / Convert.ToDouble(orderTau));
            }
        }
        //experiment build tree pathes
        struct PointPaths
        {
            public int testPointId { set; get; }
            public List<int>[] paths { get; set; } //tree id, path of nodes ids
        }


        private List<PointPaths> createPathesByForest(List<GeoWave>[] forest, List<int> testingArr, double[][] Data_table)
        {
            List<PointPaths> listPointPathes=new List<PointPaths>();
          
            //foreach (int testId in testingArr)
           // Helpers.applyFor(0, testingArr.Count, i =>
           foreach (int testId in testingArr)
            {
                //int testId = testingArr[i];
                double[] point = Data_table[testId];
                PointPaths pointPaths = new PointPaths
                {
                    testPointId = testId,
                    paths = new List<int>[forest.Length]
                };
                for (int tInd = 0; tInd < forest.Length; tInd++)
                {
                    List<GeoWave> treeOrderedById = forest[tInd];
                    pointPaths.paths[tInd] = new List<int>();
                    int parentIndex = 0;
                    pointPaths.paths[tInd].Add(parentIndex);
                    while (treeOrderedById[parentIndex].typeTransformed != DecicionTree.SplitType.NotSplitted)
                    {

                        GeoWave parent = treeOrderedById[parentIndex];
                        int child0Ind = treeOrderedById[parentIndex].child0;
                        int child1Ind = treeOrderedById[parentIndex].child1;
                        GeoWave child0 = treeOrderedById[child0Ind];
                        GeoWave child1 = treeOrderedById[child1Ind];
                        //exit from loop
                        //if (child0 == null && child1 == null) break;
                        GeoWave selectedChild = null;
                        double[] transformedPoint;
                        switch (parent.typeTransformed)
                        {
                            case DecicionTree.SplitType.LocalPca:
                                transformedPoint = parent.localPca.Transform(point);
                                selectedChild = transformedPoint[child0.dimIndex] < child0.upperSplitValue
                                    ? child0
                                    : child1;
                                break;
                            case DecicionTree.SplitType.MainAxes:
                                transformedPoint = point;
                                selectedChild = transformedPoint[child0.dimIndex] < child0.upperSplitValue
                                    ? child0
                                    : child1;
                                break;
                            case DecicionTree.SplitType.Categorical:
                                transformedPoint = point;
                                //in catecorical split "upperSpliteValue" is just a categorical value
                                selectedChild = transformedPoint[child0.dimIndex] == child0.upperSplitValue
                                    ? child0
                                    : child1;
                                break;
                            case DecicionTree.SplitType.DiffMaps5Percent:
                            case DecicionTree.SplitType.DiffMaps1Percent:
                            case DecicionTree.SplitType.DiffMapsHalfPercent:
                                transformedPoint = point;
                                selectedChild = choseChildByKnearst(child0, child1, point, parent.typeTransformed);
                                break;
                            default:
                                transformedPoint = null;
                                break;
                        }
                        if (transformedPoint == null)
                        {
                            throw new Exception("analizer.cs *******askTreeMeanValue, transformed point error!!");
                        }

                        pointPaths.paths[tInd].Add(selectedChild.ID);
                        parentIndex = selectedChild.ID;

                    }

                }
                listPointPathes.Add(pointPaths);
                Debug.WriteLine("***Path points ready " +
                                Math.Round(100*(double) listPointPathes.Count/testingArr.Count, 2) + "%");
            }
        //});

            return listPointPathes;
        }

        private double errorByPathAndNormTreshold(List<PointPaths> pathes,List<GeoWave>[] rFdecTreeArr,
                                                                    double[][] Data_Lables, double treshold)
        {
            double[] errArr = new double[pathes.Count()];
            //go over all test points
            Helpers.applyFor(0, errArr.Length, i =>
           // for (int i = 0; i < errArr.Length; i++)
            {
                PointPaths singlePP = pathes[i];
                if (singlePP.testPointId == 0)
                {
                    Debug.WriteLine("ID 0 test point");
                }
                double realLabel = Data_Lables[singlePP.testPointId][0];
                double[] tmpVal = {0};
                int treesInvolved = 0;
                //estimate each tree
                Helpers.applyFor(0, _rc.rfNum, j =>
                {
                    List<int> jPath = singlePP.paths[j];
                    List<GeoWave> jTree = rFdecTreeArr[j];
                    double stepSum =
                        jPath.Select(id => jTree[id])
                            .Where(x => x.norm >= treshold || x.level==0)
                            .Select(x => x.meanDiffFromParent)
                            .Sum();
                    tmpVal[0] += stepSum;
                    if (Math.Abs(stepSum) > Double.Epsilon) treesInvolved++;
                });
                if (treesInvolved == 0) treesInvolved++;
                tmpVal[0] /= treesInvolved;
                switch (_rc.learningType)
                {
                    case recordConfig.LearnigType.Regression:
                        errArr[i] = (tmpVal[0] - realLabel) * (tmpVal[0] - realLabel);
                        break;
                    case recordConfig.LearnigType.BinaryClassification:
                        errArr[i] = checkBinaryClassificationError(realLabel, tmpVal[0]); // 1 if wrong predicted, 0 it true predicted
                        break;
                    case recordConfig.LearnigType.ManyClassification:
                        throw new Exception("Still not realized estimation for many classification classes!!!");
                        break;
                    case recordConfig.LearnigType.L1:
                        errArr[i] = Math.Abs(tmpVal[0] - realLabel);
                        break;
                }
            });

            //calculate average error of all points
            double errorAvg = 0;
            for (int i = 0; i < errArr.Count(); i++)
            {
               
                errorAvg += errArr[i]; //sum of missclassifications in binary case 
            }   
            switch (_rc.learningType)
            {
                case recordConfig.LearnigType.Regression:          
                    errorAvg = Math.Sqrt(errorAvg / Convert.ToDouble(errArr.Count()));
                    break;
                case recordConfig.LearnigType.BinaryClassification:
                    errorAvg = 100*errorAvg/Convert.ToDouble(errArr.Count()); // % of misclassification
                    break;
                case recordConfig.LearnigType.L1:
                    errorAvg = errorAvg / Convert.ToDouble(errArr.Count());
                    break;
            }
           
            return errorAvg;
        }
        //end experiment
        private int checkBinaryClassificationError(double realVal, double predicted)
        {
            int wasError;
            if (_rc.learningType != recordConfig.LearnigType.BinaryClassification)
            {
                throw new Exception("ERROR! Binary estimation used in regression!!!");
            }
            if ((predicted >= _rc.midClassValue && realVal < _rc.midClassValue) || (predicted <= _rc.midClassValue &&  realVal > _rc.midClassValue))
            {
                wasError = 1;
            }
            else
            {
                wasError = 0;
            }
            return wasError;
        }
        private double[] askTreeMeanValTransformed(double[] point, List<GeoWave> treeOrderedById, double normThreshold)
        {
            double[] MeanValue = new double[_rc.labelDim];
            double[] zeroMean = new double[_rc.labelDim];
            //SET THE ROOT MEAN VAL
            treeOrderedById[0].MeanValue.CopyTo(MeanValue, 0);
            int parentIndex = 0;
            while (treeOrderedById[parentIndex].typeTransformed != DecicionTree.SplitType.NotSplitted)
            {
                GeoWave parent = treeOrderedById[parentIndex];
                int child0Ind = treeOrderedById[parentIndex].child0;
                int child1Ind = treeOrderedById[parentIndex].child1;
                GeoWave child0 = (child0Ind == -1) ? null : treeOrderedById[child0Ind];
                GeoWave child1 = (child1Ind == -1) ? null : treeOrderedById[child1Ind];
                //exit from loop
                //if (child0 == null && child1 == null) break;
                GeoWave selectedChild = null;
                double[] transformedPoint;
                switch (parent.typeTransformed)
                {
                    case DecicionTree.SplitType.LocalPca:
                        transformedPoint = parent.localPca.Transform(point);
                        break;
                    case DecicionTree.SplitType.MainAxes:
                    case DecicionTree.SplitType.DiffMaps5Percent:
                    case DecicionTree.SplitType.DiffMaps1Percent:
                    case DecicionTree.SplitType.DiffMapsHalfPercent:
                        transformedPoint = point;
                        break;
                    default:
                        transformedPoint = null;
                        break;
                }
                if (transformedPoint == null)
                {
                   throw new Exception("analizer.cs *******askTreeMeanValue, transformed point error!!");
                }


                if (parent.typeTransformed != DecicionTree.SplitType.DiffMaps5Percent
                    && parent.typeTransformed != DecicionTree.SplitType.DiffMaps1Percent
                    && parent.typeTransformed != DecicionTree.SplitType.DiffMapsHalfPercent)
                {
                    // ReSharper disable once PossibleNullReferenceException
                    selectedChild = transformedPoint[child0.dimIndex] < child0.upperSplitValue ? child0 : child1;
                }
                else
                {
                    selectedChild = choseChildByKnearst(child0, child1, point, parent.typeTransformed);
                }
               
                if (!selectedChild.MeanValue.SequenceEqual(zeroMean) && normThreshold <= selectedChild.norm)
                        MeanValue[0] += (selectedChild.MeanValue[0] - parent.MeanValue[0]);
                    parentIndex = selectedChild.ID;
               
            } 
            return MeanValue;
        }

        private GeoWave choseChildByKnearst(GeoWave child0, GeoWave child1, IList<double> testPoint, DecicionTree.SplitType typeTransformed)
        {

            if (child0 == null || child1 == null)
            {
                throw new Exception("*****getChildByKnearst() error: one child is NULL");
            }
            int parentSize = child0.pointsIdArray.Count + child1.pointsIdArray.Count;
            double kPercent;
            switch (typeTransformed)
            {
                case DecicionTree.SplitType.DiffMaps5Percent:
                    kPercent = 0.05;
                    break;
                case DecicionTree.SplitType.DiffMaps1Percent:
                    kPercent = 0.01;
                    break;
                case DecicionTree.SplitType.DiffMapsHalfPercent:
                    kPercent = 0.005;
                    break;
                default:
                    throw new Exception("Bad Split Type in kNearst!!");
                    
            }
          
            //start old knearst
           /* int theK = (int)Math.Ceiling(kPercent * parentSize);
            double[][] child0OriginalData = child0.pointsIdArray.Select(id => _db.training_dt[id]).ToArray();
            double[][] child1OriginalData = child1.pointsIdArray.Select(id => _db.training_dt[id]).ToArray();
            

            double[][] distances = new double[parentSize][];
            Helpers.applyFor(0, parentSize, i =>
            {
                int childClass = (i < child0.pointsIdArray.Count) ? -1 : 1;
                double[] originalPoint = (childClass == -1) ? child0OriginalData[i] :
                                          child1OriginalData[i-child0.pointsIdArray.Count];
                distances[i] = new double[2]; // Will store both distance and index in here
                distances[i][0] = getEuclidianDistance(originalPoint, testPoint); //distance
                distances[i][1] = childClass; //class
            });
            return sortOnlyKelements(distances, theK) ? child0 : child1;*/
            //double votingValue = distances.OrderBy(t => t[0]).Take(theK).AsParallel().Sum(t=>t[1]);
           // return (votingValue <= 0) ? child0 : child1;

            //end old knears
            int theNewK = (int)Math.Ceiling(kPercent * parentSize);
            double[] child0Dist = child0.pointsIdArray.Select(id => getEuclidianDistance(_db.training_dt[id], testPoint)).ToArray();
            double[] child1Dist = child1.pointsIdArray.Select(id => getEuclidianDistance(_db.training_dt[id], testPoint)).ToArray();
            double vote = 0;
            for (int i = 0; i < theNewK; i++)
            {
                int indOfMin0 = Helpers.indexOfMin(child0Dist);
                int indOfMin1 = Helpers.indexOfMin(child1Dist);
                double minValue0 =  child0Dist[indOfMin0];
                double minValue1 = child1Dist[indOfMin1];
                if (minValue0 <= minValue1)
                {
                    child0Dist[indOfMin0] = Double.MaxValue;
                    //child0Dist=child0Dist.RemoveAt(indOfMin0);
                    vote--;
                }
                else
                {
                    child1Dist[indOfMin1] = Double.MaxValue;
                    //child1Dist = child1Dist.RemoveAt(indOfMin1);
                    vote++;
                }
            }
            return (vote <= 0) ? child0 : child1;
        }

        private static bool sortOnlyKelements(double[][] unsorted, int k)
        {
            double vote = 0;
            for (int i = 0; i < k; i++)
            {
                int indOfMin = Helpers.indexOfMin(unsorted);
                vote += unsorted[indOfMin][1];
                unsorted = unsorted.RemoveAt(indOfMin);
            }
            return (vote <= 0);
        }
        private static double getEuclidianDistance(IList<double> sample1, IList<double> sample2)
        {

            return sample1.Select((t, i) => t - sample2[i]).Sum(temp => temp*temp);
        }

        private double[] askTreeMeanVal(double[] point, List<GeoWave> treeOrderedById, double normThreshold)
        {
            return askTreeMeanValTransformed(point, treeOrderedById, normThreshold);
           /* //different calculation for local pca case, without bounding boxes
            if (_rc.split_type == 5 || _rc.split_type==0)
            {
                return askTreeMeanValTransformed(point, treeOrderedById, normThreshold);
            }
            // calculations for others splits types
            if (!DB.IsPntInsideBox(treeOrderedById[0].boubdingBox, point, _rc.dim))
            {
                DB.ProjectPntInsideBox(treeOrderedById[0].boubdingBox, ref point);
            }

            double[] zeroMean = new double[treeOrderedById[0].MeanValue.Count()];
            double[] MeanValue = new double[treeOrderedById[0].MeanValue.Count()];

            //SET THE ROOT MEAN VAL
            treeOrderedById[0].MeanValue.CopyTo(MeanValue, 0);
            int parent_index = 0;
            bool endOfLoop = false;
            //2m0rr0w2 start rewrite while
            while (!endOfLoop)
            {
                int child0Ind = treeOrderedById[parent_index].child0;
                int child1Ind = treeOrderedById[parent_index].child1;
                GeoWave child0 = (child0Ind == -1) ? null : treeOrderedById[child0Ind];
                GeoWave child1 = (child1Ind == -1) ? null : treeOrderedById[child1Ind];
                GeoWave selectedChild = null;
                if (child0 != null && DB.IsPntInsideBox(child0.boubdingBox, point, _rc.dim)) selectedChild = child0;
                if (child1 != null && DB.IsPntInsideBox(child1.boubdingBox, point, _rc.dim)) selectedChild = child1;

                if (selectedChild != null)
                {
                    if (!selectedChild.MeanValue.SequenceEqual(zeroMean) && normThreshold <= selectedChild.norm)
                        MeanValue[0] += (selectedChild.MeanValue[0] - treeOrderedById[parent_index].MeanValue[0]);
                    parent_index = selectedChild.ID;
                }
                else endOfLoop = true;
            }
            //2m0rr0w2 end rewrite while
            return MeanValue;*/
          
/*
            while (!endOfLoop)
            {
                if (treeOrderedById[parent_index].child0 != -1 &&
                    DB.IsPntInsideBox(treeOrderedById[treeOrderedById[parent_index].child0].boubdingBox, point, _rc.dim))
                {
                    if (!treeOrderedById[treeOrderedById[parent_index].child0].MeanValue.SequenceEqual(zeroMean) &&
                        normThreshold <= treeOrderedById[treeOrderedById[parent_index].child0].norm) //take the mean value if its not 0 and the wavelete should be taken ( norm size) - or if its the root wavelete
                    {
                        //Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue.CopyTo(MeanValue, 0);
                        //MeanValue = (Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue.Subtract(Tree_orderedById[parent_index].MeanValue)).Add(MeanValue);
                        MeanValue[0] += (treeOrderedById[treeOrderedById[parent_index].child0].MeanValue[0] - treeOrderedById[parent_index].MeanValue[0]);
                    }

                    parent_index = treeOrderedById[parent_index].child0;
                }
                else if (treeOrderedById[parent_index].child1 != -1 &&
                    DB.IsPntInsideBox(treeOrderedById[treeOrderedById[parent_index].child1].boubdingBox, point, _rc.dim))
                {
                    if (!treeOrderedById[treeOrderedById[parent_index].child1].MeanValue.SequenceEqual(zeroMean) &&
                        normThreshold <= treeOrderedById[treeOrderedById[parent_index].child1].norm) //take the mean value if its not 0 and the wavelete should be taken ( norm size) - or if its the root wavelete
                    {
                        //Tree_orderedById[Tree_orderedById[parent_index].child1].MeanValue.CopyTo(MeanValue, 0);
                        //MeanValue = (Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue.Subtract(Tree_orderedById[parent_index].MeanValue)).Add(MeanValue);
                        MeanValue[0] += (treeOrderedById[treeOrderedById[parent_index].child1].MeanValue[0] - treeOrderedById[parent_index].MeanValue[0]);
                    }

                    parent_index = treeOrderedById[parent_index].child1;
                }
                else
                    endOfLoop = true;
            }*/

     
        }
        
        private double[] askTreeMeanValBoundLevel(double[] point, List<GeoWave> treeOrderedById, double normThreshold, int boundLevel)
        {
            //if (point.Count() != Tree_orderedById[0].boubdingBox[0].Count())
            //{
            //    MessageBox.Show("the dim of the point is not compatible with the dim of the tree");
            //    return null;
            //}

            int counter = 0;
            if (!DB.IsPntInsideBox(treeOrderedById[0].boubdingBox, point, _rc.dim))
            {
                DB.ProjectPntInsideBox(treeOrderedById[0].boubdingBox, ref point);
                counter++;
            }

            double[] zeroMean = new double[treeOrderedById[0].MeanValue.Count()];
            double[] MeanValue = new double[treeOrderedById[0].MeanValue.Count()];

            //SET THE ROOT MEAN VAL
            treeOrderedById[0].MeanValue.CopyTo(MeanValue, 0);

            ////get to leaf 

            int parent_index = 0;

            while (treeOrderedById[parent_index].child0 != -1 && treeOrderedById[parent_index].level <= boundLevel)
            {
                if (DB.IsPntInsideBox(treeOrderedById[treeOrderedById[parent_index].child0].boubdingBox, point, _rc.dim))
                {
                    if (!treeOrderedById[treeOrderedById[parent_index].child0].MeanValue.SequenceEqual(zeroMean) &&
                        normThreshold <= treeOrderedById[treeOrderedById[parent_index].child0].norm) //take the mean value if its not 0 and the wavelete should be taken ( norm size) - or if its the root wavelete
                    {
                        //Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue.CopyTo(MeanValue, 0);
                        MeanValue[0] += (treeOrderedById[treeOrderedById[parent_index].child0].MeanValue[0] - treeOrderedById[parent_index].MeanValue[0]);
                    }

                    parent_index = treeOrderedById[parent_index].child0;
                }
                else
                {
                    if (!treeOrderedById[treeOrderedById[parent_index].child1].MeanValue.SequenceEqual(zeroMean) &&
                        normThreshold <= treeOrderedById[treeOrderedById[parent_index].child1].norm) //take the mean value if its not 0 and the wavelete should be taken ( norm size) - or if its the root wavelete
                    {
                        //Tree_orderedById[Tree_orderedById[parent_index].child1].MeanValue.CopyTo(MeanValue, 0);
                        MeanValue[0] += (treeOrderedById[treeOrderedById[parent_index].child1].MeanValue[0] - treeOrderedById[parent_index].MeanValue[0]);
                    }

                    parent_index = treeOrderedById[parent_index].child1;
                }
            }
            return MeanValue;
        }

        private double[] askTreeMeanValAtLevel(double[] point, List<GeoWave> treeOrderedById, int topLevel)
        {
            //if (point.Count() != Tree_orderedById[0].boubdingBox[0].Count())
            //{
            //    MessageBox.Show("the dim of the point is not compatible with the dim of the tree");
            //    return null;
            //}
            if (!DB.IsPntInsideBox(treeOrderedById[0].boubdingBox, point, _rc.dim))
            {
                DB.ProjectPntInsideBox(treeOrderedById[0].boubdingBox, ref point);
            }

            double[] zeroMean = new double[treeOrderedById[0].MeanValue.Count()];
            double[] MeanValue = new double[treeOrderedById[0].MeanValue.Count()];

            //SET THE ROOT MEAN VAL
            treeOrderedById[0].MeanValue.CopyTo(MeanValue, 0);

            ////get to leaf 

            int parent_index = 0;

            while (treeOrderedById[parent_index].child0 != -1)
            {
                if (DB.IsPntInsideBox(treeOrderedById[treeOrderedById[parent_index].child0].boubdingBox, point, _rc.dim))
                {
                    if (!treeOrderedById[treeOrderedById[parent_index].child0].MeanValue.SequenceEqual(zeroMean) &&
                        topLevel >= treeOrderedById[treeOrderedById[parent_index].child0].level) //take the mean value if its not 0 and the wavelete should be taken ( norm size) - or if its the root wavelete
                    {
                        treeOrderedById[treeOrderedById[parent_index].child0].MeanValue.CopyTo(MeanValue, 0);
                    }

                    parent_index = treeOrderedById[parent_index].child0;
                }
                else
                {
                    if (!treeOrderedById[treeOrderedById[parent_index].child1].MeanValue.SequenceEqual(zeroMean) &&
                        topLevel >= treeOrderedById[treeOrderedById[parent_index].child1].level) //take the mean value if its not 0 and the wavelete should be taken ( norm size) - or if its the root wavelete
                    {
                        treeOrderedById[treeOrderedById[parent_index].child1].MeanValue.CopyTo(MeanValue, 0);
                    }

                    parent_index = treeOrderedById[parent_index].child1;
                }
            }
            return MeanValue;
        }

        public double[][] getResidualLabelsInBoosting(List<GeoWave> tree, double[][] trainingDt, double[][] boostedLabels, double threshNorm)
        {
            List<GeoWave> Tree_orderedById = tree.OrderBy(o => o.ID).ToList();
            Helpers.applyFor(0, boostedLabels.Count(), i =>
            {
                //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                double[] point = new double[_rc.dim];//training_dt[0].Count()
                //Data_table.CopyTo(point, i);
                for (int j = 0; j < _rc.dim; j++)//training_dt[0].Count()
                    point[j] = double.Parse(trainingDt[i][j].ToString());

                double[] tmpLabel = askTreeMeanVal(point, Tree_orderedById, threshNorm);
                for (int j = 0; j < tmpLabel.Count(); j++)
                    boostedLabels[i][j] = boostedLabels[i][j] - tmpLabel[j];
            });
            return boostedLabels;
        }


        private double[][] getResidualLabelsInBoostingProoning(List<GeoWave> tree, double[][] trainingDt, double[][] boostedLabelsPooning, int bestLevel)
        {
            List<GeoWave> Tree_orderedById = tree.OrderBy(o => o.ID).ToList();
            Helpers.applyFor(0, boostedLabelsPooning.Count(), i =>
            {
                //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                double[] point = new double[_rc.dim];//training_dt[0].Count()
                //Data_table.CopyTo(point, i);
                for (int j = 0; j < _rc.dim; j++)//training_dt[0].Count()
                    point[j] = double.Parse(trainingDt[i][j].ToString());

                double[] tmpLabel = askTreeMeanValAtLevel(point, Tree_orderedById, bestLevel);
                for (int j = 0; j < tmpLabel.Count(); j++)
                    boostedLabelsPooning[i][j] = boostedLabelsPooning[i][j] - tmpLabel[j];
            });
            return boostedLabelsPooning;
        }
        
        public static void printErrorsOfTree(double[] errArr, string filename)
        {
            StreamWriter writer;
        /*    if (Form1.UseS3)
            {
                string dir_name = Path.GetDirectoryName(filename);
                string file_name = Path.GetFileName(filename);

                S3DirectoryInfo s3dir = new S3DirectoryInfo(Form1.S3client, Form1.bucketName, dir_name);
                S3FileInfo artFile = s3dir.GetFile(file_name);
                writer = new StreamWriter(artFile.OpenWrite());
            }
            else*/
                writer = new StreamWriter(filename, false);

            for (int i = 0; i < errArr.Count(); i++)
                writer.WriteLine(errArr[i]);
            writer.Close();
        }

        public static void printErrorsOfTree(double[] errArr,double[] nwavesArr, string filename)
        {
            StreamWriter writer;
         /*   if (Form1.UseS3)
            {
                string dir_name = Path.GetDirectoryName(filename);
                string file_name = Path.GetFileName(filename);

                S3DirectoryInfo s3dir = new S3DirectoryInfo(Form1.S3client, Form1.bucketName, dir_name);
                S3FileInfo artFile = s3dir.GetFile(file_name);
                writer = new StreamWriter(artFile.OpenWrite());
            }
            else*/
                writer = new StreamWriter(filename, false);

            for (int i = 0; i < errArr.Count(); i++)
                writer.WriteLine(nwavesArr[i] + " " + errArr[i]);
            writer.Close();
        }

        public static void printErrorsOfTree(double err, int nwaves, string filename)
        {
            StreamWriter writer;
        /*    if (Form1.UseS3)
            {
                string dir_name = Path.GetDirectoryName(filename);
                string file_name = Path.GetFileName(filename);

                S3DirectoryInfo s3dir = new S3DirectoryInfo(Form1.S3client, Form1.bucketName, dir_name);
                S3FileInfo artFile = s3dir.GetFile(file_name);
                writer = new StreamWriter(artFile.OpenWrite());
            }
            else*/
                writer = new StreamWriter(filename, false);

            writer.WriteLine(nwaves + " " + err);
            writer.Close();
        }
    }
}

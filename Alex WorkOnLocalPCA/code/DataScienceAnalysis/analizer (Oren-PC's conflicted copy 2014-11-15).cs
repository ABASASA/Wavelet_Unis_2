using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using System.Diagnostics;
using Amazon.S3.IO;
using Accord.MachineLearning;
using Accord.Math;
using Accord.Math.Decompositions;
using Accord.Statistics;
using Accord.Statistics.Analysis;

namespace DataScienceAnalysis
{
    class analizer
    {
        private string analysisFolderName;
        private List<List<double>> MainGrid;
        private DB db;
        private recordConfig rc;

        public analizer(string analysisFolderName, List<List<double>> MainGrid, DB db, recordConfig rc)
        {
            // TODO: Complete member initialization
            this.analysisFolderName = analysisFolderName;
            this.MainGrid = MainGrid;
            this.db = db;
            this.rc = rc;

            ////CREATE DATA STRUCTURE
            //for (int i = 0; i < Form1.dataStruct.Count(); i++)
            //{
            //    if (!System.IO.Directory.Exists(analysisFolderName + Form1.dataStruct[i]))
            //        System.IO.Directory.CreateDirectory(analysisFolderName + Form1.dataStruct[i]);
            //}  
        }

        public void analize(List<int> trainingArr, List<int> testingArr, int[][] boundingBox)
        {
            #region one tree

            //CREATE DECISION TREES
            var watch = Stopwatch.StartNew();

            //bool[] Dim2TakeOneTree = getDim2Take(rc, 1);
            //decicionTree decTree = new decicionTree(rc, db, Dim2TakeOneTree);

            decicionTree decTree = new decicionTree(rc, db);
            List<GeoWave> decision_GeoWaveArr = decTree.getdecicionTree(trainingArr, boundingBox);
            watch.Stop();
            double[] toc_time = new double[1];
            toc_time[0] = watch.ElapsedMilliseconds;


            printErrorsOfTree(toc_time, analysisFolderName + "\\time_to_generate_bsp_tree.txt");
            //dbg
            //Form1.printConstWavelets2File(decision_GeoWaveArr, analysisFolderName + "\\bsp_tree.txt"); - save DB space when printing
            Form1.printLevelWaveletNorm(decision_GeoWaveArr, analysisFolderName + "\\levelNorm.txt");
            Form1.printWaveletsProperties(decision_GeoWaveArr, analysisFolderName + "\\waveletsProperties.txt");
            //test it
            List<GeoWave> final_GeoWaveArr = decision_GeoWaveArr.OrderByDescending(o => o.norm).ToList();//see if not sorted by norm already...

            //int arrSize = Convert.ToInt32(rc.test_error_size * final_GeoWaveArr.Count / rc.hopping_size);
            int testBegin = rc.waveletsTestRange[0];
            int arrSize = rc.waveletsTestRange[1] == 0 ? Convert.ToInt32(rc.test_error_size * final_GeoWaveArr.Count / rc.hopping_size) : Convert.ToInt32((1 + rc.waveletsTestRange[1] - rc.waveletsTestRange[0]) / rc.hopping_size);

            double[] errorTree = new double[arrSize];
            double[] decayOnTraining = new double[arrSize];
            double[] errorTreeL1 = new double[arrSize];
            //double[] errorTreeLmax = new double[arrSize];
            double[] errorTreeBER = new double[arrSize];
            double[] missLabels = new double[arrSize];
            double[] Nwavelets = new double[arrSize];


            if (Form1.rumPrallel)
            {
                Parallel.For(testBegin, arrSize, i =>
                {
                    errorTree[i] = testDecisionTree(db.PCAvalidation_dt, db.validation_label, decision_GeoWaveArr, final_GeoWaveArr[i * rc.hopping_size].norm, 2);
                    //decayOnTraining[i] = testDecisionTree(db.PCAtraining_dt, db.training_label, decision_GeoWaveArr, final_GeoWaveArr[i * rc.hopping_size].norm, 2);
                    //errorTreeL1[i] = testDecisionTree(db.PCAtesting_dt, db.testing_label, decision_GeoWaveArr, final_GeoWaveArr[i * rc.hopping_size].norm, 1);
                    //errorTreeLmax[i] = testDecisionTree(db.PCAtesting_dt, db.testing_label, decision_GeoWaveArr, final_GeoWaveArr[i * rc.hopping_size].norm, -1);
                    //errorTreeBER[i] = testDecisionTree(db.PCAtesting_dt, db.testing_label, decision_GeoWaveArr, final_GeoWaveArr[i * rc.hopping_size].norm, -2);
                    //missLabels[i] = testDecisionTree(db.PCAtesting_dt, db.testing_label, decision_GeoWaveArr, final_GeoWaveArr[i * rc.hopping_size].norm, 0);
                    Nwavelets[i] = i * rc.hopping_size;
                });
            }
            else
            {
                for (int i = testBegin; i < arrSize; i++)
                {
                    //double dbgNorm = 0;
                    errorTree[i] = testDecisionTree(db.PCAvalidation_dt, db.validation_label, decision_GeoWaveArr, final_GeoWaveArr[i * rc.hopping_size].norm, 2);
                    //decayOnTraining[i] = testDecisionTree(db.PCAtraining_dt, db.training_label, decision_GeoWaveArr, final_GeoWaveArr[i * rc.hopping_size].norm, 2);
                    //errorTreeL1[i] = testDecisionTree(db.PCAtesting_dt, db.testing_label, decision_GeoWaveArr, final_GeoWaveArr[i * rc.hopping_size].norm, 1);
                    //errorTreeLmax[i] = testDecisionTree(db.PCAtesting_dt, db.testing_label, decision_GeoWaveArr, final_GeoWaveArr[i * rc.hopping_size].norm, -1);
                    //errorTreeBER[i] = testDecisionTree(db.PCAtesting_dt, db.testing_label, decision_GeoWaveArr, final_GeoWaveArr[i * rc.hopping_size].norm, -2);
                    //missLabels[i] = testDecisionTree(db.PCAtesting_dt, db.testing_label, decision_GeoWaveArr, final_GeoWaveArr[i * rc.hopping_size].norm, 0);
                    Nwavelets[i] = i * rc.hopping_size;
                }
            }

            int minErr_index = Enumerable.Range(0, errorTree.Length).Aggregate((a, b) => (errorTree[a] < errorTree[b]) ? a : b); //minerror
            double lowest_Tree_error = testDecisionTree(db.PCAtesting_dt, db.testing_label, decision_GeoWaveArr, final_GeoWaveArr[minErr_index * rc.hopping_size].norm, 2);
            printErrorsOfTree(lowest_Tree_error, minErr_index * rc.hopping_size, analysisFolderName + "\\bsp_tree_errors_by_wavelets_TestDB.txt");

            //PRINT ERRORS TO FILE...
            printErrorsOfTree(errorTree, Nwavelets, analysisFolderName + "\\bsp_tree_errors_by_wavelets_ValidationDB.txt");
            printErrorsOfTree(decayOnTraining, Nwavelets, analysisFolderName + "\\decay_on_training_ValidationDB.txt");
            //printErrorsOfTree(errorTreeBER, Nwavelets, analysisFolderName + "\\bsp_tree_BER_by_wavelets.txt");
            //printErrorsOfTree(errorTreeL1, Nwavelets, analysisFolderName + "\\bsp_tree_errors_by_waveletsL1.txt");
            //printErrorsOfTree(errorTreeLmax, Nwavelets, analysisFolderName + "\\bsp_tree_errors_by_waveletsLMAX.txt");
            //printErrorsOfTree(missLabels, Nwavelets, analysisFolderName + "\\bsp_tree_missLables_by_wavelets.txt");            
            
            #region prooning one tree            
            
            if(Form1.runProoning)
            {
                //TEST TREE WITH PROONING
                int topLevelBegin = rc.pruningTestRange[0];
                int topLevel = rc.waveletsTestRange[1] == 0 ? getTopLevel(decision_GeoWaveArr) : rc.waveletsTestRange[1];

                //int topLevel = getTopLevel(decision_GeoWaveArr);
                double[] errorTreeProoning = new double[topLevel];
                //double[] errorTreeProoningL1 = new double[topLevel];
                double[] NLevels = new double[topLevel];
                //double[] errorTreeProoningBER = new double[topLevel];

                if (Form1.rumPrallel)
                {
                    Parallel.For(topLevelBegin, topLevel, i =>
                    {
                        errorTreeProoning[i] = testDecisionTreeWithProoning(db.PCAvalidation_dt, db.validation_label, decision_GeoWaveArr, i + 1, 2);
                        //errorTreeProoningBER[i] = testDecisionTreeWithProoning(db.PCAtesting_dt, db.testing_label, decision_GeoWaveArr, i + 1, -2);
                        //errorTreeProoningL1[i] = testDecisionTreeWithProoning(db.PCAtesting_dt, db.testing_label, decision_GeoWaveArr, i + 1, 1);
                        NLevels[i] = i;// * rc.hopping_size;
                    });
                }
                else
                {
                    for (int i = topLevelBegin; i < topLevel; i++)
                    {
                        errorTreeProoning[i] = testDecisionTreeWithProoning(db.PCAvalidation_dt, db.validation_label, decision_GeoWaveArr, i + 1, 2);
                        //errorTreeProoningBER[i] = testDecisionTreeWithProoning(db.PCAtesting_dt, db.testing_label, decision_GeoWaveArr, i + 1, -2);
                        //errorTreeProoningL1[i] = testDecisionTreeWithProoning(db.PCAtesting_dt, db.testing_label, decision_GeoWaveArr, i + 1, 1);
                        NLevels[i] = i;// *rc.hopping_size;
                    }
                }


                int minErrPruning_index = Enumerable.Range(0, errorTreeProoning.Length).Aggregate((a, b) => (errorTreeProoning[a] < errorTreeProoning[b]) ? a : b); //minerror
                double lowest_TreePruning_error = testDecisionTree(db.PCAtesting_dt, db.testing_label, decision_GeoWaveArr, minErrPruning_index +1 , 2);
                printErrorsOfTree(lowest_TreePruning_error, minErr_index * rc.hopping_size, analysisFolderName + "\\bsp_tree_errors_by_waveletsPruning_TestDB.txt");
                
                //PRINT ERRORS TO FILE...
                printErrorsOfTree(errorTreeProoning, NLevels, analysisFolderName + "\\bsp_tree_errors_by_prooning_Validation.txt");
                //printErrorsOfTree(errorTreeProoningBER, NLevels, analysisFolderName + "\\bsp_tree_errors_by_prooningBER.txt");
                //printErrorsOfTree(errorTreeProoningL1, NLevels, analysisFolderName + "\\bsp_tree_errors_by_prooningL1.txt");
             #endregion

            }

            #endregion

           #region RF tree 

            if (Form1.runRf)
            {
                //create RF
                List<GeoWave>[] RFdecTreeArr = new List<GeoWave>[rc.rfNum];

                if (Form1.rumPrallel)
                {
                    Parallel.For(0, rc.rfNum, i =>
                    {
                        List<int> trainingArrRF = Bagging(trainingArr, rc.rfBaggingPercent, i);
                        //bool[] Dim2Take = getDim2Take(rc, i);
                        //decicionTree decTreeRF = new decicionTree(rc, db, Dim2Take);
                        decicionTree decTreeRF = new decicionTree(rc, db);
                        //RFdecTreeArr[i] = decTree.getdecicionTree(trainingArrRF, boundingBox);
                        RFdecTreeArr[i] = decTree.getdecicionTree(trainingArrRF, boundingBox, i);
                        //Form1.printConstWavelets2File(RFdecTreeArr[i], analysisFolderName + "\\RFdecTreeArr_" + i.ToString() + "_tree.txt");//dbg
                        //Form1.printtable(db.PCAtraining_dt, analysisFolderName + "\\PCA_DATA_" + i.ToString() + "tree.txt", trainingArrRF);
                        //Form1.printtable(db.training_label, analysisFolderName + "\\PCA_label_" + i.ToString() + "tree.txt", trainingArrRF);
                    });
                }
                else
                {
                    for (int i = 0; i < rc.rfNum; i++)
                    {
                        List<int> trainingArrRF = Bagging(trainingArr, rc.rfBaggingPercent, i);
                        //bool[] Dim2Take = getDim2Take(rc, i);
                        //decicionTree decTreeRF = new decicionTree(rc, db, Dim2Take);
                        decicionTree decTreeRF = new decicionTree(rc, db);
                        //RFdecTreeArr[i] = decTree.getdecicionTree(trainingArrRF, boundingBox);
                        RFdecTreeArr[i] = decTree.getdecicionTree(trainingArrRF, boundingBox, i);
                        //Form1.printConstWavelets2File(RFdecTreeArr[i], analysisFolderName + "\\RFdecTreeArr_" + i.ToString() + "_tree.txt");//dbg
                        //Form1.printtable(db.PCAtraining_dt, analysisFolderName + "\\PCA_DATA_" + i.ToString() + "tree.txt", trainingArrRF);
                        //Form1.printtable(db.training_label, analysisFolderName + "\\PCA_label_" + i.ToString() + "tree.txt", trainingArrRF);
                    }
                }

                //TEST IT
                //int minNwavelets = Int32.MaxValue;
                //for (int i = 0; i < RFdecTreeArr.Count(); i++)
                //{
                //    if (RFdecTreeArr[i].Count() < minNwavelets)
                //    {
                //        minNwavelets = RFdecTreeArr[i].Count();
                //    }
                //}

                List<double> NormArr = new List<double>();
                //for (int j = 0; j < RFdecTreeArr.Count(); j++)
                for (int j = 0; j < 1; j++)//go over tree j==0 (first)
                    for (int i = 0; i < RFdecTreeArr[j].Count; i++)
                    {
                        //if (RFdecTreeArr[j][i].level <= rc.BoundLevel)//restrict the level we take wavelets from
                            NormArr.Add(RFdecTreeArr[j][i].norm);
                    }
                        

                //int arrSizeRF = Convert.ToInt32(rc.test_error_size * NormArr.Count / rc.hopping_size);
                int RFtestBegin = rc.RFwaveletsTestRange[0];
                int arrSizeRF = rc.RFwaveletsTestRange[1] == 0 ? Convert.ToInt32(rc.test_error_size * NormArr.Count() / rc.hopping_size) : Convert.ToInt32((1 + rc.RFwaveletsTestRange[1] - rc.RFwaveletsTestRange[0]) / rc.hopping_size);

                //double[] errorRF = new double[arrSizeRF];
                //double[] errorManyRF = new double[arrSizeRF];
                //double[][] decayManyRF = new double[arrSizeRF][];
                //for (int i = 0; i < arrSizeRF; i++)
                //{
                //    errorManyRF[i] = new double[RFdecTreeArr.Count()];
                //    decayManyRF[i] = new double[RFdecTreeArr.Count()];
                //}

                double[][] errorManyRF = new double[arrSizeRF][];
                for (int i = 0; i < arrSizeRF; i++)
                {
                    errorManyRF[i] = new double[RFdecTreeArr.Count()];
                    errorManyRF[i] = new double[RFdecTreeArr.Count()];
                }
                    
                //double[] missLabelsRF = new double[arrSizeRF];
                double[] NwaveletsRF = new double[arrSizeRF];

                NormArr = NormArr.OrderByDescending(o => o).ToList();

                if (Form1.rumPrallel)
                {
                    Parallel.For(0, arrSizeRF, i =>
                    {
                        //errorRF[i] = testDecisionTreeRF(db.PCAvalidation_dt, db.validation_label, RFdecTreeArr, NormArr[RFtestBegin + i * rc.hopping_size], 2);
                        //errorRF[i] = testDecisionTreeManyRFNormNbound(db.PCAtesting_dt, db.testing_label, RFdecTreeArr, NormArr[i * rc.hopping_size], rc.BoundLevel, 2);
                        errorManyRF[i] = testDecisionTreeManyRF(db.PCAvalidation_dt, db.validation_label, RFdecTreeArr, NormArr[RFtestBegin + i * rc.hopping_size], 2);
                        //decayManyRF[i] = testDecisionTreeManyRF(db.PCAtraining_dt, db.training_label, RFdecTreeArr, NormArr[RFtestBegin + i * rc.hopping_size], 2);
                        //missLabelsRF[i] = testDecisionTreeRF(db.PCAtesting_dt, db.testing_label, RFdecTreeArr, NormArr[i * rc.hopping_size], 0);
                        NwaveletsRF[i] = RFtestBegin +  i * rc.hopping_size;//  / rc.rfNum - if we want to devide by the number of trees to get the degree 
                    });
                }
                else
                {
                    for (int i = 0; i < arrSizeRF; i++)
                    {
                        //errorRF[i] = testDecisionTreeRF(db.PCAvalidation_dt, db.validation_label, RFdecTreeArr, NormArr[RFtestBegin + i * rc.hopping_size], 2);
                        //errorRF[i] = testDecisionTreeManyRFNormNbound(db.PCAtesting_dt, db.testing_label, RFdecTreeArr, NormArr[i * rc.hopping_size], rc.BoundLevel, 2);
                        errorManyRF[i] = testDecisionTreeManyRF(db.PCAvalidation_dt, db.validation_label, RFdecTreeArr, NormArr[RFtestBegin + i * rc.hopping_size], 2);
                        //decayManyRF[i] = testDecisionTreeManyRF(db.PCAtraining_dt, db.training_label, RFdecTreeArr, NormArr[RFtestBegin + i * rc.hopping_size], 2);
                        //missLabelsRF[i] = testDecisionTreeRF(db.PCAtesting_dt, db.testing_label, RFdecTreeArr, NormArr[i * rc.hopping_size], 0);
                        NwaveletsRF[i] = RFtestBegin +  i * rc.hopping_size;//  / rc.rfNum - if we want to devide by the number of trees to get the degree 
                    }
                }

                //int minErrRF_index = Enumerable.Range(0, errorRF.Length).Aggregate((a, b) => (errorRF[a] < errorRF[b]) ? a : b); //minerror
                int minErrRF_index = Enumerable.Range(0, errorManyRF[arrSizeRF - 1].Length).Aggregate((a, b) => (errorManyRF[arrSizeRF - 1][a] < errorManyRF[arrSizeRF - 1][b]) ? a : b); //minerror
                double lowest_TreeRF_error = testDecisionTreeRF(db.PCAtesting_dt, db.testing_label, RFdecTreeArr, NormArr[RFtestBegin + minErrRF_index * rc.hopping_size], 2);
                printErrorsOfTree(lowest_TreeRF_error, minErrRF_index * rc.hopping_size, analysisFolderName + "\\bsp_tree_errors_by_wavelets_RF_TestDB.txt");


                //printErrorsOfTree(errorRF, NwaveletsRF, analysisFolderName + "\\RF_errors_by_waveletsbounded_at_level_" + rc.BoundLevel.ToString() + ".txt");
                //printErrorsOfTree(errorRF, NwaveletsRF, analysisFolderName + "\\RF_errors_by_wavelets_validationDB.txt");
                Form1.printtable(errorManyRF, analysisFolderName + "\\Many_RF_errors_by_wavelets_threshold.txt");
                //Form1.printtable(decayManyRF, analysisFolderName + "\\Many_RF_decay_training_by_wavelets.txt");
                printErrorsOfTree(NwaveletsRF, analysisFolderName + "\\Num_of_RF_wavelets_validationDB.txt");

                //printErrorsOfTree(missLabelsRF, NwaveletsRF, analysisFolderName + "\\RF_MissLabeling_by_wavelets.txt");
                if (Form1.runRFProoning)
                {
                    int topLevel = int.MaxValue;
                    for(int k=0; k < RFdecTreeArr.Count(); k++)
                    {
                        int tmp =getTopLevel(RFdecTreeArr[k]);
                        if (tmp < topLevel)
                            topLevel = tmp;
                    }

                    int topLevelBeginRF = rc.RFpruningTestRange[0];
                    topLevel = rc.RFpruningTestRange[1] == 0 ? topLevel : rc.RFpruningTestRange[1];

                    double[] errorRFProoning = new double[topLevel];
                    double[] NwaveletsRFProoning = new double[topLevel];
                    double[][] errorManyRFProoning = new double[topLevel][];
                    for (int i = 0; i < topLevel; i++)
                        errorManyRFProoning[i] = new double[RFdecTreeArr.Count()];

                    if (Form1.rumPrallel)
                    {
                        Parallel.For(0, topLevel, i =>
                        {
                            //errorRFProoning[i] = testDecisionTreeRF(db.PCAvalidation_dt, db.validation_label, RFdecTreeArr, i + 1, 2);
                            errorManyRFProoning[i] = testDecisionTreeManyRF(db.PCAvalidation_dt, db.validation_label, RFdecTreeArr, i + 1, 2);
                            NwaveletsRFProoning[i] = i+1;//  / rc.rfNum - if we want to devide by the number of trees to get the degree 
                        });
                    }
                    else
                    {
                        for (int i = 0; i < topLevel; i++)
                        {
                            //errorRFProoning[i] = testDecisionTreeRF(db.PCAvalidation_dt, db.validation_label, RFdecTreeArr, i + 1, 2);
                            errorManyRFProoning[i] = testDecisionTreeManyRF(db.PCAvalidation_dt, db.validation_label, RFdecTreeArr, i + 1, 2);
                            NwaveletsRFProoning[i] = i + 1;//  / rc.rfNum - if we want to devide by the number of trees to get the degree 
                        }
                    }

                    //int minErrPruningRF_index = Enumerable.Range(0, errorRFProoning.Length).Aggregate((a, b) => (errorRFProoning[a] < errorRFProoning[b]) ? a : b); //minerror
                    //double lowest_TreePruningRF_error = testDecisionTreeRF(db.PCAtesting_dt, db.testing_label, RFdecTreeArr, minErrPruningRF_index + 1, 2);
                    //printErrorsOfTree(lowest_TreePruningRF_error, minErr_index * rc.hopping_size, analysisFolderName + "\\bsp_tree_errors_by_Pruning_RF_TestDB.txt");
                
                    //printErrorsOfTree(errorRFProoning, NwaveletsRFProoning, analysisFolderName + "\\RF_errors_by_Pruning.txt");
                    Form1.printtable(errorManyRFProoning, analysisFolderName + "\\Many_RF_errors_by_Pruning_validationDB.txt");
                    printErrorsOfTree(NwaveletsRFProoning, analysisFolderName + "\\Num_of_ManyRF_levels_validationDB.txt");              
                }
            }

            #endregion

            #region Boosting tree

            if (Form1.runBoosting)
            {
                //BOOST
                List<GeoWave>[] BoostTreeArr = new List<GeoWave>[rc.boostNum];
                double[][] boostedLabels = new double[db.training_label.Count()][];
                for (int i = 0; i < db.training_label.Count(); i++)
                {
                    boostedLabels[i] = new double[db.training_label[0].Count()];
                    for (int j = 0; j < db.training_label[0].Count(); j++)
                        boostedLabels[i][j] = db.training_label[i][j];
                }

                //Array.Copy(db.training_label, 0, boostedLabels, 0, db.training_label.Length); - bad copy - by reference
                double[] best_norms = new double[rc.boostNum];
                int[] best_indeces = new int[rc.boostNum];
                for (int i = 0; i < rc.boostNum; i++)
                {
                    decicionTree decTreeBoost = new decicionTree(rc, db.PCAtraining_dt, boostedLabels, db.PCAtraining_GridIndex_dt);
                    if (i == 0 && decision_GeoWaveArr.Count > 0)
                        BoostTreeArr[i] = decision_GeoWaveArr;//take tree from first creation of "BSP" tree
                    else
                        BoostTreeArr[i] = decTreeBoost.getdecicionTree(trainingArr, boundingBox);

                    //KFUNC
                    //best_indeces[i] = getGWIndexByKfunc(BoostTreeArr[i], rc, db.PCAtraining_dt, boostedLabels, ref best_norms[i]);
                    best_indeces[i] = getGWIndexByKfuncLessAcurate(BoostTreeArr[i], rc, db.PCAtraining_dt, boostedLabels, ref best_norms[i]);
                    best_norms[i] = BoostTreeArr[i][best_indeces[i]].norm;

                    boostedLabels = GetResidualLabelsInBoosting(BoostTreeArr[i], db.PCAtraining_dt, boostedLabels, best_norms[i]);
                    //Form1.printtable(boostedLabels, analysisFolderName + "\\BoostingLabels_" + i.ToString() + "_tree.txt");
                    rc.boostlamda_0 = rc.boostlamda_0 * 0.5;

                    //dbg
                    //Form1.printConstWavelets2File(BoostTreeArr[i], analysisFolderName + "\\BoostingdecTreeArr_" + i.ToString() + "_tree.txt");//dbg
                }

                double[] tmpArr = new double[BoostTreeArr.Count()];
                for (int i = 0; i < BoostTreeArr.Count(); i++)
                {
                    tmpArr[i] = Convert.ToDouble(best_indeces[i]);
                }
                printErrorsOfTree(tmpArr, analysisFolderName + "\\num_wavelets_in_boosting.txt");
                printErrorsOfTree(best_norms, analysisFolderName + "\\threshold_norms_of_wavelets_in_boosting.txt");

                //TEST IT
                List<double> NormArrBoosting = new List<double>();
                for (int i = 0; i < BoostTreeArr.Count(); i++)
                    for (int j = 0; j < BoostTreeArr[i].Count; j++)
                        if (BoostTreeArr[i][j].norm >= best_norms[i])
                            NormArrBoosting.Add(BoostTreeArr[i][j].norm);
                NormArrBoosting = NormArrBoosting.OrderByDescending(o => o).ToList();

                int arrSizeBoost = Convert.ToInt32(rc.test_error_size * NormArrBoosting.Count / rc.hopping_size);
                double[] errorBoosting = new double[arrSizeBoost];
                //double[] missLabelsBoosting = new double[arrSizeBoost];
                double[] missLabelsBoostingBER = new double[arrSizeBoost];
                double[] NwaveletsBoosting = new double[arrSizeBoost];

                if (Form1.rumPrallel)
                {
                    Parallel.For(0, arrSizeBoost, i =>
                    {
                        errorBoosting[i] = testDecisionTreeBoosting(db.PCAtesting_dt, db.testing_label, BoostTreeArr, NormArrBoosting[i * rc.hopping_size], 2, best_norms);
                        //missLabelsBoosting[i] = testDecisionTreeBoosting(db.PCAtesting_dt, db.testing_label, BoostTreeArr, NormArr[i * rc.hopping_size], 0, best_norms);
                        //missLabelsBoostingBER[i] = testDecisionTreeBoosting(db.PCAtesting_dt, db.testing_label, BoostTreeArr, NormArrBoosting[i * rc.hopping_size], -2, best_norms);
                        NwaveletsBoosting[i] = i * rc.hopping_size;
                    });
                }
                else
                {
                    for (int i = 0; i < arrSizeBoost; i++)
                    {
                        errorBoosting[i] = testDecisionTreeBoosting(db.PCAtesting_dt, db.testing_label, BoostTreeArr, NormArrBoosting[i * rc.hopping_size], 2, best_norms);
                        //missLabelsBoostingBER[i] = testDecisionTreeBoosting(db.PCAtesting_dt, db.testing_label, BoostTreeArr, NormArrBoosting[i * rc.hopping_size], -2, best_norms);
                        //missLabelsBoosting[i] = testDecisionTreeBoosting(db.PCAtesting_dt, db.testing_label, BoostTreeArr, NormArr[i * rc.hopping_size], 0, best_norms);
                        NwaveletsBoosting[i] = i * rc.hopping_size;
                    }
                }

                printErrorsOfTree(errorBoosting, NwaveletsBoosting, analysisFolderName + "\\Boosting_errors_by_wavelets.txt");
                //printErrorsOfTree(missLabelsBoostingBER, NwaveletsBoosting, analysisFolderName + "\\Boosting_BER_by_wavelets.txt");
                //printErrorsOfTree(missLabelsBoosting, NwaveletsBoosting, analysisFolderName + "\\Boosting_missLabe_by_wavelets.txt");                        
            }
            #endregion
            
            #region Prooning Boosting tree
            
            if (Form1.runBoostingProoning)
            {
                //BOOST
                List<GeoWave>[] BoostTreeArrPooning = new List<GeoWave>[rc.boostNum];
                double[][] boostedLabelsPooning = new double[db.training_label.Count()][];
                for (int i = 0; i < db.training_label.Count(); i++)
                {
                    boostedLabelsPooning[i] = new double[db.training_label[0].Count()];
                    for (int j = 0; j < db.training_label[0].Count(); j++)
                        boostedLabelsPooning[i][j] = db.training_label[i][j];
                }

                //Array.Copy(db.training_label, 0, boostedLabels, 0, db.training_label.Length); - bad copy - by reference
                int[] best_level = new int[rc.boostNum];
                int[] best_indecesProoning = new int[rc.boostNum];
                for (int i = 0; i < rc.boostNum; i++)
                {
                    decicionTree decTreeBoost = new decicionTree(rc, db.PCAtraining_dt, boostedLabelsPooning, db.PCAtraining_GridIndex_dt);
                    if (i == 0 && decision_GeoWaveArr.Count > 0)
                        BoostTreeArrPooning[i] = decision_GeoWaveArr;//take tree from first creation of "BSP" tree
                    else
                        BoostTreeArrPooning[i] = decTreeBoost.getdecicionTree(trainingArr, boundingBox);

                    //KFUNC
                    //best_indeces[i] = getGWIndexByKfunc(BoostTreeArr[i], rc, db.PCAtraining_dt, boostedLabels, ref best_norms[i]);
                    //best_level[i] = getGWIndexByKfuncLessAcuratePooning(BoostTreeArrPooning[i], rc, db.PCAtraining_dt, boostedLabelsPooning);

                    best_level[i] = Convert.ToInt32(rc.boostProoning_0);

                    boostedLabelsPooning = GetResidualLabelsInBoostingProoning(BoostTreeArrPooning[i], db.PCAtraining_dt, boostedLabelsPooning, best_level[i]);

                    //dbg
                    //Form1.printtable(boostedLabelsPooning, analysisFolderName + "\\BoostingPruningLabels_" + i.ToString() + "_tree.txt");
                    //Form1.printConstWavelets2File(BoostTreeArrPooning[i], analysisFolderName + "\\Boosting_Prooning_decTreeArr_" + i.ToString() + "_tree.txt");//dbg
                }

                double[] tmpArr = new double[BoostTreeArrPooning.Count()];
                for (int i = 0; i < BoostTreeArrPooning.Count(); i++)
                {
                    tmpArr[i] = Convert.ToDouble(best_level[i]);
                }
                printErrorsOfTree(tmpArr, analysisFolderName + "\\tree_levels_in_boosting.txt");

                //TEST IT
                double[] errorBoostingProoning = new double[rc.boostNum];//error size in each boosting step

                testDecisionTreeBoostingProoning(db.PCAtesting_dt, db.testing_label, BoostTreeArrPooning, best_level, 2, errorBoostingProoning);
                printErrorsOfTree(errorBoostingProoning, tmpArr, analysisFolderName + "\\Boosting_Prooning_errors_by_levels.txt");

                //double[] errorBoostingProoningBER = new double[rc.boostNum];//error size in each boosting step
                //testDecisionTreeBoostingProoning(db.PCAtesting_dt, db.testing_label, BoostTreeArrPooning, best_level, -2, errorBoostingProoningBER);
                //printErrorsOfTree(errorBoostingProoningBER, tmpArr, analysisFolderName + "\\Boosting_Prooning_BER_by_levels.txt");    
            }
            #endregion

            #region Boosting tree LearningRate

            if (Form1.runBoostingLearningRate)
            {
                //BOOST
                List<GeoWave>[] BoostTreeArrLearningRate = new List<GeoWave>[rc.boostNumLearningRate];
                double[] BoostArrLearningRateNorms = new double[rc.boostNumLearningRate];
                double[][] boostedLabelsLearningRate = new double[db.training_label.Count()][];
                for (int i = 0; i < db.training_label.Count(); i++)
                {
                    boostedLabelsLearningRate[i] = new double[db.training_label[0].Count()];
                    for (int j = 0; j < db.training_label[0].Count(); j++)
                        boostedLabelsLearningRate[i][j] = db.training_label[i][j];
                }

                //Array.Copy(db.training_label, 0, boostedLabels, 0, db.training_label.Length); - bad copy - by reference
                int[] best_level = new int[rc.boostNumLearningRate];
                int[] best_indecesProoning = new int[rc.boostNumLearningRate];
                for (int i = 0; i < rc.boostNumLearningRate; i++)
                {
                    decicionTree decTreeBoost = new decicionTree(rc, db.PCAtraining_dt, boostedLabelsLearningRate, db.PCAtraining_GridIndex_dt);
                    if (i == 0 && decision_GeoWaveArr.Count > 0)
                        BoostTreeArrLearningRate[i] = decision_GeoWaveArr;//take tree from first creation of "BSP" tree
                    else
                        BoostTreeArrLearningRate[i] = decTreeBoost.getdecicionTree(trainingArr, boundingBox);

                    //KFUNC
                    //best_indeces[i] = getGWIndexByKfunc(BoostTreeArr[i], rc, db.PCAtraining_dt, boostedLabels, ref best_norms[i]);
                    //best_level[i] = getGWIndexByKfuncLessAcuratePooning(BoostTreeArrPooning[i], rc, db.PCAtraining_dt, boostedLabelsPooning);

                    //best_level[i] = Convert.ToInt32(rc.boostProoning_0);

                    List<GeoWave> tmp_GeoWaveArr = BoostTreeArrLearningRate[i].OrderByDescending(o => o.norm).ToList();//see if not sorted by norm already...
                    if (tmp_GeoWaveArr.Count > rc.NwaveletsBoosting)
                    {
                        BoostArrLearningRateNorms[i] = tmp_GeoWaveArr[rc.NwaveletsBoosting].norm;
                        boostedLabelsLearningRate = GetResidualLabelsInBoosting(BoostTreeArrLearningRate[i], db.PCAtraining_dt, boostedLabelsLearningRate, BoostArrLearningRateNorms[i]);
                    }
                    else
                    {
                        BoostArrLearningRateNorms[i] = tmp_GeoWaveArr[tmp_GeoWaveArr.Count -1].norm;
                        boostedLabelsLearningRate = GetResidualLabelsInBoosting(BoostTreeArrLearningRate[i], db.PCAtraining_dt, boostedLabelsLearningRate, BoostArrLearningRateNorms[i]);                    
                    }


                    //dbg
                    //Form1.printtable(boostedLabelsLearningRate, analysisFolderName + "\\BoostingLearningRateLabels_" + i.ToString() + "_tree.txt");
                    //Form1.printConstWavelets2File(BoostTreeArrLearningRate[i], analysisFolderName + "\\Boosting_LearningRate_decTreeArr_" + i.ToString() + "_tree.txt");//dbg
                }

                double[] BoostTreeArrLearningRateErrors = new double[rc.boostNumLearningRate];
                for (int i = 0; i < rc.boostNumLearningRate; i++ )
                    BoostTreeArrLearningRateErrors[i] = testDecisionTreeBoostingLearningRate(db.PCAtesting_dt, db.testing_label, BoostTreeArrLearningRate, 2, BoostArrLearningRateNorms, i+1);

                printErrorsOfTree(BoostTreeArrLearningRateErrors, analysisFolderName + "\\BoostTreeArrLearningRateError.txt");

                ////TEST IT
                //double[] errorBoostingProoning = new double[rc.boostNum];//error size in each boosting step

                //testDecisionTreeBoostingProoning(db.PCAtesting_dt, db.testing_label, BoostTreeArrLearningRate, best_level, 2, errorBoostingProoning);
                //printErrorsOfTree(errorBoostingProoning, tmpArr, analysisFolderName + "\\Boosting_LearningRate_errors_by_levels.txt");

                //double[] errorBoostingProoningBER = new double[rc.boostNum];//error size in each boosting step
                //testDecisionTreeBoostingProoning(db.PCAtesting_dt, db.testing_label, BoostTreeArrPooning, best_level, -2, errorBoostingProoningBER);
                //printErrorsOfTree(errorBoostingProoningBER, tmpArr, analysisFolderName + "\\Boosting_Prooning_BER_by_levels.txt");    
            }
            #endregion
        }

        private bool[] getDim2Take(recordConfig rc, int Seed)
        {
            bool[] Dim2Take = new bool[rc.dim];

            var ran = new Random(Seed);
            List<int> dimArr = Enumerable.Range(0, rc.dim).OrderBy(x => ran.Next()).ToList().GetRange(0, rc.dim);
            for (int i = 0; i < rc.NDimsinRF; i++)
            {
                Dim2Take[dimArr[i]] = true;
            }

            return Dim2Take;
        }

        private int getTopLevel(List<GeoWave> decision_GeoWaveArr)
        {
            int topLevel = 0;
            for (int i = 0; i < decision_GeoWaveArr.Count; i++)
                if (decision_GeoWaveArr[i].level > topLevel)
                    topLevel = decision_GeoWaveArr[i].level;
            return topLevel;
        }

        private double testDecisionTreeBoosting(double[][] testing_dt, double[][] testing_label, List<GeoWave>[] BoostTreeArr, double NormThreshold, int NormLPType, double[] maxNorms)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, BoostTreeArr.Count(), i =>
                {
                    BoostTreeArr[i] = BoostTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < BoostTreeArr.Count(); i++)
                {
                    BoostTreeArr[i] = BoostTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }                        

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[testing_label.Count()][];
            for (int i = 0; i < testing_label.Count(); i++)
                estimatedLabels[i] = new double[testing_label[0].Count()];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, testing_label.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//testing_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//testing_dt[0].Count()
                        point[j] = double.Parse(testing_dt[i][j].ToString());
                    double[][] tmpLabel = new double[BoostTreeArr.Count()][];
                    Parallel.For(0, BoostTreeArr.Count(), j =>
                    {
                        if (NormThreshold < maxNorms[j])
                            tmpLabel[j] = askTreeMeanVal(point, BoostTreeArr[j], maxNorms[j]);
                        else
                            tmpLabel[j] = askTreeMeanVal(point, BoostTreeArr[j], NormThreshold);
                    });

                    for (int j = 0; j < testing_label[0].Count(); j++)
                        for (int k = 0; k < BoostTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j];
                });
            }
            else
            {
                for (int i = 0; i < testing_label.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//testing_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//testing_dt[0].Count()
                        point[j] = double.Parse(testing_dt[i][j].ToString());
                    double[][] tmpLabel = new double[BoostTreeArr.Count()][];
                    for (int j = 0; j < BoostTreeArr.Count(); j++)
                    {
                        if (NormThreshold < maxNorms[j])
                            tmpLabel[j] = askTreeMeanVal(point, BoostTreeArr[j], maxNorms[j]);
                        else
                            tmpLabel[j] = askTreeMeanVal(point, BoostTreeArr[j], NormThreshold);
                    }

                    for (int j = 0; j < testing_label[0].Count(); j++)
                        for (int k = 0; k < BoostTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j];
                }
            } 

            double error = 0;
            if (NormLPType == 2)
            {
                for (int j = 0; j < testing_label[0].Count(); j++)
                    for (int i = 0; i < testing_label.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - testing_label[i][j]) * (estimatedLabels[i][j] - testing_label[i][j]);
                    }
                error = Math.Sqrt(error);
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < testing_label.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * testing_label[i][0]) <= 0)
                        error += 1;
                }
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2simplex4(estimatedLabels);

                for (int i = 0; i < testing_label.Count(); i++)
                {
                    if (0.00001 < normPoint3d(estimatedLabels[i], testing_label[i]))
                        error += 1;
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < testing_label.Count(); i++)
                {
                    if (testing_label[i][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (testing_label[i][0] == -1)
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

        private double testDecisionTreeBoostingLearningRate(double[][] testing_dt, double[][] testing_label, List<GeoWave>[] BoostTreeArr, int NormLPType, double[] maxNorms, int Ntrees)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, Ntrees, i =>
                {
                    BoostTreeArr[i] = BoostTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < Ntrees; i++)
                {
                    BoostTreeArr[i] = BoostTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[testing_label.Count()][];
            for (int i = 0; i < testing_label.Count(); i++)
                estimatedLabels[i] = new double[testing_label[0].Count()];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, testing_label.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//testing_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//testing_dt[0].Count()
                        point[j] = double.Parse(testing_dt[i][j].ToString());
                    double[][] tmpLabel = new double[BoostTreeArr.Count()][];
                    Parallel.For(0, Ntrees, j =>
                    {
                        tmpLabel[j] = askTreeMeanVal(point, BoostTreeArr[j], maxNorms[j]);
                    });

                    for (int j = 0; j < testing_label[0].Count(); j++)
                        for (int k = 0; k < Ntrees; k++)
                            estimatedLabels[i][j] += tmpLabel[k][j];

                });
            }
            else
            {
                for (int i = 0; i < testing_label.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//testing_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//testing_dt[0].Count()
                        point[j] = double.Parse(testing_dt[i][j].ToString());
                    double[][] tmpLabel = new double[BoostTreeArr.Count()][];
                    for (int j = 0; j < Ntrees; j++)
                    {
                        tmpLabel[j] = askTreeMeanVal(point, BoostTreeArr[j], maxNorms[j]);
                    }

                    for (int j = 0; j < testing_label[0].Count(); j++)
                        for (int k = 0; k < Ntrees; k++)
                            estimatedLabels[i][j] += tmpLabel[k][j];
                }
            }

            double error = 0;
            if (NormLPType == 2)
            {
                for (int j = 0; j < testing_label[0].Count(); j++)
                    for (int i = 0; i < testing_label.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - testing_label[i][j]) * (estimatedLabels[i][j] - testing_label[i][j]);
                    }
                error = Math.Sqrt(error);
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < testing_label.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * testing_label[i][0]) <= 0)
                        error += 1;
                }
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2simplex4(estimatedLabels);

                for (int i = 0; i < testing_label.Count(); i++)
                {
                    if (0.00001 < normPoint3d(estimatedLabels[i], testing_label[i]))
                        error += 1;
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < testing_label.Count(); i++)
                {
                    if (testing_label[i][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (testing_label[i][0] == -1)
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

        private void testDecisionTreeBoostingProoning(double[][] testing_dt, double[][] testing_label, List<GeoWave>[] BoostTreeArrPooning, int[] best_level, int NormLPType, double[] error)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, BoostTreeArrPooning.Count(), i =>
                {
                    BoostTreeArrPooning[i] = BoostTreeArrPooning[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < BoostTreeArrPooning.Count(); i++)
                {
                    BoostTreeArrPooning[i] = BoostTreeArrPooning[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[testing_label.Count()][][];
            for (int i = 0; i < testing_label.Count(); i++)
            {
                estimatedLabels[i] = new double[rc.boostNum][];
                for (int j = 0; j < rc.boostNum; j++ )
                    estimatedLabels[i][j] = new double[testing_label[0].Count()];
            }
            

            if (Form1.rumPrallel)
            {
                Parallel.For(0, testing_label.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//testing_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//testing_dt[0].Count()
                        point[j] = double.Parse(testing_dt[i][j].ToString());
                    double[][] tmpLabel = new double[BoostTreeArrPooning.Count()][];
                    Parallel.For(0, BoostTreeArrPooning.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, BoostTreeArrPooning[j], best_level[j]);
                    });

                    for (int j = 0; j < testing_label[0].Count(); j++)
                    {
                        double tmp = 0;
                        for (int k = 0; k < BoostTreeArrPooning.Count(); k++)
                        {
                            estimatedLabels[i][k][j] = tmp + tmpLabel[k][j];
                            tmp += tmpLabel[k][j];
                        }
                    }  
                });
            }
            else
            {
                for (int i = 0; i < testing_label.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//testing_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//testing_dt[0].Count()
                        point[j] = double.Parse(testing_dt[i][j].ToString());
                    double[][] tmpLabel = new double[BoostTreeArrPooning.Count()][];
                    for (int j = 0; j < BoostTreeArrPooning.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, BoostTreeArrPooning[j], best_level[j]);
                    }

                    for (int j = 0; j < testing_label[0].Count(); j++)
                    {
                        double tmp = 0;
                        for (int k = 0; k < BoostTreeArrPooning.Count(); k++)
                        {
                            estimatedLabels[i][k][j] = tmp + tmpLabel[k][j];
                            tmp += tmpLabel[k][j];
                        }
                    }  
                }
            }

            if (NormLPType == 2)
            {
                for (int k = 0; k < rc.boostNum; k++)
                {
                    for (int j = 0; j < testing_label[0].Count(); j++)
                        for (int i = 0; i < testing_label.Count(); i++)
                            error[k] += (estimatedLabels[i][k][j] - testing_label[i][j]) * (estimatedLabels[i][k][j] - testing_label[i][j]);
                    error[k] = Math.Sqrt(error[k]);                
                }

            }
            else if (NormLPType == 0 && estimatedLabels[0][0].Count() == 1)//+-1 labels
            {
                for (int k = 0; k < rc.boostNum; k++)
                {
                    for (int i = 0; i < testing_label.Count(); i++)
                    {
                        if ((estimatedLabels[i][k][0] * testing_label[i][0]) <= 0)
                            error[k] += 1;
                    }
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0][0].Count() == 1)//+-1 labels + BER
            {
                double[] NclassA = new double[rc.boostNum];
                double[] NclassB = new double[rc.boostNum];
                double[] NMissclassA = new double[rc.boostNum];
                double[] NMissclassB = new double[rc.boostNum];

                for (int k = 0; k < rc.boostNum; k++)
                {
                    for (int i = 0; i < testing_label.Count(); i++)
                    {
                        if (testing_label[i][0] == 1)
                        {
                            NclassA[k] += 1;
                            if (estimatedLabels[i][k][0] <= 0)
                                NMissclassA[k] += 1;
                        }
                        if (testing_label[i][0] == -1)
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

        private List<int> Bagging(List<int> trainingArr, double percent, int Seed)//percent in [0,1]
        {
            //List<int> baggedArr = new List<int>();
            int N_rows = Convert.ToInt32(trainingArr.Count * percent);
            //int Seed = (int)DateTime.Now.Ticks;
            var ran = new Random(Seed);
            return Enumerable.Range(0, trainingArr.Count).OrderBy(x => ran.Next()).ToList().GetRange(0, N_rows);
        }

        private int getGWIndexByKfunc(List<GeoWave> tmp_Tree_orderedByNorm,
                                         recordConfig rc,
                                         double[][] trainingData, 
                                         double[][] trainingLabel,
                                         ref double best_norm)
        {
            //double[] best_index_norm = new double[2];//returned value ...
            int NumOfSkips = Convert.ToInt16(1 / rc.NskipsinKfunc);
            int skipSize = Convert.ToInt16(Math.Floor(rc.NskipsinKfunc * tmp_Tree_orderedByNorm.Count));

            if (skipSize * NumOfSkips > tmp_Tree_orderedByNorm.Count)
                MessageBox.Show("skipping made us go out of range - shuold not get here");

            double[] errArr = new double[NumOfSkips-1];

            ////DO THE HOPPING/SKIPPING
            if (Form1.rumPrallel)
            {
                Parallel.For(1, NumOfSkips, i =>
                {
                    double thresholdNorm = tmp_Tree_orderedByNorm[i * skipSize].norm;
                    double Tgt_approx_error = testDecisionTree(trainingData, trainingLabel, tmp_Tree_orderedByNorm, thresholdNorm, rc.boostNormTarget);
                    double geowave_total_norm = getgeowaveNorm(tmp_Tree_orderedByNorm, i * skipSize, rc.boostNormsecond, rc.boostTau);
                    errArr[i - 1] = Tgt_approx_error + (rc.boostlamda_0 * geowave_total_norm);
                });
            }
            else
            {
                for (int i = 1; i < NumOfSkips; i++)
                {
                    double thresholdNorm = tmp_Tree_orderedByNorm[i * skipSize].norm;
                    double Tgt_approx_error = testDecisionTree(trainingData, trainingLabel, tmp_Tree_orderedByNorm, thresholdNorm, rc.boostNormTarget);
                    double geowave_total_norm = getgeowaveNorm(tmp_Tree_orderedByNorm, i * skipSize, rc.boostNormsecond, rc.boostTau);
                    errArr[i - 1] = Tgt_approx_error + (rc.boostlamda_0 * geowave_total_norm);
                }
            } 

            int best_index = Enumerable.Range(0, errArr.Length).Aggregate((a, b) => (errArr[a] < errArr[b]) ? a : b); //minerror

            int first_index, last_index;
            if (best_index == 0)
            {
                first_index = 0;
                last_index = Math.Min(2 * skipSize, tmp_Tree_orderedByNorm.Count);
            }
            else if (best_index == (NumOfSkips - 2))
            {
                first_index = Math.Max((best_index) * skipSize, 0);
                last_index = tmp_Tree_orderedByNorm.Count;
            }
            else
            {
                first_index = Math.Max((best_index) * skipSize, 0);
                last_index = Math.Min((best_index + 2) * skipSize, tmp_Tree_orderedByNorm.Count);
            }

            errArr = new double[last_index - first_index];

            //SEARCH IN THE BOUNDING 
            if (Form1.rumPrallel)
            {
                Parallel.For(first_index, last_index, i =>
                {
                    double thresholdNorm = tmp_Tree_orderedByNorm[i].norm;
                    double Tgt_approx_error = testDecisionTree(trainingData, trainingLabel, tmp_Tree_orderedByNorm, thresholdNorm, rc.boostNormTarget);
                    double geowave_total_norm = getgeowaveNorm(tmp_Tree_orderedByNorm, i, rc.boostNormsecond, rc.boostTau);
                    errArr[i - first_index] = Tgt_approx_error + (rc.boostlamda_0 * geowave_total_norm);
                });
            }
            else
            {
                for (int i = first_index; i < last_index; i++)
                {
                    double thresholdNorm = tmp_Tree_orderedByNorm[i].norm;
                    double Tgt_approx_error = testDecisionTree(trainingData, trainingLabel, tmp_Tree_orderedByNorm, thresholdNorm, rc.boostNormTarget);
                    double geowave_total_norm = getgeowaveNorm(tmp_Tree_orderedByNorm, i, rc.boostNormsecond, rc.boostTau);
                    errArr[i - first_index] = Tgt_approx_error + (rc.boostlamda_0 * geowave_total_norm);
                }
            } 

            best_index = Enumerable.Range(0, errArr.Length).Aggregate((a, b) => (errArr[a] < errArr[b]) ? a : b); //minerror
            best_norm = tmp_Tree_orderedByNorm[first_index + best_index].norm;

            return (first_index+ best_index);//indicates the number of waveletes to take (calced in order by ID)
        }

        private int getGWIndexByKfuncLessAcurate(List<GeoWave> tmp_Tree_orderedByNorm,
                                         recordConfig rc,
                                         double[][] trainingData,
                                         double[][] trainingLabel,
                                         ref double best_norm)
        {
            //double[] best_index_norm = new double[2];//returned value ...
            int skipSize  = Convert.ToInt16(1 / rc.NskipsinKfunc);
            int NumOfSkips  = Convert.ToInt16(Math.Floor(rc.NskipsinKfunc * tmp_Tree_orderedByNorm.Count));

            if (skipSize * NumOfSkips > tmp_Tree_orderedByNorm.Count)
                MessageBox.Show("skipping made us go out of range - shuold not get here");

            double[] errArr = new double[NumOfSkips];

            ////DO THE HOPPING/SKIPPING
            if (Form1.rumPrallel)
            {
                Parallel.For(0, NumOfSkips, i =>
                {
                    double thresholdNorm = tmp_Tree_orderedByNorm[i * skipSize].norm;
                    double Tgt_approx_error = testDecisionTree(trainingData, trainingLabel, tmp_Tree_orderedByNorm, thresholdNorm, rc.boostNormTarget);
                    double geowave_total_norm = getgeowaveNorm(tmp_Tree_orderedByNorm, i * skipSize, rc.boostNormsecond, rc.boostTau);
                    if (rc.boostNormsecond == 0)
                        geowave_total_norm += 1;
                    errArr[i] = Tgt_approx_error + (rc.boostlamda_0 * geowave_total_norm);
                });
            }
            else
            {
                for (int i = 0; i < NumOfSkips; i++)
                {
                    double thresholdNorm = tmp_Tree_orderedByNorm[i * skipSize].norm;
                    double Tgt_approx_error = testDecisionTree(trainingData, trainingLabel, tmp_Tree_orderedByNorm, thresholdNorm, rc.boostNormTarget);
                    double geowave_total_norm = getgeowaveNorm(tmp_Tree_orderedByNorm, i * skipSize, rc.boostNormsecond, rc.boostTau);
                    if (rc.boostNormsecond == 0)
                        geowave_total_norm += 1;

                    errArr[i] = Tgt_approx_error + (rc.boostlamda_0 * geowave_total_norm);
                }
            }

            int best_index = Enumerable.Range(0, errArr.Length).Aggregate((a, b) => (errArr[a] < errArr[b]) ? a : b); //minerror

            return best_index * skipSize;
        }

        private int getGWIndexByKfuncLessAcuratePooning(List<GeoWave> BoostedTreeArrPooning, recordConfig rc, double[][] training_dt, double[][] boostedLabelsPooning)
        {
            int topLevel = getTopLevel(BoostedTreeArrPooning);
            double[] errorTreeProoning = new double[topLevel];
            double[] NLevels = new double[topLevel];
            double[] errArr = new double[topLevel];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, topLevel, i =>
                {
                    errorTreeProoning[i] = testDecisionTreeWithProoning(db.PCAtesting_dt, db.testing_label, BoostedTreeArrPooning, i + 1, 2);
                    NLevels[i] = i;// * rc.hopping_size;
                    errArr[i] = errorTreeProoning[i] + (NLevels[i] * rc.boostProoning_0);
                });
            }
            else
            {
                for (int i = 0; i < topLevel; i++)
                {
                    errorTreeProoning[i] = testDecisionTreeWithProoning(db.PCAtesting_dt, db.testing_label, BoostedTreeArrPooning, i + 1, 2);
                    NLevels[i] = i;// *rc.hopping_size;
                    errArr[i] = errorTreeProoning[i] + (NLevels[i] * rc.boostProoning_0);
                }
            }

            int best_level = Enumerable.Range(0, errArr.Length).Aggregate((a, b) => (errArr[a] < errArr[b]) ? a : b); //minerror

            return best_level;
        }
        
        private double testDecisionTree(double[][] Data_table, double[][] Data_Lables, List<GeoWave> Tree_orderedById, double NormThreshold, int NormLPType)
        {
            Tree_orderedById = Tree_orderedById.OrderBy(o => o.ID).ToList();

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[Data_Lables.Count()][];
            for (int i = 0; i < Data_Lables.Count() ; i++)
                estimatedLabels[i] = new double[Data_Lables[0].Count()];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, Data_Lables.Count(), i =>
                {
                    estimatedLabels[i] = askTreeMeanVal(Data_table[i], Tree_orderedById, NormThreshold);
                });
            }
            else
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    estimatedLabels[i] = askTreeMeanVal(Data_table[i], Tree_orderedById, NormThreshold);
                }
            } 


            double error = 0;
            if (NormLPType == 2)
            { 
                for(int j=0; j < Data_Lables[0].Count() ; j++)
                    for (int i = 0; i < Data_Lables.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - Data_Lables[i][j]) * (estimatedLabels[i][j] - Data_Lables[i][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(Data_Lables.Count()));
            }
            else if (NormLPType == 1)
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < Data_Lables.Count(); i++)
                    {
                        error += Math.Abs(estimatedLabels[i][j] - Data_Lables[i][j]) ;
                    }
            }
            else if (NormLPType == -1)//max
            {
                List<double> errList = new List<double>();
                double tmp = 0;
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    tmp = 0;
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        tmp += Math.Abs(estimatedLabels[i][j] - Data_Lables[i][j]);
                    }
                    errList.Add(tmp);
                }
                error = errList.Max();
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {                
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if ( (estimatedLabels[i][0] * Data_Lables[i][0]) <= 0 )
                        error += 1;
                }                
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2simplex4(estimatedLabels);

                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if (0.00001 < normPoint3d(estimatedLabels[i], Data_Lables[i]))
                        error += 1;
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;
                
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if (Data_Lables[i][0] == 1)
                    {
                        NclassA += 1;
                        if(estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (Data_Lables[i][0] == -1)
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

        private double testDecisionTreeWithProoning(double[][] Data_table, double[][] Data_Lables, List<GeoWave> Tree_orderedById, int topLevel, int NormLPType)
        {
            Tree_orderedById = Tree_orderedById.OrderBy(o => o.ID).ToList();

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[Data_Lables.Count()][];
            for (int i = 0; i < Data_Lables.Count() ; i++)
                estimatedLabels[i] = new double[Data_Lables[0].Count()];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, Data_Lables.Count(), i =>
                {
                    estimatedLabels[i] = askTreeMeanValAtLevel(Data_table[i], Tree_orderedById, topLevel);
                });
            }
            else
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    estimatedLabels[i] = askTreeMeanValAtLevel(Data_table[i], Tree_orderedById, topLevel);
                }
            } 

            double error = 0;
            if (NormLPType == 2)
            { 
                for(int j=0; j < Data_Lables[0].Count() ; j++)
                    for (int i = 0; i < Data_Lables.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - Data_Lables[i][j]) * (estimatedLabels[i][j] - Data_Lables[i][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(Data_Lables.Count()));
            }
            else if (NormLPType == 1)
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < Data_Lables.Count(); i++)
                    {
                        error += Math.Abs(estimatedLabels[i][j] - Data_Lables[i][j]) ;
                    }
            }
            else if (NormLPType == -1)//max
            {
                List<double> errList = new List<double>();
                double tmp = 0;
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    tmp = 0;
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        tmp += Math.Abs(estimatedLabels[i][j] - Data_Lables[i][j]);
                    }
                    errList.Add(tmp);
                }
                error = errList.Max();
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * Data_Lables[i][0]) <= 0)
                        error += 1;
                }
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2simplex4(estimatedLabels);

                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if (0.00001 < normPoint3d(estimatedLabels[i], Data_Lables[i]))
                        error += 1;
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if (Data_Lables[i][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (Data_Lables[i][0] == -1)
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

        private void adjustlabels2simplex4(double[][] estimatedLabels)
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
                    dist[k] = normPoint3d(estimatedLabels[i], Data_Lables[k]);
                int minIndex = Array.IndexOf(dist, dist.Min());
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

        private double normPoint3d(double[] p, double[] p_2)
        {
            double norm = 0;
            for (int i = 0; i < p.Count(); i++)
                norm += (p[i] - p_2[i]) * (p[i] - p_2[i]);
            return norm;
        }

        private double testDecisionTreeRF(double[][] Data_table, double[][] Data_Lables, List<GeoWave>[] RFdecTreeArr , double NormThreshold, int NormLPType)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, RFdecTreeArr.Count(), i =>
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }                         

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[Data_Lables.Count()][];
            for (int i = 0; i < Data_Lables.Count(); i++)
                estimatedLabels[i] = new double[Data_Lables[0].Count()];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, Data_Lables.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];
                    Parallel.For(0, RFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArr[j], NormThreshold);
                    });

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(RFdecTreeArr.Count());
                });
            }
            else
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];

                    for (int j = 0; j < RFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArr[j], NormThreshold);
                    }

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(RFdecTreeArr.Count());
                }
            } 

            double error = 0;
            if (NormLPType == 2)
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < Data_Lables.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - Data_Lables[i][j]) * (estimatedLabels[i][j] - Data_Lables[i][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(Data_Lables.Count()));
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * Data_Lables[i][0]) <= 0)
                        error += 1;
                }
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2simplex4(estimatedLabels);

                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if (0.00001 < normPoint3d(estimatedLabels[i], Data_Lables[i]))
                        error += 1;
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if (Data_Lables[i][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (Data_Lables[i][0] == -1)
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

        private double testDecisionTreeManyRFNormNbound(double[][] Data_table, double[][] Data_Lables, List<GeoWave>[] RFdecTreeArr, double NormThreshold, int boundLevel, int NormLPType)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, RFdecTreeArr.Count(), i =>
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }                         

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[Data_Lables.Count()][];
            for (int i = 0; i < Data_Lables.Count(); i++)
                estimatedLabels[i] = new double[Data_Lables[0].Count()];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, Data_Lables.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];
                    Parallel.For(0, RFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanValBoundLevel(point, RFdecTreeArr[j], NormThreshold,boundLevel);
                    });

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(RFdecTreeArr.Count());
                });
            }
            else
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];

                    for (int j = 0; j < RFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanValBoundLevel(point, RFdecTreeArr[j], NormThreshold, boundLevel);
                    }

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(RFdecTreeArr.Count());
                }
            } 

            double error = 0;
            if (NormLPType == 2)
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < Data_Lables.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - Data_Lables[i][j]) * (estimatedLabels[i][j] - Data_Lables[i][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(Data_Lables.Count()));
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * Data_Lables[i][0]) <= 0)
                        error += 1;
                }
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2simplex4(estimatedLabels);

                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if (0.00001 < normPoint3d(estimatedLabels[i], Data_Lables[i]))
                        error += 1;
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if (Data_Lables[i][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (Data_Lables[i][0] == -1)
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
        
        private double[] testDecisionTreeManyRF(double[][] Data_table, double[][] Data_Lables, List<GeoWave>[] RFdecTreeArr, double NormThreshold, int NormLPType)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, RFdecTreeArr.Count(), i =>
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[RFdecTreeArr.Count()][][];//num of trees, label index, label values (or value in most cases)
            for (int i = 0; i < RFdecTreeArr.Count(); i++)
            {
                estimatedLabels[i] = new double[Data_Lables.Count()][];
                for (int j = 0; j < Data_Lables.Count(); j++)
                    estimatedLabels[i][j] = new double[Data_Lables[0].Count()];
            }

            if (Form1.rumPrallel)
            {
                Parallel.For(0, Data_Lables.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];
                    Parallel.For(0, RFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArr[j], NormThreshold);
                    });

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        estimatedLabels[0][i][j] = tmpLabel[0][j];
                        for (int k = 1; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = (Convert.ToDouble(k) / (Convert.ToDouble(k) + 1)) * estimatedLabels[k - 1][i][j] + (1 / (Convert.ToDouble(k) + 1)) * tmpLabel[k][j];              
                    }
                });
            }
            else
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];

                    for (int j = 0; j < RFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArr[j], NormThreshold);
                    }


                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        estimatedLabels[0][i][j] = tmpLabel[0][j];
                        for (int k = 1; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = (Convert.ToDouble(k) / (Convert.ToDouble(k) + 1)) * estimatedLabels[k - 1][i][j] + (1 / (Convert.ToDouble(k) + 1)) * tmpLabel[k][j];    
                    }
                }
            }

            double[] error = new double[RFdecTreeArr.Count()];
            if (NormLPType == 2)
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < Data_Lables.Count(); i++)
                            error[k] += (estimatedLabels[k][i][j] - Data_Lables[i][j]) * (estimatedLabels[k][i][j] - Data_Lables[i][j]);
                    error[k] = Math.Sqrt(error[k]/ Convert.ToDouble(Data_Lables.Count()));
                }
            }
            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double[] testDecisionTreeManyRFbyIndex(double[][] Data_table, double[][] Data_Lables, List<GeoWave>[] RFdecTreeArr, int IndexThreshold, int NormLPType)
        {
            List<GeoWave>[] RFdecTreeArrById = new List<GeoWave>[RFdecTreeArr.Count()];
            if (Form1.rumPrallel)
            {
                Parallel.For(0, RFdecTreeArr.Count(), i =>
                {
                    RFdecTreeArrById[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                {
                    RFdecTreeArrById[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[RFdecTreeArr.Count()][][];//num of trees, label index, label values (or value in most cases)
            for (int i = 0; i < RFdecTreeArr.Count(); i++)
            {
                estimatedLabels[i] = new double[Data_Lables.Count()][];
                for (int j = 0; j < Data_Lables.Count(); j++)
                    estimatedLabels[i][j] = new double[Data_Lables[0].Count()];
            }

            if (Form1.rumPrallel)
            {
                Parallel.For(0, Data_Lables.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];
                    Parallel.For(0, RFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArrById[j], RFdecTreeArr[j][IndexThreshold].norm);
                    });

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        estimatedLabels[0][i][j] = tmpLabel[0][j];
                        for (int k = 1; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = (Convert.ToDouble(k) / (Convert.ToDouble(k) + 1)) * estimatedLabels[k - 1][i][j] + (1 / (Convert.ToDouble(k) + 1)) * tmpLabel[k][j];              
                    }
                });
            }
            else
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];

                    for (int j = 0; j < RFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArrById[j], RFdecTreeArr[j][IndexThreshold].norm);
                    }


                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        estimatedLabels[0][i][j] = tmpLabel[0][j];
                        for (int k = 1; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = (Convert.ToDouble(k) / (Convert.ToDouble(k) + 1)) * estimatedLabels[k - 1][i][j] + (1 / (Convert.ToDouble(k) + 1)) * tmpLabel[k][j];    
                    }
                }
            }

            double[] error = new double[RFdecTreeArr.Count()];
            if (NormLPType == 2)
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < Data_Lables.Count(); i++)
                            error[k] += (estimatedLabels[k][i][j] - Data_Lables[i][j]) * (estimatedLabels[k][i][j] - Data_Lables[i][j]);
                    error[k] = Math.Sqrt(error[k] / Convert.ToDouble(Data_Lables.Count()));
                }
            }
            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }
        
        private double testDecisionTreeRF(double[][] Data_table, double[][] Data_Lables, List<GeoWave>[] RFdecTreeArr, int topLevel, int NormLPType)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, RFdecTreeArr.Count(), i =>
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[Data_Lables.Count()][];
            for (int i = 0; i < Data_Lables.Count(); i++)
                estimatedLabels[i] = new double[Data_Lables[0].Count()];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, Data_Lables.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];
                    Parallel.For(0, RFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, RFdecTreeArr[j], topLevel);
                    });

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(RFdecTreeArr.Count());
                });
            }
            else
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];

                    for (int j = 0; j < RFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, RFdecTreeArr[j], topLevel);
                    }

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(RFdecTreeArr.Count());
                }
            }

            double error = 0;
            if (NormLPType == 2)
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < Data_Lables.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - Data_Lables[i][j]) * (estimatedLabels[i][j] - Data_Lables[i][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(Data_Lables.Count()));
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * Data_Lables[i][0]) <= 0)
                        error += 1;
                }
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2simplex4(estimatedLabels);

                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if (0.00001 < normPoint3d(estimatedLabels[i], Data_Lables[i]))
                        error += 1;
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if (Data_Lables[i][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (Data_Lables[i][0] == -1)
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

        private double[] testDecisionTreeManyRF(double[][] Data_table, double[][] Data_Lables, List<GeoWave>[] RFdecTreeArr, int topLevel, int NormLPType)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, RFdecTreeArr.Count(), i =>
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[RFdecTreeArr.Count()][][];//num of trees, label index, label values (or value in most cases)
            for (int i = 0; i < RFdecTreeArr.Count(); i++)
            {
                estimatedLabels[i] = new double[Data_Lables.Count()][];
                for (int j = 0; j < Data_Lables.Count(); j++)
                    estimatedLabels[i][j] = new double[Data_Lables[0].Count()];
            }

            if (Form1.rumPrallel)
            {
                Parallel.For(0, Data_Lables.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];
                    Parallel.For(0, RFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, RFdecTreeArr[j], topLevel);
                    });

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        estimatedLabels[0][i][j] = tmpLabel[0][j];
                        for (int k = 1; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = (Convert.ToDouble(k) / (Convert.ToDouble(k) + 1)) * estimatedLabels[k - 1][i][j] + (1 / (Convert.ToDouble(k) + 1)) * tmpLabel[k][j];    
                    }
                });
            }
            else
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];

                    for (int j = 0; j < RFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, RFdecTreeArr[j], topLevel);
                    }

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        estimatedLabels[0][i][j] = tmpLabel[0][j];
                        for (int k = 1; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = (Convert.ToDouble(k) / (Convert.ToDouble(k) + 1)) * estimatedLabels[k - 1][i][j] + (1 / (Convert.ToDouble(k) + 1)) * tmpLabel[k][j];    
                    }
                }
            }

            double[] error = new double[RFdecTreeArr.Count()];
            if (NormLPType == 2)
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < Data_Lables.Count(); i++)
                            error[k] += (estimatedLabels[k][i][j] - Data_Lables[i][j]) * (estimatedLabels[k][i][j] - Data_Lables[i][j]);
                    error[k] = Math.Sqrt(error[k] / Convert.ToDouble(Data_Lables.Count()));
                }
            }
            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double getgeowaveNorm(List<GeoWave> tmp_Tree_orderedByNorm, int Nwavelets, int NormSecond, int orderTau)
        {
            double norm = 0;
            if (NormSecond == 0)
                return 1.0 * Nwavelets;// I dont add +1 because if I root estimation I want to give norm 0
            else if (orderTau == 1)
            {
                for (int i = 0; i <= Nwavelets; i++)
                    norm += tmp_Tree_orderedByNorm[i].norm;
                return norm;            
            }
            else if (orderTau == 2)
            {
                for (int i = 0; i <= Nwavelets; i++)
                    norm += (tmp_Tree_orderedByNorm[i].norm) * (tmp_Tree_orderedByNorm[i].norm);
                return Math.Sqrt(norm);
            }
            else
            {
                for (int i = 0; i <= Nwavelets; i++)
                    norm += Math.Pow(tmp_Tree_orderedByNorm[i].norm, orderTau);
                return Math.Pow(norm, 1 / Convert.ToDouble(orderTau));
            }
        }

        //private double[] askTreeMeanVal(double[] point, List<GeoWave> Tree_orderedById, double NormThreshold)
        //{
        //    //if (point.Count() != Tree_orderedById[0].boubdingBox[0].Count())
        //    //{
        //    //    MessageBox.Show("the dim of the point is not compatible with the dim of the tree");
        //    //    return null;
        //    //}

        //    int counter = 0;
        //    if (!DB.IsPntInsideBox(Tree_orderedById[0].boubdingBox, point, rc.dim))
        //    {
        //        DB.ProjectPntInsideBox(Tree_orderedById[0].boubdingBox, ref point);
        //        counter++;
        //    }

        //    double[] zeroMean = new double[Tree_orderedById[0].MeanValue.Count()];
        //    double[] MeanValue = new double[Tree_orderedById[0].MeanValue.Count()];

        //    //SET THE ROOT MEAN VAL
        //    Tree_orderedById[0].MeanValue.CopyTo(MeanValue, 0);

        //    ////get to leaf 

        //    int parent_index = 0;

        //    while (Tree_orderedById[parent_index].child0 != -1)
        //    {
        //        if (DB.IsPntInsideBox(Tree_orderedById[Tree_orderedById[parent_index].child0].boubdingBox, point, rc.dim))
        //        {
        //            if (!Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue.SequenceEqual(zeroMean) &&
        //                NormThreshold <= Tree_orderedById[Tree_orderedById[parent_index].child0].norm) //take the mean value if its not 0 and the wavelete should be taken ( norm size) - or if its the root wavelete
        //            {
        //                Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue.CopyTo(MeanValue, 0);
        //            }

        //            parent_index = Tree_orderedById[parent_index].child0;
        //        }
        //        else
        //        {
        //            if (!Tree_orderedById[Tree_orderedById[parent_index].child1].MeanValue.SequenceEqual(zeroMean) &&
        //                NormThreshold <= Tree_orderedById[Tree_orderedById[parent_index].child1].norm) //take the mean value if its not 0 and the wavelete should be taken ( norm size) - or if its the root wavelete
        //            {
        //                Tree_orderedById[Tree_orderedById[parent_index].child1].MeanValue.CopyTo(MeanValue, 0);
        //            }

        //            parent_index = Tree_orderedById[parent_index].child1;
        //        }
        //    }
        //    return MeanValue;
        //}

        private double[] askTreeMeanVal(double[] point, List<GeoWave> Tree_orderedById, double NormThreshold)
        {

            int counter = 0;
            if (!DB.IsPntInsideBox(Tree_orderedById[0].boubdingBox, point, rc.dim))
            {
                DB.ProjectPntInsideBox(Tree_orderedById[0].boubdingBox, ref point);
                counter++;
            }

            double[] zeroMean = new double[Tree_orderedById[0].MeanValue.Count()];
            double[] MeanValue = new double[Tree_orderedById[0].MeanValue.Count()];

            //SET THE ROOT MEAN VAL
            Tree_orderedById[0].MeanValue.CopyTo(MeanValue, 0);

            ////get to leaf 

            int parent_index = 0;

            while (Tree_orderedById[parent_index].child0 != -1)
            {
                if (DB.IsPntInsideBox(Tree_orderedById[Tree_orderedById[parent_index].child0].boubdingBox, point, rc.dim))
                {
                    if (!Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue.SequenceEqual(zeroMean) &&
                        NormThreshold <= Tree_orderedById[Tree_orderedById[parent_index].child0].norm) //take the mean value if its not 0 and the wavelete should be taken ( norm size) - or if its the root wavelete
                    {
                        //Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue.CopyTo(MeanValue, 0);
                        //MeanValue = (Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue.Subtract(Tree_orderedById[parent_index].MeanValue)).Add(MeanValue);
                        MeanValue[0] += (Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue[0] - Tree_orderedById[parent_index].MeanValue[0]);
                    }

                    parent_index = Tree_orderedById[parent_index].child0;
                }
                else
                {
                    if (!Tree_orderedById[Tree_orderedById[parent_index].child1].MeanValue.SequenceEqual(zeroMean) &&
                        NormThreshold <= Tree_orderedById[Tree_orderedById[parent_index].child1].norm) //take the mean value if its not 0 and the wavelete should be taken ( norm size) - or if its the root wavelete
                    {
                        //Tree_orderedById[Tree_orderedById[parent_index].child1].MeanValue.CopyTo(MeanValue, 0);
                        //MeanValue = (Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue.Subtract(Tree_orderedById[parent_index].MeanValue)).Add(MeanValue);
                        MeanValue[0] += (Tree_orderedById[Tree_orderedById[parent_index].child1].MeanValue[0] - Tree_orderedById[parent_index].MeanValue[0]);
                    }

                    parent_index = Tree_orderedById[parent_index].child1;
                }
            }
            return MeanValue;
        }
        private double[] askTreeMeanValBoundLevel(double[] point, List<GeoWave> Tree_orderedById, double NormThreshold, int BoundLevel)
        {
            //if (point.Count() != Tree_orderedById[0].boubdingBox[0].Count())
            //{
            //    MessageBox.Show("the dim of the point is not compatible with the dim of the tree");
            //    return null;
            //}

            int counter = 0;
            if (!DB.IsPntInsideBox(Tree_orderedById[0].boubdingBox, point, rc.dim))
            {
                DB.ProjectPntInsideBox(Tree_orderedById[0].boubdingBox, ref point);
                counter++;
            }

            double[] zeroMean = new double[Tree_orderedById[0].MeanValue.Count()];
            double[] MeanValue = new double[Tree_orderedById[0].MeanValue.Count()];

            //SET THE ROOT MEAN VAL
            Tree_orderedById[0].MeanValue.CopyTo(MeanValue, 0);

            ////get to leaf 

            int parent_index = 0;

            while (Tree_orderedById[parent_index].child0 != -1 && Tree_orderedById[parent_index].level <= BoundLevel)
            {
                if (DB.IsPntInsideBox(Tree_orderedById[Tree_orderedById[parent_index].child0].boubdingBox, point, rc.dim))
                {
                    if (!Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue.SequenceEqual(zeroMean) &&
                        NormThreshold <= Tree_orderedById[Tree_orderedById[parent_index].child0].norm) //take the mean value if its not 0 and the wavelete should be taken ( norm size) - or if its the root wavelete
                    {
                        //Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue.CopyTo(MeanValue, 0);
                        MeanValue[0] += (Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue[0] - Tree_orderedById[parent_index].MeanValue[0]);
                    }

                    parent_index = Tree_orderedById[parent_index].child0;
                }
                else
                {
                    if (!Tree_orderedById[Tree_orderedById[parent_index].child1].MeanValue.SequenceEqual(zeroMean) &&
                        NormThreshold <= Tree_orderedById[Tree_orderedById[parent_index].child1].norm) //take the mean value if its not 0 and the wavelete should be taken ( norm size) - or if its the root wavelete
                    {
                        //Tree_orderedById[Tree_orderedById[parent_index].child1].MeanValue.CopyTo(MeanValue, 0);
                        MeanValue[0] += (Tree_orderedById[Tree_orderedById[parent_index].child1].MeanValue[0] - Tree_orderedById[parent_index].MeanValue[0]);
                    }

                    parent_index = Tree_orderedById[parent_index].child1;
                }
            }
            return MeanValue;
        }

        private double[] askTreeMeanValAtLevel(double[] point, List<GeoWave> Tree_orderedById, int topLevel)
        {
            //if (point.Count() != Tree_orderedById[0].boubdingBox[0].Count())
            //{
            //    MessageBox.Show("the dim of the point is not compatible with the dim of the tree");
            //    return null;
            //}

            int counter = 0;
            if (!DB.IsPntInsideBox(Tree_orderedById[0].boubdingBox, point, rc.dim))
            {
                DB.ProjectPntInsideBox(Tree_orderedById[0].boubdingBox, ref point);
                counter++;
            }

            double[] zeroMean = new double[Tree_orderedById[0].MeanValue.Count()];
            double[] MeanValue = new double[Tree_orderedById[0].MeanValue.Count()];

            //SET THE ROOT MEAN VAL
            Tree_orderedById[0].MeanValue.CopyTo(MeanValue, 0);

            ////get to leaf 

            int parent_index = 0;

            while (Tree_orderedById[parent_index].child0 != -1)
            {
                if (DB.IsPntInsideBox(Tree_orderedById[Tree_orderedById[parent_index].child0].boubdingBox, point, rc.dim))
                {
                    if (!Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue.SequenceEqual(zeroMean) &&
                        topLevel >= Tree_orderedById[Tree_orderedById[parent_index].child0].level) //take the mean value if its not 0 and the wavelete should be taken ( norm size) - or if its the root wavelete
                    {
                        Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue.CopyTo(MeanValue, 0);
                    }

                    parent_index = Tree_orderedById[parent_index].child0;
                }
                else
                {
                    if (!Tree_orderedById[Tree_orderedById[parent_index].child1].MeanValue.SequenceEqual(zeroMean) &&
                        topLevel >= Tree_orderedById[Tree_orderedById[parent_index].child1].level) //take the mean value if its not 0 and the wavelete should be taken ( norm size) - or if its the root wavelete
                    {
                        Tree_orderedById[Tree_orderedById[parent_index].child1].MeanValue.CopyTo(MeanValue, 0);
                    }

                    parent_index = Tree_orderedById[parent_index].child1;
                }
            }
            return MeanValue;
        }

        public double[][] GetResidualLabelsInBoosting(List<GeoWave> Tree, double[][] training_dt, double[][] boostedLabels, double threshNorm)
        {
            List<GeoWave> Tree_orderedById = Tree.OrderBy(o => o.ID).ToList();

            if (Form1.rumPrallel)
            {
                Parallel.For(0, boostedLabels.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//training_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//training_dt[0].Count()
                        point[j] = double.Parse(training_dt[i][j].ToString());

                    double[] tmpLabel = askTreeMeanVal(point, Tree_orderedById, threshNorm);
                    for (int j = 0; j < tmpLabel.Count(); j++)
                        boostedLabels[i][j] = boostedLabels[i][j] - tmpLabel[j];
                });
            }
            else
            {
                for (int i = 0; i < boostedLabels.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//training_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//training_dt[0].Count()
                        point[j] = double.Parse(training_dt[i][j].ToString());

                    double[] tmpLabel = askTreeMeanVal(point, Tree_orderedById, threshNorm);
                    for (int j = 0; j < tmpLabel.Count(); j++)
                        boostedLabels[i][j] = boostedLabels[i][j] - tmpLabel[j];
                }
            } 

            return boostedLabels;
        }

        //public double[][] GetResidualLabelsInBoostingLearningRate(List<GeoWave> Tree, double[][] training_dt, double[][] boostedLabels, double threshNorm, double LearningRate)
        //{
        //    List<GeoWave> Tree_orderedById = Tree.OrderBy(o => o.ID).ToList();

        //    if (Form1.rumPrallel)
        //    {
        //        Parallel.For(0, boostedLabels.Count(), i =>
        //        {
        //            //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
        //            double[] point = new double[rc.dim];//training_dt[0].Count()
        //            //Data_table.CopyTo(point, i);
        //            for (int j = 0; j < rc.dim; j++)//training_dt[0].Count()
        //                point[j] = double.Parse(training_dt[i][j].ToString());

        //            double[] tmpLabel = askTreeMeanVal(point, Tree_orderedById, threshNorm);
        //            for (int j = 0; j < tmpLabel.Count(); j++)
        //                boostedLabels[i][j] = boostedLabels[i][j] - LearningRate*tmpLabel[j];
        //        });
        //    }
        //    else
        //    {
        //        for (int i = 0; i < boostedLabels.Count(); i++)
        //        {
        //            //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
        //            double[] point = new double[rc.dim];//training_dt[0].Count()
        //            //Data_table.CopyTo(point, i);
        //            for (int j = 0; j < rc.dim; j++)//training_dt[0].Count()
        //                point[j] = double.Parse(training_dt[i][j].ToString());

        //            double[] tmpLabel = askTreeMeanVal(point, Tree_orderedById, threshNorm);
        //            for (int j = 0; j < tmpLabel.Count(); j++)
        //                boostedLabels[i][j] = boostedLabels[i][j] - LearningRate * tmpLabel[j];
        //        }
        //    }

        //    return boostedLabels;
        //}

        private double[][] GetResidualLabelsInBoostingProoning(List<GeoWave> Tree, double[][] training_dt, double[][] boostedLabelsPooning, int best_level)
        {
            List<GeoWave> Tree_orderedById = Tree.OrderBy(o => o.ID).ToList();

            if (Form1.rumPrallel)
            {
                Parallel.For(0, boostedLabelsPooning.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//training_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//training_dt[0].Count()
                        point[j] = double.Parse(training_dt[i][j].ToString());

                    double[] tmpLabel = askTreeMeanValAtLevel(point, Tree_orderedById, best_level);
                    for (int j = 0; j < tmpLabel.Count(); j++)
                        boostedLabelsPooning[i][j] = boostedLabelsPooning[i][j] - tmpLabel[j];
                });
            }
            else
            {
                for (int i = 0; i < boostedLabelsPooning.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//training_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//training_dt[0].Count()
                        point[j] = double.Parse(training_dt[i][j].ToString());

                    double[] tmpLabel = askTreeMeanValAtLevel(point, Tree_orderedById, best_level);
                    for (int j = 0; j < tmpLabel.Count(); j++)
                        boostedLabelsPooning[i][j] = boostedLabelsPooning[i][j] - tmpLabel[j];
                }
            }

            return boostedLabelsPooning;
        }
        
        public static void printErrorsOfTree(double[] errArr, string filename)
        {
            StreamWriter writer;
            if (Form1.UseS3)
            {
                string dir_name = Path.GetDirectoryName(filename);
                string file_name = Path.GetFileName(filename);

                S3DirectoryInfo s3dir = new S3DirectoryInfo(Form1.S3client, Form1.bucketName, dir_name);
                S3FileInfo artFile = s3dir.GetFile(file_name);
                writer = new StreamWriter(artFile.OpenWrite());
            }
            else
                writer = new StreamWriter(filename, false);

            for (int i = 0; i < errArr.Count(); i++)
                writer.WriteLine(errArr[i]);
            writer.Close();
        }

        public static void printErrorsOfTree(double[] errArr,double[] NwavesArr, string filename)
        {
            StreamWriter writer;
            if (Form1.UseS3)
            {
                string dir_name = Path.GetDirectoryName(filename);
                string file_name = Path.GetFileName(filename);

                S3DirectoryInfo s3dir = new S3DirectoryInfo(Form1.S3client, Form1.bucketName, dir_name);
                S3FileInfo artFile = s3dir.GetFile(file_name);
                writer = new StreamWriter(artFile.OpenWrite());
            }
            else
                writer = new StreamWriter(filename, false);

            for (int i = 0; i < errArr.Count(); i++)
                writer.WriteLine(NwavesArr[i] + " " + errArr[i]);
            writer.Close();
        }

        public static void printErrorsOfTree(double err, int Nwaves, string filename)
        {
            StreamWriter writer;
            if (Form1.UseS3)
            {
                string dir_name = Path.GetDirectoryName(filename);
                string file_name = Path.GetFileName(filename);

                S3DirectoryInfo s3dir = new S3DirectoryInfo(Form1.S3client, Form1.bucketName, dir_name);
                S3FileInfo artFile = s3dir.GetFile(file_name);
                writer = new StreamWriter(artFile.OpenWrite());
            }
            else
                writer = new StreamWriter(filename, false);

            writer.WriteLine(Nwaves + " " + err);
            writer.Close();
        }
    }
}

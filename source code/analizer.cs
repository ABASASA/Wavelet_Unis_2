using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace DataSetsSparsity
{
    class analizer
    {
        private string analysisFolderName;
        private List<List<double>> MainGrid;
        private DB db;
        public static double[][][] resultsByTree;
        public static double[] resultsByForest;
        public static double MidPoint;

        public analizer(string analysisFolderName, List<List<double>> MainGrid, DB db)
        {
            this.analysisFolderName = analysisFolderName;
            this.MainGrid = MainGrid;
            this.db = db;
        }

        public void analize(List<int> trainingArr, List<int> testingArr, List<int> validatingArr, int[][] boundingBox)
        {
           #region RF tree 

            int tmp_N_rows = Convert.ToInt32(trainingArr.Count * userConfig.bagginPercent);
            List<int>[] trainingArrRF_indecesList = new List<int>[userConfig.nTrees];
            List<GeoWave>[] RFdecTreeArr = new List<GeoWave>[userConfig.nTrees];
            List<double> globalNorms = new List<double>();
            double normthreshold = -1;
            #region search for m termand threshold
            if (userConfig.m_terms ==-1)
            {
                //create RF
                wf.Program.applyFor(0, userConfig.nTrees, i =>
                {
                    List<int> trainingArrRF;
                    trainingArrRF = Bagging(trainingArr, userConfig.bagginPercent, i);
                    trainingArrRF_indecesList[i] = trainingArrRF;
                    bool[] Dim2Take = getDim2Take( i);
                    decicionTree decTreeRF = new decicionTree(db, Dim2Take);
                    RFdecTreeArr[i] = decTreeRF.getdecicionTree(trainingArrRF, boundingBox, i);
                });

                //set norms
                globalNorms = new List<double>();
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                    for (int j = 0; j < RFdecTreeArr[i].Count; j++)
                        globalNorms.Add(RFdecTreeArr[i][j].norm);

                globalNorms = globalNorms.OrderByDescending(d => d).ToList();

                //SORT each tree BY ID - for next computation
                Parallel.For(0, RFdecTreeArr.Count(), i =>
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });


                //array size - new section 
                int Mterms = globalNorms.Count;
                int hopp = userConfig.hopping == -1 ? 2 * RFdecTreeArr.Count() : userConfig.hopping;
                int steps = Mterms / hopp;

                List<int> indeces = new List<int>();    
                if (userConfig.nonLinearHopping)
                {
                    steps = 0;
                    for (int i = 0; i < globalNorms.Count; i++)
                    {
                        int j = skipIndex(i, RFdecTreeArr.Count());
                        if (j >= globalNorms.Count)
                            break;
                        else
                            steps++;
                        indeces.Add(j-1);
                    }
                    indeces = indeces.Distinct().ToList();
                    steps = indeces.Count;
                }

                double[] errorRyByWaveletsOfOneTest = new double[steps];
                double[] errorRyByWaveletsOfOnevalid = new double[steps];
                double[] errorRyByWaveletsOfOne = new double[steps];
                double[] NwaveletsInWaveletByWaveletOfOne = new double[steps];

                wf.Program.applyFor(0, steps, i =>
                {
                    int j = userConfig.nonLinearHopping ? indeces[i] : i * hopp;
                    errorRyByWaveletsOfOne[i] = testDecisionForest(trainingArr, db.training_dt, db.training_label, RFdecTreeArr, globalNorms[j], userConfig.errTypeTest, false);
                    errorRyByWaveletsOfOnevalid[i] = testDecisionForest(validatingArr, db.validation_dt, db.validation_label, RFdecTreeArr, globalNorms[j], userConfig.errTypeTest, false);
                    errorRyByWaveletsOfOneTest[i] = testDecisionForest(testingArr, db.testing_dt, db.testing_label, RFdecTreeArr, globalNorms[j], userConfig.errTypeTest, false);
                    NwaveletsInWaveletByWaveletOfOne[i] = j + 1;
                });

                int minIndex = Array.IndexOf(errorRyByWaveletsOfOnevalid, errorRyByWaveletsOfOnevalid.Min());
                int Mterm = (int)NwaveletsInWaveletByWaveletOfOne[minIndex];
                normthreshold = globalNorms[Mterm - 1];

                //PRINT
                wf.Program.printList(errorRyByWaveletsOfOne.ToList(), analysisFolderName + "\\mTermErrorOnTraining.txt");
                wf.Program.printList(errorRyByWaveletsOfOnevalid.ToList(), analysisFolderName + "\\mTermErrorOnValidating.txt");
                wf.Program.printList(errorRyByWaveletsOfOneTest.ToList(), analysisFolderName + "\\mTermErrorOnTesting.txt");
                wf.Program.printList(NwaveletsInWaveletByWaveletOfOne.ToList(), analysisFolderName + "\\mTermNwavelets.txt");
                List<double> tmpLst = new List<double>();
                tmpLst.Add(normthreshold);
                wf.Program.printList(tmpLst, analysisFolderName + "\\threshold.txt");
                tmpLst.Clear();
                tmpLst.Add(Mterm);
                wf.Program.printList(tmpLst, analysisFolderName + "\\Mterm.txt");
                tmpLst.Clear();


                //SMOOTHNESS ANALYSIS
                List<double> NwaveletsInWaveletByWaveletOfOneMterm = NwaveletsInWaveletByWaveletOfOne.ToList().GetRange(0, minIndex + 1);
                List<double> errorRyByWaveletsOfOneMterm = errorRyByWaveletsOfOne.ToList().GetRange(0, minIndex + 1);
                double alpha = getSmothness(NwaveletsInWaveletByWaveletOfOneMterm, errorRyByWaveletsOfOneMterm);
                tmpLst = new List<double>();
                tmpLst.Add(alpha);
                wf.Program.printList(tmpLst, analysisFolderName + "\\alphaMterm.txt");
                tmpLst = new List<double>();
                tmpLst.Add(Convert.ToDouble(1 + minIndex * hopp));
                wf.Program.printList(tmpLst, analysisFolderName + "\\Mterms.txt");
            }
            #endregion


            //build forest without validation
            for (int i = 0; i < validatingArr.Count(); i++)
                    trainingArr.Add(validatingArr[i]);
            RFdecTreeArr = new List<GeoWave>[userConfig.nTrees];
            wf.Program.applyFor(0, userConfig.nTrees, i =>
            {
                List<int> trainingArrRF;
                trainingArrRF = Bagging(trainingArr, userConfig.bagginPercent, i);
                trainingArrRF_indecesList[i] = trainingArrRF;
                bool[] Dim2Take = getDim2Take(i);
                decicionTree decTreeRF = new decicionTree(db, Dim2Take);
                RFdecTreeArr[i] = decTreeRF.getdecicionTree(trainingArrRF, boundingBox, i);
            });

            if (userConfig.saveTrees)
            {
                if (!System.IO.Directory.Exists(analysisFolderName + "\\archive")) 
                    System.IO.Directory.CreateDirectory(analysisFolderName + "\\archive");
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                {
                    wf.Program.printWaveletsProperties(RFdecTreeArr[i], analysisFolderName + "\\archive\\waveletsPropertiesTree_" + i.ToString() + ".txt");
                }
            }

            //SET NORM THRESHOLD
            normthreshold = userConfig.fixThreshold == -1 ? normthreshold : userConfig.fixThreshold;

            double[] VI = new double[db.training_dt[0].Count()];//array of variables for importance
            globalNorms = new List<double>();
            for (int i = 0; i < RFdecTreeArr.Count(); i++)
                for (int j = 0; j < RFdecTreeArr[i].Count; j++)
                {
                    globalNorms.Add(RFdecTreeArr[i][j].norm);
                    if (RFdecTreeArr[i][j].dimIndex >= 0 && RFdecTreeArr[i][j].norm >= normthreshold)
                        VI[RFdecTreeArr[i][j].dimIndex] += RFdecTreeArr[i][j].norm;

                }                    
            globalNorms = globalNorms.OrderByDescending(d => d).ToList();

            wf.Program.printList(VI.ToList(), analysisFolderName + "\\VI.txt");

            //SET NORM THRESHOLD BY MTERMS IF GIVVEN
            if (userConfig.m_terms != -1 && userConfig.m_terms < globalNorms.Count)
                normthreshold = globalNorms[userConfig.m_terms];

            //PREPARE FOR TESTING
            if (userConfig.setClassification)
                setMidpoint(db.validation_label);

            //SORT each tree BY ID 
            Parallel.For(0, RFdecTreeArr.Count(), i =>
            {
                RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
            });

            //IF TEST FULL RF (regardles to wf) 
            if (userConfig.testRF)
            {
                double[] tmperrorRyByForest = new double[RFdecTreeArr.Count()];
                double[] tmperrorValidByForest = new double[RFdecTreeArr.Count()];
                double[] tmperrorTrainByForest = new double[RFdecTreeArr.Count()];
                tmperrorRyByForest = testDecisionTreeManyRFnew(testingArr, db.testing_dt, db.testing_label, RFdecTreeArr, 0.0, userConfig.errTypeTest);
                tmperrorValidByForest = testDecisionTreeManyRFnew(validatingArr, db.validation_dt, db.validation_label, RFdecTreeArr, 0.0, userConfig.errTypeTest);
                tmperrorTrainByForest = testDecisionTreeManyRFnew(trainingArr, db.training_dt, db.training_label, RFdecTreeArr, 0.0, userConfig.errTypeTest);
                List<double> tmpNwaveletsInRF = new List<double>();
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                    tmpNwaveletsInRF.Add(RFdecTreeArr[i].Count());

                wf.Program.printList(tmperrorRyByForest.ToList(), analysisFolderName + "\\TesterrorByForest.txt");
                wf.Program.printList(tmperrorValidByForest.ToList(), analysisFolderName + "\\ValiderrorByForest.txt");
                wf.Program.printList(tmperrorTrainByForest.ToList(), analysisFolderName + "\\TrainerrorByForest.txt");
                wf.Program.printList(tmpNwaveletsInRF, analysisFolderName + "\\NwaveletsInRF.txt");
            }

            //IF TEST RF WITH THRESHOLD ON WAVELETS NORM
            if (userConfig.testWf )
            {
                double[] tmperrorRyByForest = new double[RFdecTreeArr.Count()];
                tmperrorRyByForest = testDecisionTreeManyRFnew(testingArr, db.testing_dt, db.testing_label, RFdecTreeArr, normthreshold, userConfig.errTypeTest);
                double[] tmperrorValidRyByForest = new double[RFdecTreeArr.Count()];
                tmperrorValidRyByForest = testDecisionTreeManyRFnew(validatingArr, db.validation_dt, db.validation_label, RFdecTreeArr, normthreshold, userConfig.errTypeTest);
                double[] tmperrorTrainRyByForest = new double[RFdecTreeArr.Count()];
                tmperrorTrainRyByForest = testDecisionTreeManyRFnew(trainingArr, db.training_dt, db.training_label, RFdecTreeArr, normthreshold, userConfig.errTypeTest);

                List<double> tmpNwaveletsInRF = new List<double>();
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                    tmpNwaveletsInRF.Add(RFdecTreeArr[i].Count());

                wf.Program.printList(tmperrorRyByForest.ToList(), analysisFolderName + "\\TesterrorByForestWithThreshold" + normthreshold.ToString() + ".txt");
                wf.Program.printList(tmperrorValidRyByForest.ToList(), analysisFolderName + "\\ValiderrorByForestWithThreshold" + normthreshold.ToString() + ".txt");
                wf.Program.printList(tmperrorTrainRyByForest.ToList(), analysisFolderName + "\\TrainerrorByForestWithThreshold" + normthreshold.ToString() + ".txt");
                wf.Program.printList(tmpNwaveletsInRF, analysisFolderName + "\\NwaveletsInRFWithThreshold" + normthreshold.ToString() + ".txt");
            }
            #endregion

            //add VI
        }


        private int skipIndex(int i, int numTrees)
        {
            return (int)Math.Sqrt(Math.Pow(2, i));


            //if (i <= numTrees)
            //    return i;
            //else if (i < 10 * numTrees)
            //    i = i * numTrees;
            //else if (i < 20 * numTrees)
            //    i = i* 2* numTrees;
            //else if (i < 30 * numTrees)
            //    i = i * 3 * numTrees;
            //else if (i < 40 * numTrees)
            //    i = i * 4 * numTrees;
            //else if (i < 50 * numTrees)
            //    i = i * 5 * numTrees;
            //else if (i < 100 * numTrees)
            //    i = i * 10 * numTrees;
            //else if (i < 200 * numTrees)
            //    i = i * 20 * numTrees;
            //else
            //    i = i * 1000 * numTrees;//100000
            //return i;
        }

        //private int skipIndex(int i)
        //{
        //    if (i <= 100)
        //        return i;
        //    else if (i < 300)
        //        i += 25;
        //    else if (i < 1000)
        //        i += 50;
        //    else if (i < 5000)
        //        i += 100;
        //    else if (i < 10000)
        //        i += 200;
        //    else if (i < 100000)
        //        i += 500;
        //    else if (i < 200000)
        //        i += 50000;//1000
        //    else if (i < 300000)
        //        i += 100000;//50000
        //    else
        //        i += 1000000;//100000
        //    return i;
        //}

        private void testWaveletsOneByOne(List<GeoWave>[] RFdecTreeArr, List<int> testingArr, double[][] dt, double[][] labels, 
                                          ref List<double> NwaveletsInWaveletByWavelet, ref List<double> errorRyByWavelets)
        {
            double[] normsOfTrees = new double[RFdecTreeArr.Count()];//[N trees][testingArr]
            //resultsByTree = new double[RFdecTreeArr.Count()][][];//[N trees][testingArr]
            Parallel.For(0, RFdecTreeArr.Count(), i =>
            {
                normsOfTrees[i] = -1;// ITS SET ON THE FLY
            });            
            
            double tmpErr = 0;
            List<double[]> NormMultyArr = new List<double[]>();
            for (int i = 0; i < RFdecTreeArr.Count(); i++)
                for (int j = 0; j < RFdecTreeArr[i].Count; j++)
                {
                    double[] pair = new double[2];
                    pair[0] = RFdecTreeArr[i][j].norm;
                    pair[1] = i;
                    NormMultyArr.Add(pair);
                }

            NormMultyArr = NormMultyArr.OrderByDescending(t => t[0]).ToList();

            int indexeTreeChanged = -1;
            bool newTree = false;

            //SET GLOBAL PARAMETER MODEFIED_LABLES (TO IMPROV EPERFORMANCE)
            resultsByForest = new double[testingArr.Count()];//[N trees][testingArr]
            //resultsByTree = new double[RFdecTreeArr.Count()][][];//[N trees][testingArr]
            double [][] modefied_Lables = new double[testingArr.Count()][];
            for (int i = 0; i < testingArr.Count(); i++)
                modefied_Lables[i] = new double[db.validation_label[0].Count()];

            int NerrorRyByWavelets = NormMultyArr.Count();
            int N_treesInUse = 0;
            for (int i = 0; i < NerrorRyByWavelets; i++)
            {
                if (i > 100 && N_treesInUse == RFdecTreeArr.Count())
                {
                    if (i < 500)
                        i += 25;
                    else if (i < 1000)
                        i += 50;
                    else if (i < 5000)
                        i += 100;
                    else if (i < 10000)
                        i += 200;
                    else if (i < 100000)
                        i += 500;
                    else if (i < 200000)
                        i += 50000;//1000
                    else if (i < 300000)
                        i += 100000;//50000
                    else
                        i += 1000000;//100000

                    if (i >= NerrorRyByWavelets)
                        i = NerrorRyByWavelets-1;
                    preparenormsOfTrees(ref normsOfTrees, i, NormMultyArr);
                    tmpErr = calcResultByTree(testingArr, dt, labels, RFdecTreeArr, normsOfTrees);
                    errorRyByWavelets.Add(tmpErr);
                    NwaveletsInWaveletByWavelet.Add(i + 1);
                }
                else
                {
                    indexeTreeChanged = (int)NormMultyArr[i][1];
                    if (normsOfTrees[indexeTreeChanged] == -1)
                    {
                        newTree = true;
                        N_treesInUse++;
                    }
                    normsOfTrees[indexeTreeChanged] = NormMultyArr[i][0];
                    tmpErr = testerrorRyByWavelets(testingArr, dt, labels, RFdecTreeArr, normsOfTrees, ref N_treesInUse, ref indexeTreeChanged, newTree, userConfig.errTypeTest);
                    errorRyByWavelets.Add(tmpErr);
                    newTree = false;
                    NwaveletsInWaveletByWavelet.Add(i + 1);
                }
            }
        }

        private void setMidpoint(double[][] Data_Lables)
        {
            double minVal = 0;
            double maxVal = 1;
            for (int i = 0; i < Data_Lables.Count(); i++)
            {
                minVal = (Data_Lables[i][0] < minVal) ? Data_Lables[i][0] : minVal;
                maxVal = (Data_Lables[i][0] > maxVal) ? Data_Lables[i][0] : maxVal;
            }
            MidPoint = 0.5 * (minVal + maxVal);
        }

        private double[] testDecisionTreeManyRFnew(List<int> testingArr, double[][] Data_table, double[][] Data_Lables, List<GeoWave>[] RFdecTreeArr, double NormThreshold, string NormLPTypeInEstimation)
        {
            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[RFdecTreeArr.Count()][][];//num of trees, label index, label values (or value in most cases)
            for (int i = 0; i < RFdecTreeArr.Count(); i++)
            {
                estimatedLabels[i] = new double[testingArr.Count()][];
                for (int j = 0; j < testingArr.Count(); j++)
                    estimatedLabels[i][j] = new double[Data_Lables[0].Count()];
            }

            if (userConfig.useParallel)
            {
                Parallel.For(0, testingArr.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[Data_table[0].Count()];//
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < Data_table[0].Count(); j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[testingArr[i]][j].ToString());
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
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[Data_table[0].Count()];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < Data_table[0].Count(); j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[testingArr[i]][j].ToString());
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
            if (estimatedLabels[0][0].Count() > 1)//multi labling -> classification
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        var indexAtMaxEstimate = estimatedLabels[k][i].ToList().IndexOf(estimatedLabels[k][i].Max());
                        var indexAtMaxLabel = Data_Lables[testingArr[i]].ToList().IndexOf(Data_Lables[testingArr[i]].Max());
                        if (indexAtMaxEstimate!= indexAtMaxLabel)
                            error[k]++;
                    }
                    error[k] = (error[k] / (double)testingArr.Count());
                }

            }
            else if (NormLPTypeInEstimation == "2")
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                            error[k] += (estimatedLabels[k][i][j] - Data_Lables[testingArr[i]][j]) * (estimatedLabels[k][i][j] - Data_Lables[testingArr[i]][j]);
                    error[k] = Math.Sqrt(error[k] / Convert.ToDouble(testingArr.Count()));
                }
            }
            else if (NormLPTypeInEstimation == "1")
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                            error[k] += Math.Abs(estimatedLabels[k][i][j] - Data_Lables[testingArr[i]][j]);
                }
            }
            else if(NormLPTypeInEstimation == "0")
            {
                int do_nothing = 0;
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                        {
                            if((estimatedLabels[k][i][j] >= MidPoint) &&  (Data_Lables[testingArr[i]][j] >= MidPoint) ||
                                (estimatedLabels[k][i][j] < MidPoint) &&  (Data_Lables[testingArr[i]][j] < MidPoint))
                                do_nothing++;
                            else
                                error[k]++;
                        }
                    error[k] = (error[k]/ (double)testingArr.Count());
                }                
                
            }
            return error;

            //go over all points
            //for each point evaluated voted point
            //find largest index
            //if max index is not the same ++ error

            //devide by number of points

        }

        //FROM LESS IMPORTANT TO MORE IMPORTANT
        private List<int> reorderTreesByContribution(List<List<GeoWave>> RFdecTreeArr, string MarginType, List<int> trainingArr, double[][] trainingData, double[][] trainingLabel)
        {
            //TABLE OF ERRORS FOR EACH TREE IN PREDICTION (BASED ON TRAINING)
            double[,] errorTable = new double[RFdecTreeArr.Count, trainingArr.Count];//each column is a prediction of error of tree

            //GET ESTIMATED LABELS
            double[][][] estimatedLabels = getEstimatedLabeles(RFdecTreeArr, trainingArr, trainingData, trainingLabel);
            
            //FIND MIN MAX VALUES TO DETECT REGRESSION OR CLASSIFICATION
            double minVal = 0;
            double maxVal = 1;
            for (int i = 0; i < trainingLabel.Count(); i++)
            {
                minVal = (trainingLabel[i][0] < minVal) ? trainingLabel[i][0] : minVal;
                maxVal = (trainingLabel[i][0] > maxVal) ? trainingLabel[i][0] : maxVal;
            }
            double midPoint = 0.5 * (minVal + maxVal);
            bool regression = (maxVal == 1 && (minVal == -1 || minVal == 0)) ? false : true;

            //ERROR[I] IS THE MARGIN VECTOR of full tree and err for each tree is saved at errorTable

            //ERR[I] IS THE ERROR OF POINT I USING ALL TREES
            double[] error = getAvgErr(trainingArr, trainingLabel, RFdecTreeArr, regression, estimatedLabels, midPoint, ref errorTable);

            //PREPARE INDEX LIST OF REMOVED TREES
            List<int> indexOfRemoved = new List<int>();
            List<int> indexOfTrees = new List<int>();
            for (int i = 0; i < RFdecTreeArr.Count; i++)
                indexOfTrees.Add(i);

            //OPERATED ON ALL TREES UNTILL ONE TREE IS LEFT
            for (int i = 0; i < RFdecTreeArr.Count; i++ )
            {
                //THE INDEX IS FROM THE [0,INDEXOFTREES] RANGE
                int indexTreeWithLowestImpact = getindexTreeWithLowestImpact(indexOfTrees, trainingArr, errorTable, error, MarginType);

                //ADD
                indexOfRemoved.Add(indexOfTrees[indexTreeWithLowestImpact]);

                //UPDATE AVG ERROR
                for (int k = 0; k < error.Count(); k++)
                {
                    error[k] = ((indexOfTrees.Count() * error[k]) - errorTable[indexOfTrees[indexTreeWithLowestImpact], k]) / (indexOfTrees.Count() - 1);
                }

                //REMOVE LESS IMPORTANT TREE
                indexOfTrees.RemoveAt(indexTreeWithLowestImpact);
            }
            return indexOfRemoved;
        }

        private int getindexTreeWithLowestImpact(List<int> indexOfTrees, List<int> trainingArr, double[,] errorTable, double[] error, string MarginType)
        {
            int indexTreeWithLowestImpact = -1;
            double lowImpact = double.MaxValue;
            for (int j = 0; j < indexOfTrees.Count; j++)
            {
                double tmpErr = 0;
                if (MarginType == "AVG")
                {
                    for (int k = 0; k < trainingArr.Count(); k++)
                    {
                        tmpErr += Math.Abs(errorTable[indexOfTrees[j], k] - error[k]);
                    }                
                }
                else if (MarginType == "MIN")
                {
                    tmpErr = double.MaxValue;
                    for (int k = 0; k < trainingArr.Count(); k++)
                    {
                        if (error[k] < tmpErr) 
                            tmpErr = error[k];
                    }  
                }

                if (tmpErr < lowImpact)
                {
                    lowImpact = tmpErr;
                    indexTreeWithLowestImpact = j;
                }
            }
            return indexTreeWithLowestImpact;
        }

        private double[] getAvgErr(List<int> trainingArr, double[][] trainingLabel, List<List<GeoWave>> RFdecTreeArr, bool regression, double[][][] estimatedLabels, double midPoint, ref double[,] errorTable)
        {
            double[] error = new double[trainingArr.Count];//or margin
            for (int i = 0; i < trainingArr.Count(); i++)//go over ALL LABELS
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)//GO OVER ALL TREES
                {
                    double err = 0;
                    if (regression)
                    {
                        err = Math.Pow((estimatedLabels[k][i][0] - trainingLabel[trainingArr[i]][0]), 2);
                    }
                    else
                        err = ((estimatedLabels[k][i][0] > midPoint && trainingLabel[trainingArr[i]][0] > midPoint))
                            || (estimatedLabels[k][i][0] < midPoint && trainingLabel[trainingArr[i]][0] < midPoint) ? 1 : -1;

                    errorTable[k, i] = err;

                    error[i] += err;
                }
                error[i] /= RFdecTreeArr.Count();
            }
            return error;
        }

        private double[][][] getEstimatedLabeles(List<List<GeoWave>> RFdecTreeArr, List<int> trainingArr, double[][] trainingData, double[][] trainingLabel)
        {
            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[RFdecTreeArr.Count()][][];//index of tree, label index, label values (or value in most cases)
            for (int i = 0; i < RFdecTreeArr.Count(); i++)
            {
                estimatedLabels[i] = new double[trainingArr.Count()][];
                for (int j = 0; j < trainingArr.Count(); j++)
                    estimatedLabels[i][j] = new double[trainingLabel[0].Count()];
            }

            if (userConfig.useParallel)
            {
                Parallel.For(0, trainingArr.Count(), i =>
                {
                    double[] point = new double[trainingData[0].Count()];
                    for (int j = 0; j < trainingData[0].Count(); j++)
                        point[j] = double.Parse(trainingData[trainingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];
                    Parallel.For(0, RFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArr[j], 0);// 0 FOR TAKING WORST CASE
                    });

                    for (int j = 0; j < trainingLabel[0].Count(); j++)
                    {
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = tmpLabel[k][j];
                    }
                });
            }
            else
            {
                for (int i = 0; i < trainingArr.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[trainingData[0].Count()];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < trainingData[0].Count(); j++)//Data_table[0].Count()
                        point[j] = double.Parse(trainingData[trainingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];

                    for (int j = 0; j < RFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArr[j], 0);// 0 FOR TAKING WORST CASE
                    }

                    for (int j = 0; j < trainingLabel[0].Count(); j++)
                    {
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = tmpLabel[k][j];
                    }
                }
            }
            return estimatedLabels;
        }

        private double calcResultByTree(List<int> testingArr, double[][] Data_table, double[][] Data_Lables, 
                                             List<GeoWave>[] RFdecTreeArrById, double[] normsOfTrees)
        {
            double[] errArr = new double[testingArr.Count()];

            Parallel.For(0, testingArr.Count(), i =>
            {
                //GO OVER ALL TREES
                double tmpVal = 0;
                for (int j = 0; j < RFdecTreeArrById.Count(); j++)
                {
                    double[] val = askTreeMeanVal(Data_table[testingArr[i]], RFdecTreeArrById[j], normsOfTrees[j]);
                    tmpVal += val[0];
                }
                tmpVal /= RFdecTreeArrById.Count();

                if (userConfig.errTypeTest == "0")
                {
                    if ((((tmpVal >= MidPoint) && (Data_Lables[testingArr[i]][0] < MidPoint)) ||
                                ((tmpVal < MidPoint) && (Data_Lables[testingArr[i]][0] >= MidPoint))))
                        errArr[i]++;
                }
                else
                    errArr[i] = (tmpVal - Data_Lables[testingArr[i]][0]) * (tmpVal - Data_Lables[testingArr[i]][0]);
            });

            //for (int i = 0; i < testingArr.Count(); i++)
            //{

            //    //GO OVER ALL TREES
            //    double tmpVal = 0;
            //    for (int j = 0; j < RFdecTreeArrById.Count(); j++)
            //    {
            //        double[] val = askTreeMeanVal(Data_table[testingArr[i]], RFdecTreeArrById[j], normsOfTrees[j]);
            //        tmpVal += val[0];
            //    }
            //    tmpVal /= RFdecTreeArrById.Count();

            //    if (userConfig.errTypeTest == "0")
            //    {
            //        if ((((tmpVal >= MidPoint) && (Data_Lables[testingArr[i]][0] < MidPoint)) ||
            //                    ((tmpVal < MidPoint) && (Data_Lables[testingArr[i]][0] >= MidPoint))))
            //            errArr[i]++;
            //    }
            //    else
            //        errArr[i] = (tmpVal - Data_Lables[testingArr[i]][0]) * (tmpVal - Data_Lables[testingArr[i]][0]);
            //}

            double error = 0;
            for(int i=0; i < errArr.Count(); i++)
                error +=errArr[i];
            if(userConfig.errTypeTest == "0")
                error /=(double)testingArr.Count();
            else
                error = Math.Sqrt(error / Convert.ToDouble(testingArr.Count()));
            return error;               
        }

        private void preparenormsOfTrees(ref double[] normsOfTrees, int index2start, List<double[]> NormMultyArr)
        {
            bool[] wasSet = new bool[normsOfTrees.Count()];
            int ID = -1;
            int totalSet = 0;
            while (totalSet != normsOfTrees.Count())
            {
                ID = (int)NormMultyArr[index2start][1];
                if (wasSet[ID])
                {
                    index2start--;
                    continue;
                }

                normsOfTrees[ID] = NormMultyArr[index2start][0];
                wasSet[ID] = true;
                totalSet++;
                index2start--;
            }
        }

        private double testerrorRyByWavelets(List<int> testingArr, double[][] Data_table, double[][] Data_Lables, 
                                             List<GeoWave>[] RFdecTreeArrById, double[] normsOfTrees, 
                                             ref int N_treesInUse, ref int indexeTreeChanged, bool newTree, string NormLPTypeInEstimation)
        {
            //GO TO THE TREE THAT HAD BEEN CHANGED (ONE MORE WAVELET) AND RE-CALCULATE IT 
            int tmpIndexeTreeChanged = indexeTreeChanged;
            double[] oldVal = new double[Data_Lables[0].Count()];
            double[] newVal = new double[Data_Lables[0].Count()];
            //double weightOldGroup = Convert.ToDouble(N_treesInUse - 1) / (Convert.ToDouble(N_treesInUse));
            //double weightChangedTree = Convert.ToDouble(1) / (Convert.ToDouble(N_treesInUse) );

            double[][] modefied_Lables = new double[testingArr.Count()][];
            for (int i = 0; i < testingArr.Count(); i++)
                modefied_Lables[i] = new double[1];
            wf.Program.applyFor(0, testingArr.Count(), i =>
            {
                modefied_Lables[i] = askTreeMeanVal(Data_table[testingArr[i]], RFdecTreeArrById[tmpIndexeTreeChanged], normsOfTrees[tmpIndexeTreeChanged]);
                oldVal = resultsByTree[tmpIndexeTreeChanged][i];// - GET OLD VAL BEFORE CHANGE (COULD BE ZERO)
                resultsByTree[tmpIndexeTreeChanged][i] = modefied_Lables[i];
                newVal = resultsByTree[tmpIndexeTreeChanged][i];
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

            double error = 0;
            if (NormLPTypeInEstimation == "0")
            {
                double errArr = 0;
                for (int i = 0; i < testingArr.Count(); i++)
                    if ((((resultsByForest[i] >= MidPoint) && (Data_Lables[testingArr[i]][0] < MidPoint)) || ((resultsByForest[i] < MidPoint) && (Data_Lables[testingArr[i]][0] >= MidPoint))))
                        errArr++;
                error = errArr/ (double)testingArr.Count();
            }
            else
            {
                for (int i = 0; i < testingArr.Count(); i++)
                    error += (resultsByForest[i] - Data_Lables[testingArr[i]][0]) * (resultsByForest[i] - Data_Lables[testingArr[i]][0]);
                error = Math.Sqrt(error / Convert.ToDouble(testingArr.Count()));            
            }

            return error;
        }


        private bool[] getDim2Take(int Seed)
        {
            bool[] Dim2Take = new bool[db.training_dt[0].Count()];

            var ran = new Random(Seed);
            //List<int> dimArr = Enumerable.Range(0, rc.dim).OrderBy(x => ran.Next()).ToList().GetRange(0, rc.dim);
            //List<int> dimArr = Enumerable.Range(0, rc.dim).OrderBy(x => ran.Next()).ToList().GetRange(0, rc.dim);
            for (int i = 0; i < userConfig.nFeatures; i++)
            {
                int index = ran.Next(0, db.training_dt[0].Count());
                if (Dim2Take[index] == true)
                    i--;
                else
                    Dim2Take[index] = true;
            }

            return Dim2Take;
        }

        private List<int> Bagging(List<int> trainingArr, double percent, int Seed)//percent in [0,1]
        {
            //List<int> baggedArr = new List<int>();
            int N_rows = Convert.ToInt32(trainingArr.Count * percent);
            //int Seed = (int)DateTime.Now.Ticks;
            var ran = new Random(Seed);
//            return Enumerable.Range(0, trainingArr.Count).OrderBy(x => ran.Next()).ToList().GetRange(0, N_rows);
            return trainingArr.OrderBy(x => ran.Next()).ToList().GetRange(0, N_rows);
        }

        private List<int> BaggingBreiman(List<int> trainingArr, int Seed)//percent in [0,1]
        {
            bool[] isSet = new bool[trainingArr.Count];
            List<int> baggedArr = new List<int>();
            var ran = new Random(Seed);
            for (int i = 0; i < trainingArr.Count; i++)
            {
                int j = ran.Next(0, trainingArr.Count);
                if (isSet[j] == false)
                    baggedArr.Add(trainingArr[j]);
                isSet[j] = true;
            }          
            return baggedArr;
        }

        private double testDecisionTree(List<int> testingArr, double[][] Data_table, double[][] Data_Lables, List<GeoWave> Tree_orderedById, double NormThreshold, string NormLPTypeInEstimation, bool Sort)
        {
            if(Sort)
                Tree_orderedById = Tree_orderedById.OrderBy(o => o.ID).ToList();

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[testingArr.Count()][];
            for (int i = 0; i < testingArr.Count(); i++)
                estimatedLabels[i] = new double[Data_Lables[0].Count()];

            if (userConfig.useParallel)
            {
                Parallel.For(0, testingArr.Count(), i =>
                {
                    estimatedLabels[i] = askTreeMeanVal(Data_table[testingArr[i]], Tree_orderedById, NormThreshold);
                });
            }
            else
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    estimatedLabels[i] = askTreeMeanVal(Data_table[testingArr[i]], Tree_orderedById, NormThreshold);
                }
            }


            double error = 0;
            //if (estimatedLabels[0].Count() > 1 || )//multi labling -> classification
            //{
            //    for (int i = 0; i < testingArr.Count(); i++)
            //    {
            //        var indexAtMaxEstimate = estimatedLabels[i].ToList().IndexOf(estimatedLabels[i].Max());
            //        var indexAtMaxLabel = Data_Lables[testingArr[i]].ToList().IndexOf(Data_Lables[testingArr[i]].Max());
            //        if (indexAtMaxEstimate != indexAtMaxLabel)
            //            error++;
            //    }
            //    error = (error / (double)testingArr.Count());
            //}

            if (estimatedLabels[0].Count() > 1 || NormLPTypeInEstimation == "3")//multi labling 
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]) * (estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]);
                    }
                error = error / Convert.ToDouble(testingArr.Count());
            }
            else if (NormLPTypeInEstimation == "2")
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]) * (estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(testingArr.Count()));
            }
            else if (NormLPTypeInEstimation == "1")
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        error += Math.Abs(estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]);
                    }
            }
            else if (NormLPTypeInEstimation == "0")
            {
                int do_nothing = 0;
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                     {
                         if ((estimatedLabels[i][j] >= MidPoint) && (Data_Lables[testingArr[i]][j] >= MidPoint) ||
                                (estimatedLabels[i][j] < MidPoint) && (Data_Lables[testingArr[i]][j] < MidPoint))
                                do_nothing++;
                            else
                                error++;
                        }
                    error /= (double)testingArr.Count();

            }
            else if (NormLPTypeInEstimation == "-1")//max
            {
                List<double> errList = new List<double>();
                double tmp = 0;
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    tmp = 0;
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        tmp += Math.Abs(estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]);
                    }
                    errList.Add(tmp);
                }
                error = errList.Max();
            }
            else if (NormLPTypeInEstimation == "0" && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * Data_Lables[testingArr[i]][0]) <= 0)
                        error += 1;
                }
            }
            else if (NormLPTypeInEstimation == "-2" && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if (Data_Lables[testingArr[i]][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (Data_Lables[testingArr[i]][0] == -1)
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

        private double[] askTreeMeanVal(double[] point, List<GeoWave> Tree_orderedById, double NormThreshold)
        {

            int counter = 0;
            if (!DB.IsPntInsideBox(Tree_orderedById[0].boubdingBox, point, db.training_dt[0].Count()))
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
            bool endOfLoop = false;

            while (!endOfLoop)
            {
                if (Tree_orderedById[parent_index].hyperPlane == null)
                {
                    break;
                }
                bool tmpBool = DB.CheckOfEachChild(Tree_orderedById[parent_index].hyperPlane,
                        point, Tree_orderedById[parent_index].PositionMean); // tmpBool = true (child0)
                if (Tree_orderedById[parent_index].child0 != -1 && tmpBool)
                {
                    if (!Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue.SequenceEqual(zeroMean) &&
                        NormThreshold <= Tree_orderedById[Tree_orderedById[parent_index].child0].norm) //take the mean value if its not 0 and the wavelete should be taken ( norm size) - or if its the root wavelete
                    {
                        for (int t = 0; t < MeanValue.Count(); t++)
                            MeanValue[t] += (Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue[t] - Tree_orderedById[parent_index].MeanValue[t]);
                    }

                    parent_index = Tree_orderedById[parent_index].child0;
                }
                else if (Tree_orderedById[parent_index].child1 != -1 && !tmpBool)
                {
                    if (NormThreshold <= Tree_orderedById[Tree_orderedById[parent_index].child1].norm) //take the mean value if its above threshold
                    {
                        for (int t = 0; t < MeanValue.Count(); t++)
                            MeanValue[t] += (Tree_orderedById[Tree_orderedById[parent_index].child1].MeanValue[t] - Tree_orderedById[parent_index].MeanValue[t]);
                    }

                    parent_index = Tree_orderedById[parent_index].child1;
                }
                else
                    endOfLoop = true;
            }
            /*for(int indexLabel = 0; indexLabel < MeanValue.Count();indexLabel ++)
            {
                MeanValue[indexLabel] *= Tree_orderedById[0].m_MaxLabel[indexLabel] - Tree_orderedById[0].m_MinLabel[indexLabel];
                MeanValue[indexLabel] += Tree_orderedById[0].m_MinLabel[indexLabel];

            }*/
            return MeanValue;
        }

        private double testDecisionForest(List<int> testingArr, double[][] Data_table, double[][] Data_Lables, List<GeoWave>[] RFdecTreeArrById, double NormThreshold, string NormLPTypeInEstimation, bool Sort)
        {
            if (Sort)
            {
                //SORT each tree BY ID 
                Parallel.For(0, RFdecTreeArrById.Count(), i =>
                {
                    RFdecTreeArrById[i] = RFdecTreeArrById[i].OrderBy(o => o.ID).ToList();
                });
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[testingArr.Count()][];
            for (int i = 0; i < testingArr.Count(); i++)
                estimatedLabels[i] = new double[Data_Lables[0].Count()];

            //for each label
            for (int i = 0; i < testingArr.Count(); i++)
            {
                double[] tmpVal = new double[RFdecTreeArrById[0][0].MeanValue.Count()];
                //for each tree
                for (int j = 0; j < RFdecTreeArrById.Count(); j++)
                {
                    double[] tmptmpVal = askTreeMeanVal(Data_table[testingArr[i]], RFdecTreeArrById[j], NormThreshold);
                    tmpVal = addDevide(tmpVal, tmptmpVal, RFdecTreeArrById.Count());
                }
                tmpVal.CopyTo(estimatedLabels[i], 0);
                //for (int j = 0; j < RFdecTreeArrById[i][0].MeanValue.Count(); j++)
                //    estimatedLabels[i][j] = tmpVal[j];
            }


            double error = 0;
            if (estimatedLabels[0].Count() > 1 || NormLPTypeInEstimation == "3")//multi labling 
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]) * (estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]);
                    }
                error = error / Convert.ToDouble(testingArr.Count());
            }

            else if (NormLPTypeInEstimation == "2")
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]) * (estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(testingArr.Count()));
            }
            else if (NormLPTypeInEstimation == "1")
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        error += Math.Abs(estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]);
                    }
            }
            else if (NormLPTypeInEstimation == "0")
            {
                int do_nothing = 0;
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        if ((estimatedLabels[i][j] >= MidPoint) && (Data_Lables[testingArr[i]][j] >= MidPoint) ||
                               (estimatedLabels[i][j] < MidPoint) && (Data_Lables[testingArr[i]][j] < MidPoint))
                            do_nothing++;
                        else
                            error++;
                    }
                error /= (double)testingArr.Count();

            }
            else if (NormLPTypeInEstimation == "-1")//max
            {
                List<double> errList = new List<double>();
                double tmp = 0;
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    tmp = 0;
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        tmp += Math.Abs(estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]);
                    }
                    errList.Add(tmp);
                }
                error = errList.Max();
            }
            else if (NormLPTypeInEstimation == "0" && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * Data_Lables[testingArr[i]][0]) <= 0)
                        error += 1;
                }
            }
            else if (NormLPTypeInEstimation == "-2" && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if (Data_Lables[testingArr[i]][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (Data_Lables[testingArr[i]][0] == -1)
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

        private double[] addDevide(double[] tmpVal, double[] tmptmpVal, int deviding)
        {
            double[] result = new double[tmpVal.Count()];
            for (int j = 0; j < tmpVal.Count(); j++)
                result[j] = tmpVal[j] + (tmptmpVal[j] / Convert.ToDouble(deviding));

            return result;
        }

        private double getSmothness(List<double> nwaveletsInWaveletByWaveletOfOne, List<double> errorRyByWaveletsOfOne)
        {
            double[] x = new double[nwaveletsInWaveletByWaveletOfOne.Count];
            double[] y = new double[nwaveletsInWaveletByWaveletOfOne.Count];
            for (int i = 0; i < x.Count(); i++)
            {
                x[i] = Math.Log(nwaveletsInWaveletByWaveletOfOne[i]);
                y[i] = Math.Log(errorRyByWaveletsOfOne[i]);

            }
            double rsquared;
            double yintercept;
            double slope;
            wf.Program.LinearRegression(x, y, 0, x.Count(), out rsquared, out yintercept, out slope);
            return (-1) * slope;
        }
    }
}

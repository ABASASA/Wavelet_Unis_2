using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Forms;
//using System.IO;

using Accord.Math;
//using Amazon.S3;
//using Amazon.S3.IO;
// ReSharper disable PossibleNullReferenceException



namespace DataScienceAnalysis
{
    public partial class Form1 : Form
    {
        //new params
        public static double[][] boundingBox;
        public static List<List<double>> MainGrid;
        public static string MainFolderName; //THE DIR OF THE ROOT FOLDER
        public static string[] seperator = { " ", ";", "/t", "/n", "," };
        static public bool rumPrallel;
        static public bool UseS3;
        static public string bucketName;
        //static public AmazonS3Client S3client = new AmazonS3Client();
        public static bool runBoosting;
        public static bool runRf;
        public static bool runProoning;
        public static bool runBoostingProoning;
        public static bool runBoostingLearningRate;
        public static bool runRFProoning;
        public static double upper_label;
        public static double lower_label;

        public static userConfig u_config = new userConfig();

        public static Dictionary<Tuple<int, int>, bool> trainNaTable = new Dictionary<Tuple<int, int>, bool>();
        public static Dictionary<Tuple<int, int>, bool> testNaTable = new Dictionary<Tuple<int, int>, bool>();
        public static Dictionary<Tuple<int, int>, bool> validNaTable = new Dictionary<Tuple<int, int>, bool>();

        //global distances for diffusion map
        public double[,] globalDistances;
   
        public Form1()
        {
            InitializeComponent();
            setTypeLearningList();
            //READ AND SET PROPERTIES DEBUG 2m0rr0w2
            //2m0rr0w2
            u_config.readConfig(@"C:\Wavelets decomposition\config.txt");
             setfromConfig();
        }
        private void  setTypeLearningList()
        {
            comboLearningType.Items.Add("regression");
            comboLearningType.Items.Add("binary classification");
            comboLearningType.Items.Add("L1");
            comboLearningType.Items.Add("many classification");
            comboLearningType.SelectedIndex = 0;
        }
        static string evaluateString(string expression, int k)
        {
            string str = expression.Replace("k", k.ToString());
            DataTable loDataTable = new DataTable();
            DataColumn loDataColumn = new DataColumn("Eval", typeof(double), str);
            loDataTable.Columns.Add(loDataColumn);
            loDataTable.Rows.Add(0);
            //return (double)(loDataTable.Rows[0]["Eval"]);
            return loDataTable.Rows[0]["Eval"].ToString();
        }

     
     
        public static void calcTreesVariance(List<int>[] table, double[][] label, int labelindex, string filename)
        {
            List<double> average = new List<double>();
            List<double> averaged_variance = new List<double>();
            
            //calc avg
            for (int i = 0; i < table.Count(); i++)
            {
                double tmpAvg = 0;
                for (int j = 0; j < table[i].Count(); j++)
                {
                    tmpAvg += label[table[i][j]][labelindex];
                }
                average.Add(tmpAvg / table[i].Count());
            }

            //calc var
            for (int i = 0; i < table.Count(); i++)
            {
                double tmpVar = 0;
                for (int j = 0; j < table[i].Count(); j++)
                {
                    tmpVar += (label[table[i][j]][labelindex] - average[i]) * (label[table[i][j]][labelindex] - average[i]);
                }
                averaged_variance.Add(tmpVar / table[i].Count());
            }

            PrintEngine.printList(averaged_variance, filename);
        }
        


        public static bool isBoxSingular(double[][] box)
        {
            for (int i = 0; i < box[0].Count(); i++)
            {
                if (box[1][i] <= box[0][i])
                    return true;
            }
            return false;
        }

        public static bool isBoxSingular(int[][] box, int dim)
        {
            for (int i = 0; i < dim; i++)
            {
                if (box[1][i] < box[0][i])
                    return true;
            }

            return box[0].SequenceEqual(box[1]);
        }

        private void executeManyConfigsByClick(object sender, EventArgs e)
        {
           
            const string pathToConfigFolderAmazon = "C:\\Users\\Administrator\\Desktop\\AlexDropbox\\Dropbox\\Alex-Oren\\CF\\";
           // const string pathToConfigFolderLocal = "C:\\Users\\Alex\\Dropbox\\Alex-Oren\\CFtest\\";
            const string pathToConfigFolderLocal = "C:\\Users\\Alex\\Dropbox\\CFtest\\";
            string executePath = (Directory.Exists(pathToConfigFolderAmazon))
                ? pathToConfigFolderAmazon
                : pathToConfigFolderLocal;
            //disable default config read! 2m0rr0w2
           // u_config.readConfig(executePath + "configDefault.dat");
            Refresh();
            foreach (string file in  Directory.EnumerateFiles(executePath, "*.txt"))
            {
                StreamWriter statusWriter = new StreamWriter(executePath + "status.dat", true);
                u_config.readConfig(file);
                setfromConfig();
                statusWriter.WriteLine("dataset: " + file + " started execution....");
                label4.Text = file;
                btnScript_Click(sender, e);
  
                statusWriter.WriteLine("dataset "+file+" ready!!!!");
                statusWriter.Close();
               
            }
           
           
            btnScript.BackColor = Color.Green;
        }

  
        private void btnScript_Click(object sender, EventArgs e)
        {      

            set2Config();
            Refresh();
            u_config.printConfig(@"C:\Wavelets decomposition\config.txt");
           // AmazonS3Client client = Helpers.configAmazonS3ClientS3Client();

            UseS3 = UseS3CB.Checked;
            rumPrallel = rumPrallelCB.Checked;
            runBoosting = runBoostingCB.Checked;
            runProoning = runProoningCB.Checked;
            runBoostingProoning = runBoostingProoningCB.Checked;
            runRFProoning = runRFProoningCB.Checked;
            runRf = runRfCB.Checked;
            runBoostingLearningRate = runBoostingLearningRateCB.Checked;

            bucketName = bucketTB.Text;
            string results_path = @ResultsTB.Text;
            string db_path = @DBTB.Text+ "\\";//@"C:\Users\Administrator\Dropbox\ADA\ada_valid\"; //"D:\\Phd\\Shai\\code\\tests\\helix tests\\noise_5\\noise_5\\"; // "C:\\reasearch\\tests\\lena\\";


            //get dir
            MainFolderName = results_path;
            //Helpers.createMainDirectoryOrResultPath(results_path, bucketName, client);
            Helpers.createMainDirectoryOrResultPath(results_path, bucketName);
            //READ DATA
            DB db = new DB();
            db.training_dt = db.getDataTable(db_path + "trainingData.txt");
            db.testing_dt = db.getDataTable(db_path + "testingData.txt");
            db.validation_dt = db.getDataTable(db_path + "ValidData.txt");

            db.training_label = db.getDataTable(db_path + "trainingLabel.txt");
            db.testing_label = db.getDataTable(db_path + "testingLabel.txt");
            db.validation_label = db.getDataTable(db_path + "ValidLabel.txt");

            upper_label = db.training_label.Max();
            lower_label = db.training_label.Min();
            

                double trainingPercent = double.Parse(trainingPercentTB.Text);  // 0.02;

                long rowToRemoveFrom = Convert.ToInt64(db.training_dt.Count() * trainingPercent);
                db.training_dt = db.training_dt.Where((el, i) => i < rowToRemoveFrom).ToArray();
                db.training_label = db.training_label.Where((el, i) => i < rowToRemoveFrom).ToArray();
                db.testing_dt = db.testing_dt.Where((el, i) => i < rowToRemoveFrom).ToArray();
                db.testing_label = db.testing_label.Where((el, i) => i < rowToRemoveFrom).ToArray();
                db.validation_dt = db.training_dt.Where((el, i) => i < rowToRemoveFrom).ToArray();
                db.validation_label = db.validation_label.Where((el, i) => i < rowToRemoveFrom).ToArray();
             

                //REDUCE DIM, GLOBAL PCA
                if (usePCA.Checked)
                {
                    DimReduction dimreduction = new DimReduction(db.training_dt);
                    db.PCAtraining_dt = dimreduction.getGlobalPca(db.training_dt);
                    db.PCAtesting_dt = dimreduction.getGlobalPca(db.testing_dt);
                    db.PCAvalidation_dt = dimreduction.getGlobalPca(db.validation_dt);          
                }
                else
                {
                    //de-activate pca for dbg
                    db.PCAtraining_dt = db.training_dt;
                    db.PCAtesting_dt = db.testing_dt;
                    db.PCAvalidation_dt = db.validation_dt;
                }

                db.PCAtraining_GridIndex_dt = new long[db.PCAtraining_dt.Count()][];
                for (int i = 0; i < db.PCAtraining_dt.Count(); i++)
                    db.PCAtraining_GridIndex_dt[i] = new long[db.PCAtraining_dt[i].Count()];

                //BOUNDING BOX AND MAIN GRID              
                boundingBox = db.getboundingBox(db.PCAtraining_dt);
                MainGrid = db.getMainGrid(db.PCAtraining_dt, boundingBox, ref db.PCAtraining_GridIndex_dt);
            

                //READ CONFIG
                methodConfig mc = new methodConfig(true);
                int Nloops = int.Parse(NloopsTB.Text) - 1;
                int Kfolds = 0;
                if (int.TryParse(croosValidTB.Text, out Kfolds))
                    Nloops = Kfolds - 1;

                for (int k = 0; k < Nloops; k++)
                    mc.boostlamda_0.Add(3.8);// - create variant in number of pixels
               

                mc.generateRecordConfigArr();



                for (int k = 0; k < mc.recArr.Count(); k++)
                {
                    //manual set indeces of categorical variable
                    //mc.recArr[k].indOfCategorical.Add(3);

                    mc.recArr[k].learningType = (recordConfig.LearnigType) comboLearningType.SelectedIndex; // regression, binary class, multy class
                    if (mc.recArr[k].learningType == recordConfig.LearnigType.BinaryClassification)
                    {
                        mc.recArr[k].binaryMinClass = lower_label;
                        mc.recArr[k].binaryMaxClass = upper_label;
                        mc.recArr[k].midClassValue = 0.5*(lower_label + upper_label);
                    }
                    mc.recArr[k].dim = NfeaturesTB.Text == @"all" ? db.PCAtraining_dt[0].Count() : int.Parse(evaluateString(NfeaturesTB.Text, k));
                    mc.recArr[k].approxThresh = double.Parse(evaluateString(approxThreshTB.Text, k));// 0.1;
                    mc.recArr[k].partitionErrType = int.Parse(evaluateString(partitionTypeTB.Text, k)); //2;
                    mc.recArr[k].minWaveSize = int.Parse(evaluateString(minNodeSizeTB.Text, k)); //1;//CHANGE AFTER DBG
                    mc.recArr[k].hopping_size = int.Parse(evaluateString(waveletsSkipEstimationTB.Text, k)); //25;// 10 + 5 * (k + 1);// +5 * (k % 10);// 1;//25;
                    mc.recArr[k].test_error_size = double.Parse(evaluateString(waveletsPercentEstimationTB.Text, k));// +0.05 * (k % 10);// 1;// 0.1;//percent of waves to check
                    mc.recArr[k].NskipsinKfunc = double.Parse(evaluateString(boostingKfuncPercentTB.Text, k));// 0.0025;
                    mc.recArr[k].rfBaggingPercent = double.Parse(evaluateString(bagginPercentTB.Text, k)); // 0.6;
                    mc.recArr[k].rfNum = int.Parse(evaluateString(NrfTB.Text, k));// k + 1;//10 + k*10;// 100 / (k + 46) * 2;// int.Parse(Math.Pow(10, k + 1).ToString());
                    mc.recArr[k].boostNum = int.Parse(evaluateString(NboostTB.Text, k)); // 10;
                    mc.recArr[k].boostProoning_0 = int.Parse(evaluateString(NfirstPruninginBoostingTB.Text, k)); //13
                    mc.recArr[k].boostlamda_0 = double.Parse(evaluateString(boostingLamda0TB.Text, k));// 0.01 - (k + 1) * 0.001; //0.05;// 0.0801 + k * 0.001;// Math.Pow(0.1, k);// 0.22 + k*0.005;
                    mc.recArr[k].NwaveletsBoosting = int.Parse(evaluateString(NfirstwaveletsBoostingTB.Text, k)); //  4;// k + 1;
                    //mc.recArr[k].learningRate = 0;// 0.01;
                    mc.recArr[k].boostNumLearningRate = int.Parse(evaluateString(NboostingLearningRateTB.Text, k));// 55;// 18;
                    mc.recArr[k].percent_training_db = trainingPercent;
                    mc.recArr[k].BoundLevel = int.Parse(evaluateString(boundLevelTB.Text, k));//1024;
                    mc.recArr[k].NDimsinRF = NfeaturesrfTB.Text == @"all" ? db.PCAtraining_dt[0].Count() : int.Parse(evaluateString(NfeaturesrfTB.Text, k));
                    mc.recArr[k].split_type = int.Parse(evaluateString(splitTypeTB.Text, k)); //0
                    mc.recArr[k].NormLPType = int.Parse(evaluateString(errTypeEstimationTB.Text, k));
                    mc.recArr[k].RFpruningTestRange[1] = int.Parse(evaluateString(RFpruningEstimationRange1TB.Text, k)); // 12;// k + 9;
                    mc.recArr[k].boundDepthTree = int.Parse(evaluateString(boundDepthTB.Text, k));//1024;
                    mc.recArr[k].CrossValidFold = k;
                    // 2m0rr0w2 save labels dim in confif
                    mc.recArr[k].labelDim = db.training_label[0].Count();
                    //mc.recArr[k].boostNum =  t ;// tmp to delete !!!!!!!

                    //mc.recArr[k].RFwaveletsTestRange[0] = 25;
                    //mc.recArr[k].RFwaveletsTestRange[1] = 50;
                }
                // Helpers.createOutputDirectories(mc.recArr, client, u_config, bucketName, results_path);
                 Helpers.createOutputDirectories(mc.recArr, u_config, bucketName, results_path);
                //SET ID ARRAY LIST
                List<int> trainingID = Enumerable.Range(0, db.PCAtraining_dt.Count()).ToList();
                List<int> testingID = Enumerable.Range(0, db.PCAtesting_dt.Count()).ToList();

                //cross validation
                List<List<int>> trainingFoldId = new List<List<int>>();
                List<List<int>> testingFoldId = new List<List<int>>();

                Random ran = new Random(2);
                List<int> training_rand = trainingID.OrderBy(x => ran.Next()).ToList().GetRange(0, trainingID.Count);

                //THE LARGEST GROUP IS TRAINING
                if (int.TryParse(croosValidTB.Text, out Kfolds))
                    createCrossValid(Kfolds, training_rand, trainingFoldId, testingFoldId);

                //bounding intervals
                int[][] BB = new int[2][];
                BB[0] = new int[boundingBox[0].Count()];
                BB[1] = new int[boundingBox[0].Count()];
                for (int i = 0; i < boundingBox[0].Count(); i++)
                {
                    BB[1][i] = MainGrid[i].Count() - 1;//set last index in each dim
                }
                
               //loop over folds DEBUG i=2
                for (int i = 3; i < mc.recArr.Count; i++)
                {
                    recordConfig rc = mc.recArr[i];
                    Analizer analizer = new Analizer(MainFolderName + "\\" + rc.getShortName(), MainGrid, db, rc);
                     //exclude variables one by one
                    double[] numFeachuresVSerror = new double[rc.dim];
                    int indFeachureToExclude = -1;
                    double predResult = 0;
                    //loop over feachures
                    for (int j = rc.dim-1; j >= 0; j--)
                    {
                        //DEBUG, DISABLE FEACHURE EXCLUDE
                       // j = 0;

                        if (!croosValidCB.Checked)
                        {
                            analizer.analize(trainingID, testingID, BB, ref indFeachureToExclude, ref predResult);
                        }
                        else
                        {
                            analizer.analize(trainingFoldId[i], testingFoldId[i], BB, ref indFeachureToExclude, ref predResult);//cross validation
                        }
                        numFeachuresVSerror[j] = predResult; // index j, used j+1 featchures
                        analizer.excludeFeatureFromDb(indFeachureToExclude);
                    }
                    PrintEngine.printBestErrorByNumberOfFeatchures(analizer._analysisFolderName, numFeachuresVSerror);
                  
                }

            btnScript.BackColor = Color.Green;
        }

        
        //THE LARGEST GROUP IS TRAINING
        private static void createCrossValid(int kfolds, List<int> trainingId, List<List<int>> trainingFoldId, List<List<int>> testingFoldId)
        {
            //ADD LISTS
            for (int i = 0; i < kfolds; i++)
            {
                trainingFoldId.Add(new List<int>());
                testingFoldId.Add(new List<int>());
            }

            int Npoints = trainingId.Count / kfolds;
            //ADD POINTS ID
            int upper_bound = Npoints;
            //int lower_bound = -1;
            int counter =-1;
            for (int i = 0; i < trainingId.Count; i++)
            {
                if (i % Npoints == 0)
                {
                    counter++;//should happen Kfolds times
                    //if (i == (Kfolds * Npoints))
                    //    counter--;
                }
                
                for (int j = 0; j < kfolds; j++)
                {
                    if(j==counter)
                        testingFoldId[j].Add(trainingId[i]);
                    else
                        trainingFoldId[j].Add(trainingId[i]);
                }
            }            
        }

        private void setfromConfig()
        {
            if(u_config.croosValidCB == "1")
             croosValidCB.Checked =true;
            if (u_config.usePCA == "1")
                usePCA.Checked = true;
            if (u_config.runRFProoningCB == "1")
                runRFProoningCB.Checked = true;
            if (u_config.runBoostingLearningRateCB == "1")
                runBoostingLearningRateCB.Checked = true;
            if (u_config.runBoostingProoningCB == "1")
                runBoostingProoningCB.Checked = true;
            if (u_config.runProoningCB == "1")
                runProoningCB.Checked = true;
            if (u_config.runRfCB == "1")
                runRfCB.Checked = true;
            if (u_config.runBoostingCB == "1")
                runBoostingCB.Checked = true;
            if (u_config.rumPrallelCB == "1")
                rumPrallelCB.Checked = true;
            if (u_config.UseS3CB == "1")
                UseS3CB.Checked = true;
            if (u_config.saveTressCB == "1")
                saveTressCB.Checked = true;
            if (u_config.estimateOneTreeCb == "1")
                runOneTreeCB.Checked = true;
            if (u_config.estimateRFonTrainingCB == "1")
                estimateRFonTrainingCB.Checked = true;
            if (u_config.runOneTreeOnTtrainingCB == "1")
                runOneTreeOnTtrainingCB.Checked = true;
            if (u_config.estimateRFnoVotingCB == "1")
                estimateRFnoVotingCB.Checked = true;
            if (u_config.estimateRFwaveletsCB == "1")
                estimateRFwaveletsCB.Checked = true;
            if (u_config.BaggingWithRepCB == "1")
                BaggingWithRepCB.Checked = true;
            if (u_config.sparseRfCB == "1")
                sparseRfCB.Checked = true;



            croosValidTB.Text = u_config.croosValidTB;
            bucketTB.Text = u_config.bucketTB;
            DBTB.Text = u_config.DBTB;
            ResultsTB.Text = u_config.ResultsTB;
            NboostTB.Text = u_config.NboostTB;
            boostingLamda0TB.Text = u_config.boostingLamda0TB;
            NfirstwaveletsBoostingTB.Text = u_config.NfirstwaveletsBoostingTB;
            NfirstPruninginBoostingTB.Text = u_config.NfirstPruninginBoostingTB;
            NboostingLearningRateTB.Text = u_config.NboostingLearningRateTB;
            boostingKfuncPercentTB.Text = u_config.boostingKfuncPercentTB;
            NfeaturesTB.Text = u_config.NfeaturesTB;
            approxThreshTB.Text = u_config.approxThreshTB;
            minNodeSizeTB.Text = u_config.minNodeSizeTB;
            partitionTypeTB.Text = u_config.partitionTypeTB;
            splitTypeTB.Text = u_config.splitTypeTB;
            boundLevelTB.Text = u_config.boundLevelTB;
            pruningEstimationRange0TB.Text = u_config.pruningEstimationRange0TB;
            waveletsEstimationRange0TB.Text = u_config.waveletsEstimationRange0TB;
            errTypeEstimationTB.Text = u_config.errTypeEstimationTB;
            waveletsSkipEstimationTB.Text = u_config.waveletsSkipEstimationTB;
            waveletsPercentEstimationTB.Text = u_config.waveletsPercentEstimationTB;
            trainingPercentTB.Text = u_config.trainingPercentTB;
            NloopsTB.Text = u_config.NloopsTB;
            pruningEstimationRange1TB.Text = u_config.pruningEstimationRange1TB;
            waveletsEstimationRange1TB.Text = u_config.waveletsEstimationRange1TB;
            RFpruningEstimationRange1TB.Text = u_config.RFpruningEstimationRange1TB;
            NrfTB.Text = u_config.NrfTB;
            RFwaveletsEstimationRange1TB.Text = u_config.RFwaveletsEstimationRange1TB;
            RFpruningEstimationRange0TB.Text = u_config.RFpruningEstimationRange0TB;
            NfeaturesrfTB.Text = u_config.NfeaturesrfTB;
            RFwaveletsEstimationRange0TB.Text = u_config.RFwaveletsEstimationRange0TB;
            bagginPercentTB.Text = u_config.bagginPercentTB;
            sparseRfTB.Text = u_config.sparseRfTB;
            boundDepthTB.Text = u_config.boundDepthTB;
        }

        private void set2Config()
        {
            u_config.croosValidCB = croosValidCB.Checked ? "1" : "0";
            u_config.croosValidTB = croosValidTB.Text;
            u_config.usePCA = usePCA.Checked ? "1" : "0";
            u_config.runRFProoningCB = runRFProoningCB.Checked ? "1" : "0";
            u_config.runBoostingLearningRateCB = runBoostingLearningRateCB.Checked ? "1" : "0";
            u_config.runBoostingProoningCB = runBoostingProoningCB.Checked ? "1" : "0";
            u_config.runProoningCB = runProoningCB.Checked ? "1" : "0";
            u_config.runRfCB = runRfCB.Checked ? "1" : "0";
            u_config.runBoostingCB = runBoostingCB.Checked ? "1" : "0";
            u_config.rumPrallelCB = rumPrallelCB.Checked ? "1" : "0";
            u_config.UseS3CB = UseS3CB.Checked ? "1" : "0";
            u_config.estimateOneTreeCb = runOneTreeCB.Checked ? "1" : "0";
            u_config.estimateRFonTrainingCB = estimateRFonTrainingCB.Checked ? "1" : "0";         
            u_config.bucketTB = bucketTB.Text;
            u_config.DBTB = DBTB.Text;
            u_config.ResultsTB = ResultsTB.Text;
            u_config.NboostTB = NboostTB.Text;
            u_config.boostingLamda0TB = boostingLamda0TB.Text;
            u_config.NfirstwaveletsBoostingTB = NfirstwaveletsBoostingTB.Text;
            u_config.NfirstPruninginBoostingTB = NfirstPruninginBoostingTB.Text;
            u_config.NboostingLearningRateTB = NboostingLearningRateTB.Text;
            u_config.boostingKfuncPercentTB = boostingKfuncPercentTB.Text;
            u_config.NfeaturesTB = NfeaturesTB.Text;
            u_config.approxThreshTB = approxThreshTB.Text;
            u_config.minNodeSizeTB = minNodeSizeTB.Text;
            u_config.partitionTypeTB = partitionTypeTB.Text;
            u_config.splitTypeTB = splitTypeTB.Text;
            u_config.boundLevelTB = boundLevelTB.Text;
            u_config.pruningEstimationRange0TB = pruningEstimationRange0TB.Text;
            u_config.waveletsEstimationRange0TB = waveletsEstimationRange0TB.Text;
            u_config.errTypeEstimationTB = errTypeEstimationTB.Text;
            u_config.waveletsSkipEstimationTB = waveletsSkipEstimationTB.Text;
            u_config.waveletsPercentEstimationTB = waveletsPercentEstimationTB.Text;
            u_config.trainingPercentTB = trainingPercentTB.Text;
            u_config.NloopsTB = NloopsTB.Text;
            u_config.pruningEstimationRange1TB = pruningEstimationRange1TB.Text;
            u_config.waveletsEstimationRange1TB = waveletsEstimationRange1TB.Text;
            u_config.RFpruningEstimationRange1TB = RFpruningEstimationRange1TB.Text;
            u_config.NrfTB = NrfTB.Text;
            u_config.RFwaveletsEstimationRange1TB = RFwaveletsEstimationRange1TB.Text;
            u_config.RFpruningEstimationRange0TB = RFpruningEstimationRange0TB.Text;
            u_config.NfeaturesrfTB = NfeaturesrfTB.Text;
            u_config.RFwaveletsEstimationRange0TB = RFwaveletsEstimationRange0TB.Text;
            u_config.bagginPercentTB = bagginPercentTB.Text;
            u_config.sparseRfTB = sparseRfTB.Text;
            u_config.boundDepthTB = boundDepthTB.Text;
            
            u_config.saveTressCB = saveTressCB.Checked ? "1" : "0";
            u_config.runOneTreeOnTtrainingCB = runOneTreeOnTtrainingCB.Checked ? "1" : "0";
            u_config.estimateRFnoVotingCB = estimateRFnoVotingCB.Checked ? "1" : "0";
            u_config.estimateRFwaveletsCB = estimateRFwaveletsCB.Checked ? "1" : "0";
            u_config.BaggingWithRepCB = BaggingWithRepCB.Checked ? "1" : "0";
            u_config.sparseRfCB = sparseRfCB.Checked ? "1" : "0";
        }

        private void comboLearningType_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

   

    }
}
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.IO;
//using AForge.MachineLearning;
using Accord.MachineLearning;
using Accord.Math;
using Accord.Math.Decompositions;
using Accord.Statistics;
using Accord.Statistics.Analysis;
//using Imsl.Chart2D;
using System.Windows.Forms.DataVisualization.Charting;
using System.Threading.Tasks;


using System.Threading;
using Amazon;
using Amazon.Runtime;
using Amazon.S3;
using Amazon.S3.Model;
using Amazon.S3.IO;

namespace DataSetsSparsity
{
    public partial class Form1 : Form
    {
        //CONSTRUCTOR
        public Form1()
        {
            InitializeComponent();

            //READ AND SET PROPERTIES
            u_config.readConfig(@"C:\Wavelets decomposition\config.txt");
            setfromConfig();
        }

        //new params
        public static double[][] boundingBox;
        public static List<List<double>> MainGrid;
        public static string MainFolderName; //THE DIR OF THE ROOT FOLDER
        public static string[] seperator = { " ", ";", "/t", "/n", ","};
        static public bool rumPrallel;
        static public bool UseS3;
        static public string bucketName;
        static public AmazonS3Client S3client = new AmazonS3Client();
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

        static string EvaluateString(string expression, int k)
        {
            string str = expression.Replace("k", k.ToString());
            var loDataTable = new DataTable();
            var loDataColumn = new DataColumn("Eval", typeof(double), str);
            loDataTable.Columns.Add(loDataColumn);
            loDataTable.Rows.Add(0);
            //return (double)(loDataTable.Rows[0]["Eval"]);
            return loDataTable.Rows[0]["Eval"].ToString();
        }

        public static void printtable(double[][] table, string filename)
        {
            StreamWriter sw;
            if (Form1.UseS3)
            {
                string dir_name = Path.GetDirectoryName(filename);
                string file_name = Path.GetFileName(filename);

                S3DirectoryInfo s3dir = new S3DirectoryInfo(Form1.S3client, Form1.bucketName, dir_name);
                S3FileInfo artFile = s3dir.GetFile(file_name);
                sw = new StreamWriter(artFile.OpenWrite());
            }
            else
                sw = new StreamWriter(filename, false);

            string line = "";

            for (int i = 0; i < table.Count(); i++)
            {
                line = "";
                if (table[i] == null)
                {
                    continue;
                }
                for (int j = 0; j < table[i].Count(); j++)
                {
                    line += table[i][j].ToString() + " ";
                }
                sw.WriteLine(line);
            }

            sw.Close();
        }

        public static void printtable(List<int>[] table, string filename)
        {
            StreamWriter sw;
            if (Form1.UseS3)
            {
                string dir_name = Path.GetDirectoryName(filename);
                string file_name = Path.GetFileName(filename);

                S3DirectoryInfo s3dir = new S3DirectoryInfo(Form1.S3client, Form1.bucketName, dir_name);
                S3FileInfo artFile = s3dir.GetFile(file_name);
                sw = new StreamWriter(artFile.OpenWrite());
            }
            else
                sw = new StreamWriter(filename, false);

            string line = "";

            for (int i = 0; i < table.Count(); i++)
            {
                line = "";
                for (int j = 0; j < table[i].Count(); j++)
                {
                    line += table[i][j].ToString() + " ";
                }
                sw.WriteLine(line);
            }

            sw.Close();
        }

        public static void calcTreesVariance(List<int>[] table, double[][] label, int Labelindex, string filename)
        {
            List<double> average = new List<double>();
            List<double> averaged_variance = new List<double>();
            
            //calc avg
            for (int i = 0; i < table.Count(); i++)
            {
                double tmpAvg = 0;
                for (int j = 0; j < table[i].Count(); j++)
                {
                    tmpAvg += label[table[i][j]][Labelindex];
                }
                average.Add(tmpAvg / table[i].Count());
            }

            //calc var
            for (int i = 0; i < table.Count(); i++)
            {
                double tmpVar = 0;
                for (int j = 0; j < table[i].Count(); j++)
                {
                    tmpVar += (label[table[i][j]][Labelindex] - average[i]) * (label[table[i][j]][Labelindex] - average[i]);
                }
                averaged_variance.Add(tmpVar / table[i].Count());
            }

            printList(averaged_variance, filename);
        }
        
        public static void printtable(double[][] table, string filename, List<int> intArr)
        {
            StreamWriter sw;
            if (Form1.UseS3)
            {
                string dir_name = Path.GetDirectoryName(filename);
                string file_name = Path.GetFileName(filename);

                S3DirectoryInfo s3dir = new S3DirectoryInfo(Form1.S3client, Form1.bucketName, dir_name);
                S3FileInfo artFile = s3dir.GetFile(file_name);
                sw = new StreamWriter(artFile.OpenWrite());
            }
            else
                sw = new StreamWriter(filename, false);

            string line = "";

            for (int i = 0; i < intArr.Count(); i++)
            {
                line = "";
                for (int j = 0; j < table[i].Count(); j++)
                {
                    line += table[intArr[i]][j].ToString() + " ";
                }
                sw.WriteLine(line);
            }

            sw.Close();
        }

        public static void printList(List<double> lst, string filename)
        {
            StreamWriter sw;
            if (Form1.UseS3)
            {
                string dir_name = Path.GetDirectoryName(filename);
                string file_name = Path.GetFileName(filename);

                S3DirectoryInfo s3dir = new S3DirectoryInfo(Form1.S3client, Form1.bucketName, dir_name);
                S3FileInfo artFile = s3dir.GetFile(file_name);
                sw = new StreamWriter(artFile.OpenWrite());
            }
            else
                sw = new StreamWriter(filename, false);

            for (int i = 0; i < lst.Count(); i++)
            {
                sw.WriteLine(lst[i]);
            }

            sw.Close();
        }

        static public void printConstWavelets2File(List<GeoWave> decision_GeoWaveArr, string filename)
        {
            StreamWriter sw;
            if (Form1.UseS3)
            {
                string dir_name = Path.GetDirectoryName(filename);
                string file_name = Path.GetFileName(filename);

                S3DirectoryInfo s3dir = new S3DirectoryInfo(Form1.S3client, Form1.bucketName, dir_name);
                S3FileInfo artFile = s3dir.GetFile(file_name);
                sw = new StreamWriter(artFile.OpenWrite());
            }
            else
                sw = new StreamWriter(filename, false);
            int dataDim = decision_GeoWaveArr[0].rc.dim;
            int labelDim = decision_GeoWaveArr[0].MeanValue.Count();

            //save metadata

            string line;
            for (int i = 0; i < decision_GeoWaveArr.Count; i++)
            {
                line = "";

                line = decision_GeoWaveArr[i].ID.ToString() + "; " + decision_GeoWaveArr[i].child0.ToString() + "; " + decision_GeoWaveArr[i].child1.ToString() + "; ";
                for (int j = 0; j < dataDim; j++)
                {
                    line += decision_GeoWaveArr[i].boubdingBox[0][j].ToString() + "; " + decision_GeoWaveArr[i].boubdingBox[1][j].ToString() + "; "
                        + MainGrid[j][decision_GeoWaveArr[i].boubdingBox[0][j]].ToString() + "; " + MainGrid[j][decision_GeoWaveArr[i].boubdingBox[1][j]].ToString() + "; ";
                }
                line += decision_GeoWaveArr[i].level + "; ";

                for (int j = 0; j < labelDim; j++)
                {
                    line += decision_GeoWaveArr[i].MeanValue[j].ToString() + "; ";
                }

                line += decision_GeoWaveArr[i].norm + "; " + decision_GeoWaveArr[i].parentID.ToString();

                sw.WriteLine(line);
            }
            sw.Close();
        }

        static public void printLevelWaveletNorm(List<GeoWave> decision_GeoWaveArr, string filename)
        {
            StreamWriter sw;
            if (Form1.UseS3)
            {
                string dir_name = Path.GetDirectoryName(filename);
                string file_name = Path.GetFileName(filename);

                S3DirectoryInfo s3dir = new S3DirectoryInfo(Form1.S3client, Form1.bucketName, dir_name);
                S3FileInfo artFile = s3dir.GetFile(file_name);
                sw = new StreamWriter(artFile.OpenWrite());
            }
            else
                sw = new StreamWriter(filename, false);
            int dataDim = decision_GeoWaveArr[0].rc.dim;
            int labelDim = decision_GeoWaveArr[0].MeanValue.Count();

            for (int i = 0; i < decision_GeoWaveArr.Count; i++)
                sw.WriteLine(decision_GeoWaveArr[i].level + ", " + decision_GeoWaveArr[i].norm);
            
            sw.Close();
        }

        static public void printWaveletsProperties(List<GeoWave> decision_GeoWaveArr, string filename)
        {
            StreamWriter sw;
            if (Form1.UseS3)
            {
                string dir_name = Path.GetDirectoryName(filename);
                string file_name = Path.GetFileName(filename);

                S3DirectoryInfo s3dir = new S3DirectoryInfo(Form1.S3client, Form1.bucketName, dir_name);
                S3FileInfo artFile = s3dir.GetFile(file_name);
                sw = new StreamWriter(artFile.OpenWrite());
            }
            else
                sw = new StreamWriter(filename, false);
            int dataDim = decision_GeoWaveArr[0].rc.dim;
            int labelDim = decision_GeoWaveArr[0].MeanValue.Count();

            sw.WriteLine("norm, level, Npoints, volume, dimSplit, MainGridIndexSplit, MeanValue[0]"); 

            for (int i = 0; i < decision_GeoWaveArr.Count; i++)
            {
                double volume = 1;
                //if (pnt[i] < Form1.MainGrid[i][BoxOfIndeces[0][i]] || pnt[i] > Form1.MainGrid[i][BoxOfIndeces[1][i]])
                for (int j = 0; j < dataDim/*decision_GeoWaveArr[i].boubdingBox[0].Count()*/; j++ )
                    volume *= (Form1.MainGrid[j][decision_GeoWaveArr[i].boubdingBox[1][j]] - Form1.MainGrid[j][decision_GeoWaveArr[i].boubdingBox[0][j]]);

                decision_GeoWaveArr[i].pointsIdArray.Count();
                sw.WriteLine(decision_GeoWaveArr[i].norm + ", " + decision_GeoWaveArr[i].level + ", " 
                             + decision_GeoWaveArr[i].pointsIdArray.Count() + ", " + volume
                             + ", " + decision_GeoWaveArr[i].dimIndex + ", " + decision_GeoWaveArr[i].Maingridindex
                             + ", " + decision_GeoWaveArr[i].MaingridValue + ", " + decision_GeoWaveArr[i].MeanValue[0]);            
            }

            sw.Close();
        }

        public List<GeoWave> GetConstWaveletsFromFile(string filename, recordConfig rc)
        {
            if (!Form1.UseS3 && !File.Exists(filename))//this func was not debugged after modification
            {
                MessageBox.Show("the file " + Path.GetFileName(filename) + " doesnt exist  in " + Path.GetFullPath(filename));
                return null;
            }

            StreamReader sr;
            if (Form1.UseS3)
            {
                string dir_name = Path.GetDirectoryName(filename);
                string file_name = Path.GetFileName(filename);

                S3DirectoryInfo s3dir = new S3DirectoryInfo(Form1.S3client, Form1.bucketName, dir_name);
                S3FileInfo artFile = s3dir.GetFile(file_name);
                sr = artFile.OpenText();
            }
            else
                sr = new StreamReader(File.OpenRead(filename));

            string[] values = { "" };
            string line = "";
            string DimensionReductionMatrix = "";
            int numOfWavlets = -1;
            int dimension = -1;
            int labelDimension = -1;
            double approxOrder = -1;

            while (!sr.EndOfStream && values[0] != "StartReading")
            {
                line = sr.ReadLine();
                values = line.Split(Form1.seperator, StringSplitOptions.RemoveEmptyEntries);
                if (values[0] == "DimensionReductionMatrix")
                    DimensionReductionMatrix = values[1];
                else if (values[0] == "numOfWavlets")
                    numOfWavlets = int.Parse(values[1]);
                else if (values[0] == "approxOrder")
                    approxOrder = int.Parse(values[1]);
                else if (values[0] == "dimension")
                    dimension = int.Parse(values[1]);
                else if (values[0] == "labelDimension")
                    labelDimension = int.Parse(values[1]);
                else if (values[0] == "StartReading")
                { ;}
                else
                    MessageBox.Show("the file " + Path.GetFileName(filename) + " already exist in " + Path.GetFullPath(filename) + " might have bad input !");
            }

            //read values
            List<GeoWave> gwArr = new List<GeoWave>();
            while (!sr.EndOfStream)
            {
                GeoWave gw = new GeoWave(dimension, labelDimension, rc);
                line = sr.ReadLine();
                values = line.Split(Form1.seperator, StringSplitOptions.RemoveEmptyEntries);
                gw.ID = int.Parse(values[0]);
                gw.child0 = int.Parse(values[1]);
                gw.child1 = int.Parse(values[2]);
                int counter = 0;
                for (int j = 0; j < dimension; j++)
                {
                    gw.boubdingBox[0][j] = int.Parse(values[3 + 4 * j]);//the next are the actual values and not the indeces int the maingrid - so we skip 4 elementsat a time
                    gw.boubdingBox[1][j] = int.Parse(values[4 + 4 * j]);
                    counter = 4 + 2 * 4;
                }
                gw.level = int.Parse(values[counter + 1]);
                counter = counter + 2;
                for (int j = 0; j < labelDimension; j++)
                {
                    gw.MeanValue[j] = double.Parse(values[counter + j]);
                    counter++;
                }
                gw.norm = double.Parse(values[counter]);
                gw.parentID = int.Parse(values[counter + 1]);
                gwArr.Add(gw);
            }

            sr.Close();
            return gwArr;
        }
                
        //public List<List<GeoWave>>  GetConstWaveletsFromFolder(string FolderName)
        //{
        //    if (!Directory.Exists(FolderName))
        //    {
        //        MessageBox.Show("the folder " + FolderName + " doesnt exist" );
        //        return null;
        //    }

        //    string[] Dirfilenames = Directory.GetFiles(FolderName);
        //    int numOfBosst = (Dirfilenames.Count() + 1) / 2;//in the same folder we have original labels file and list of norm
        //    List<List<GeoWave>> Boosted_decision_GeoWaveArr = new List<List<GeoWave>>();
        //    for (int i = 0; i < numOfBosst; i++)
        //    {
        //        List<GeoWave> gw = GetConstWaveletsFromFile(FolderName + "\\BosstingTree_" + i.ToString() + ".txt");
        //        Boosted_decision_GeoWaveArr.Add(gw);
        //    }
        //    return Boosted_decision_GeoWaveArr;
        //}

        //public List<List<GeoWave>> GetRFConstWaveletsFromFolder(string FolderName)
        //{
        //    if (!Directory.Exists(FolderName))
        //    {
        //        MessageBox.Show("the folder " + FolderName + " doesnt exist");
        //        return null;
        //    }

        //    string[] Dirfilenames = Directory.GetFiles(FolderName);
        //    int numOfTrees = (Dirfilenames.Count() );//num of trees
        //    List<List<GeoWave>> RF_GeoWaveArr = new List<List<GeoWave>>();
        //    for (int i = 0; i < numOfTrees; i++)
        //    {
        //        List<GeoWave> gw = GetConstWaveletsFromFile(FolderName + "\\RFTree_" + i.ToString() + ".txt");
        //        gw = gw.OrderBy(o => o.ID).ToList();
        //        RF_GeoWaveArr.Add(gw);
        //    }
        //    return RF_GeoWaveArr;
        //}

        //public List<List<GeoWave>> GetBoostingConstWaveletsFromFolder(string FolderName)
        //{
        //    if (!Directory.Exists(FolderName))
        //    {
        //        MessageBox.Show("the folder " + FolderName + " doesnt exist" );
        //        return null;
        //    }

        //    string[] Dirfilenames = Directory.GetFiles(FolderName);
        //    int numOfBosst = (Dirfilenames.Count() -1)  / 2;//in the same folder we have original labels file and list of norm - 2 general files
        //    List<List<GeoWave>> Boosted_decision_GeoWaveArr = new List<List<GeoWave>>();
        //    for (int i = 0; i < numOfBosst; i++)
        //    {
        //        List<GeoWave> gw = GetConstWaveletsFromFile(FolderName + "\\BosstingTree_" + i.ToString() + ".txt");
        //        gw = gw.OrderBy(o => o.ID).ToList();
        //        Boosted_decision_GeoWaveArr.Add(gw);
        //    }
        //    return Boosted_decision_GeoWaveArr;
        //}

        public static bool IsBoxSingular(double[][] Box)
        {
            for (int i = 0; i < Box[0].Count(); i++)
            {
                if (Box[1][i] <= Box[0][i])
                    return true;
            }
            return false;
        }

        public static bool IsBoxSingular(int[][] Box, int dim)
        {
            for (int i = 0; i < dim; i++)
            {
                if (Box[1][i] < Box[0][i])
                    return true;
            }

            if  (Enumerable.SequenceEqual(Box[0], Box[1]))
                return true;
            
            return false;
        }

        public List<double> GetBoostingNormThresholdList(string filename) 
        {
            if (!Form1.UseS3 && !File.Exists(filename))
            {
                MessageBox.Show("the file " + Path.GetFileName(filename) + " doesnt exist  in " + Path.GetFullPath(filename));
                return null;
            }

            StreamReader sr;
            if (Form1.UseS3)
            {
                string dir_name = Path.GetDirectoryName(filename);
                string file_name = Path.GetFileName(filename);

                S3DirectoryInfo s3dir = new S3DirectoryInfo(Form1.S3client, Form1.bucketName, dir_name);
                S3FileInfo artFile = s3dir.GetFile(file_name);
                sr = artFile.OpenText();
            }
            else
                sr = new StreamReader(File.OpenRead(filename));

            string line = sr.ReadLine();
            string[] values = line.Split(Form1.seperator, StringSplitOptions.RemoveEmptyEntries);

            List<double> NormArry = new List<double>();

            for (int i = 0; i < values.Count(); i++)
            {
                NormArry.Add(double.Parse(values[i]));
            }

            sr.Close();
            return NormArry;
        }
        
        public void printErrorsToFile(string filename, double l2, double l1, double l0, double test_size)
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

            //WRITE 
            writer.WriteLine("l2 estimation error: " + l2.ToString());
            writer.WriteLine("l1 estimation error: " + l1.ToString());
            writer.WriteLine("num of miss labels: " + l0.ToString());
            writer.WriteLine("num of tests: " + test_size.ToString());
            writer.WriteLine("sucess rate : " + ( 1 - (l0 / test_size)).ToString());
            writer.Close();            
        }

        private void btnScript_Click(object sender, EventArgs e)
        {        
            set2Config();
            u_config.printConfig(@"C:\Wavelets decomposition\config.txt", null);
            
            // Create a client
            AmazonS3Config confisS3 = new AmazonS3Config();
            confisS3.ProxyHost = null;
            TimeSpan timeOUT = new TimeSpan(3,0,0);
            confisS3.ReadWriteTimeout = timeOUT;
            confisS3.Timeout = timeOUT;
            AmazonS3Client client = new AmazonS3Client(confisS3);

            UseS3 = UseS3CB.Checked;
            rumPrallel = rumPrallelCB.Checked;
            runBoosting = runBoostingCB.Checked;
            runProoning = runProoningCB.Checked;
            runBoostingProoning = runBoostingProoningCB.Checked;
            runRFProoning = runRFProoningCB.Checked;
            runRf = runRfCB.Checked;
            runBoostingLearningRate = runBoostingLearningRateCB.Checked;



            //bucketName = "oren-elisha";
            //string results_path = @"tests/yahoo_set2\article_compression";//@"C:\Users\Administrator\Dropbox\ADA\tmp\results"; // "C:\\Users\\Oren\\Dropbox\\helix\\results"; // 
            //string db_path = @"tests/yahoo_set2\";//@"C:\Users\Administrator\Dropbox\ADA\ada_valid\"; //"D:\\Phd\\Shai\\code\\tests\\helix tests\\noise_5\\noise_5\\"; // "C:\\reasearch\\tests\\lena\\";

            bucketName = bucketTB.Text;
            string results_path = @ResultsTB.Text;
            string db_path = @DBTB.Text+ "\\";//@"C:\Users\Administrator\Dropbox\ADA\ada_valid\"; //"D:\\Phd\\Shai\\code\\tests\\helix tests\\noise_5\\noise_5\\"; // "C:\\reasearch\\tests\\lena\\";


            //get dir

            MainFolderName = results_path;
            if (!UseS3)
            { 
                if(!System.IO.Directory.Exists(MainFolderName))
                    System.IO.Directory.CreateDirectory(MainFolderName);
                //if(!System.IO.Directory.Exists(MainFolderName+"\\archive"))
                //    System.IO.Directory.CreateDirectory(MainFolderName + "\\archive");
            }
            if (UseS3)
            {
                S3DirectoryInfo s3results_path = new S3DirectoryInfo(client, bucketName, results_path);
                if (!s3results_path.Exists)
                    s3results_path.Create();
                //set archive path
                //S3DirectoryInfo s3archive_path = new S3DirectoryInfo(client, bucketName, results_path + "\\archive");
                //if (!s3archive_path.Exists)
                //    s3archive_path.Create();
            }

            ////START TMP LOOP FOR FEATURE TEST
            //for (int t = 0; t < 57; t++ )
            //{
                
                //READ DATA
                DB db = new DB();
                //db.training_dt = db.getDataTableWithNan(db_path + "trainingData.txt", ref trainNaTable);
                //db.testing_dt = db.getDataTableWithNan(db_path + "testingData.txt", ref testNaTable);
                //db.validation_dt = db.getDataTableWithNan(db_path + "ValidData.txt", ref validNaTable);
                //db.training_label = db.getDataTableTMP(db_path + "trainingLabel.txt");
                //db.testing_label = db.getDataTableTMP(db_path + "testingLabel.txt");
                //db.validation_label = db.getDataTableTMP(db_path + "ValidLabel.txt");

                db.training_dt = db.getDataTable(db_path + "trainingData.txt");
                db.testing_dt = db.getDataTable(db_path + "testingData.txt");
                db.validation_dt = db.getDataTable(db_path + "ValidData.txt");
                //db.training_dt = db.getDataTable(db_path + "trainingData" + t.ToString() + ".txt");
                //db.testing_dt = db.getDataTable(db_path + "testingData" + t.ToString() + ".txt");
                //db.validation_dt = db.getDataTable(db_path + "ValidData" + t.ToString() + ".txt");

                db.training_label = db.getDataTable(db_path + "trainingLabel.txt");
                db.testing_label = db.getDataTable(db_path + "testingLabel.txt");
                db.validation_label = db.getDataTable(db_path + "ValidLabel.txt");

                //db.training_label = db.getDataTable(db_path + "trainingLabel" + t.ToString() + ".txt");

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
                ////SET VALIDATION DATA
                //double validationPercent = 1;//10%
                //long rowToRemoveForValidation = Convert.ToInt64(db.testing_dt.Count() * validationPercent);
                //db.validation_dt = db.testing_dt.Where((el, i) => i < rowToRemoveForValidation).ToArray();//take first rows from testing data
                //db.validation_label = db.testing_label.Where((el, i) => i < rowToRemoveForValidation).ToArray();

                //dbg
                //printtable(db.training_dt, MainFolderName + "\\training_dt.txt");
                //printtable(db.testing_dt, MainFolderName + "\\testing_dt.txt");

                ////REDUCE DIM
                if (usePCA.Checked)
                {
                    dimReduction dimreduction = new dimReduction(db.training_dt);
                    db.PCAtraining_dt = dimreduction.getPCA(db.training_dt);
                    db.PCAtesting_dt = dimreduction.getPCA(db.testing_dt);
                    db.PCAvalidation_dt = dimreduction.getPCA(db.validation_dt);

                    //printtable(db.PCAtraining_dt, MainFolderName + "\\PCAtraining_dt.txt");//dbg
                    //printtable(db.PCAtesting_dt, MainFolderName + "\\PCAtesting_dt.txt");//dbg            
                }
                else
                {
                    //de-activate pca for dbg
                    db.PCAtraining_dt = db.training_dt;
                    db.PCAtesting_dt = db.testing_dt;
                    db.PCAvalidation_dt = db.validation_dt;
                }

                // YAIR - adding a new array of training set with place for the PLS
                //db.training_dt_with_PLS = new double[db.PCAtraining_dt.Count()][];
                //for (int i = 0; i < db.PCAtraining_dt.Count(); i++)
                //    db.training_dt_with_PLS[i] = new double[db.PCAtraining_dt[0].Count() + GeoWave.Y_nPLSDim];
                //for (int i = 0; i < db.PCAtraining_dt.Count(); i++)
                //{
                //    double[] tmp = new double[db.training_dt_with_PLS[0].Count()];
                //    for (int j = 0; j < db.PCAtraining_dt[0].Count(); j++) // Fixing the row vector. all the original variables are copied
                //        tmp[j] = db.PCAtraining_dt[i][j];
                //    for (int j = 0; j < GeoWave.Y_nPLSDim; j++)
                //        tmp[j + db.PCAtraining_dt[0].Count()] = i; // j to avoid empty variable...
                //    db.training_dt_with_PLS[i] = tmp;  // only the original data is coppied. the line vector of variables is assigned to the i-th line
                //}

                //db.PCAtraining_dt = db.training_dt_with_PLS;


                db.PCAtraining_GridIndex_dt = new long[db.PCAtraining_dt.Count()][];
                for (int i = 0; i < db.PCAtraining_dt.Count(); i++)
                    db.PCAtraining_GridIndex_dt[i] = new long[db.PCAtraining_dt[i].Count()];

                //BOUNDING BOX AND MAIN GRID              
                boundingBox = db.getboundingBox(db.PCAtraining_dt);
                MainGrid = db.getMainGrid(db.PCAtraining_dt, boundingBox, ref db.PCAtraining_GridIndex_dt);
                //printList(MainGrid[0], MainFolderName + "\\MainGrid_x.txt");//dbg
                //printList(MainGrid[1], MainFolderName + "\\MainGrid_y.txt");//dbg
                //printList(MainGrid[2], MainFolderName + "\\MainGrid_z.txt");//dbg

                //READ CONFIG
                methodConfig mc = new methodConfig(true);
                int Nloops = int.Parse(NloopsTB.Text) - 1;
                int Kfolds = 0;
                if (int.TryParse(croosValidTB.Text, out Kfolds))
                    Nloops = Kfolds - 1;

                for (int k = 0; k < Nloops; k++)
                    mc.boostlamda_0.Add(3.8);// - create variant in number of pixels
                //mc.boostlamda_0.Add(1500);// - create variant in number of pixels
                //mc.boostlamda_0.Add(2500);// - create variant in number of pixels
                //mc.boostlamda_0.Add(3000);// - create variant in number of pixels

                mc.generateRecordConfigArr();
                for (int k = 0; k < mc.recArr.Count(); k++)
                {
                    mc.recArr[k].dim = NfeaturesTB.Text == "all" ? db.PCAtraining_dt[0].Count() : int.Parse(EvaluateString(NfeaturesTB.Text, k));
                    //mc.recArr[k].dim -= GeoWave.Y_nPLSDim;  // reduce the added columns of the PLS
                    mc.recArr[k].approxThresh = double.Parse(EvaluateString(approxThreshTB.Text, k));// 0.1;
                    mc.recArr[k].partitionErrType = int.Parse(EvaluateString(partitionTypeTB.Text, k)); //2;
                    mc.recArr[k].minWaveSize = int.Parse(EvaluateString(minNodeSizeTB.Text, k)); //1;//CHANGE AFTER DBG
                    mc.recArr[k].hopping_size = int.Parse(EvaluateString(waveletsSkipEstimationTB.Text, k)); //25;// 10 + 5 * (k + 1);// +5 * (k % 10);// 1;//25;
                    mc.recArr[k].test_error_size = double.Parse(EvaluateString(waveletsPercentEstimationTB.Text, k));// +0.05 * (k % 10);// 1;// 0.1;//percent of waves to check
                    mc.recArr[k].NskipsinKfunc = double.Parse(EvaluateString(boostingKfuncPercentTB.Text, k));// 0.0025;
                    mc.recArr[k].rfBaggingPercent = double.Parse(EvaluateString(bagginPercentTB.Text, k)); // 0.6;
                    mc.recArr[k].rfNum = int.Parse(EvaluateString(NrfTB.Text, k));// k + 1;//10 + k*10;// 100 / (k + 46) * 2;// int.Parse(Math.Pow(10, k + 1).ToString());
                    mc.recArr[k].boostNum = int.Parse(EvaluateString(NboostTB.Text, k)); // 10;
                    mc.recArr[k].boostProoning_0 = int.Parse(EvaluateString(NfirstPruninginBoostingTB.Text, k)); //13
                    mc.recArr[k].boostlamda_0 = double.Parse(EvaluateString(boostingLamda0TB.Text, k));// 0.01 - (k + 1) * 0.001; //0.05;// 0.0801 + k * 0.001;// Math.Pow(0.1, k);// 0.22 + k*0.005;
                    mc.recArr[k].NwaveletsBoosting = int.Parse(EvaluateString(NfirstwaveletsBoostingTB.Text, k)); //  4;// k + 1;
                    //mc.recArr[k].learningRate = 0;// 0.01;
                    mc.recArr[k].boostNumLearningRate = int.Parse(EvaluateString(NboostingLearningRateTB.Text, k));// 55;// 18;
                    mc.recArr[k].percent_training_db = trainingPercent;
                    mc.recArr[k].BoundLevel = int.Parse(EvaluateString(boundLevelTB.Text, k));//1024;
                    mc.recArr[k].NDimsinRF = NfeaturesrfTB.Text == "all" ? db.PCAtraining_dt[0].Count() : int.Parse(EvaluateString(NfeaturesrfTB.Text, k));
                    //mc.recArr[k].NDimsinRF -= GeoWave.Y_nPLSDim;    // reduce the added columns of the PLS
                    mc.recArr[k].split_type = int.Parse(EvaluateString(splitTypeTB.Text, k)); //0
                    mc.recArr[k].NormLPType = int.Parse(EvaluateString(errTypeEstimationTB.Text, k));
                    mc.recArr[k].RFpruningTestRange[1] = int.Parse(EvaluateString(RFpruningEstimationRange1TB.Text, k)); // 12;// k + 9;
                    mc.recArr[k].boundDepthTree = int.Parse(EvaluateString(boundDepthTB.Text, k));//1024;
                    mc.recArr[k].CrossValidFold = k; 
                    //mc.recArr[k].boostNum =  t ;// tmp to delete !!!!!!!

                    //mc.recArr[k].RFwaveletsTestRange[0] = 25;
                    //mc.recArr[k].RFwaveletsTestRange[1] = 50;
                }

                GeoWave.Y_nPLSDim = mc.recArr[0].NDimsinRF;

                //for (int k = 0; k < mc.recArr.Count(); k++)
                //{
                //    mc.recArr[k].dim =  db.PCAtraining_dt[0].Count();//all
                //    mc.recArr[k].approxThresh = 0.1;
                //    mc.recArr[k].partitionErrType = 2;
                //    mc.recArr[k].minWaveSize = 1;//CHANGE AFTER DBG
                //    mc.recArr[k].hopping_size = 25;// 10 + 5 * (k + 1);// +5 * (k % 10);// 1;//25;
                //    mc.recArr[k].test_error_size = 1;// +0.05 * (k % 10);// 1;// 0.1;//percent of waves to check
                //    mc.recArr[k].NskipsinKfunc = 1;// 0.0025;
                //    mc.recArr[k].rfBaggingPercent = 0.6;
                //    mc.recArr[k].rfNum = 100;// k + 1;//10 + k*10;// 100 / (k + 46) * 2;// int.Parse(Math.Pow(10, k + 1).ToString());
                //    mc.recArr[k].boostNum = 10;
                //    mc.recArr[k].boostProoning_0 = 13;
                //    mc.recArr[k].boostlamda_0 = 0;// 0.01 - (k + 1) * 0.001; //0.05;// 0.0801 + k * 0.001;// Math.Pow(0.1, k);// 0.22 + k*0.005;
                //    mc.recArr[k].NwaveletsBoosting = 4;// k + 1;
                //    //mc.recArr[k].learningRate = 0;// 0.01;
                //    mc.recArr[k].boostNumLearningRate = 55;// 18;
                //    mc.recArr[k].percent_training_db = trainingPercent;
                //    mc.recArr[k].BoundLevel = 1024;
                //    mc.recArr[k].NDimsinRF =  db.PCAtraining_dt[0].Count();
                //    mc.recArr[k].split_type = 0;
                //    mc.recArr[k].NormLPType = 2;
                //    mc.recArr[k].RFpruningTestRange[1] = 12;// k + 9;

                //    //mc.recArr[k].RFwaveletsTestRange[0] = 25;
                //    //mc.recArr[k].RFwaveletsTestRange[1] = 50;
                //}

                //mc.recArr

                //create dirs
                for (int i = 0; i < mc.recArr.Count; i++)
                {
                    if (!UseS3 && !System.IO.Directory.Exists(MainFolderName + "\\" + mc.recArr[i].getShortName()))
                    {
                        System.IO.Directory.CreateDirectory(MainFolderName + "\\" + mc.recArr[i].getShortName());
                        StreamWriter sw = new StreamWriter(MainFolderName + "\\" + mc.recArr[i].getShortName() + "\\record_properties.txt", false);
                        sw.WriteLine(mc.recArr[i].getFullName());
                        sw.Close();
                        u_config.printConfig(MainFolderName + "\\config.txt", null);
                    }
                    if (UseS3)
                    {
                        S3DirectoryInfo s3results_path_with_folders = new S3DirectoryInfo(client, bucketName, results_path + "\\" + mc.recArr[i].getShortName());
                        if (!s3results_path_with_folders.Exists)
                        {
                            s3results_path_with_folders.Create();
                            S3FileInfo outFile = s3results_path_with_folders.GetFile("record_properties.txt");
                            StreamWriter sw = new StreamWriter(outFile.OpenWrite());
                            sw.WriteLine(mc.recArr[i].getFullName());
                            sw.Close();

                            S3FileInfo configFile = s3results_path_with_folders.GetFile("config.txt");
                            u_config.printConfig("", configFile);
                        }
                    }
                }

                //SET ID ARRAY LIST
                List<int> trainingID = Enumerable.Range(0, db.PCAtraining_dt.Count()).ToList();
                List<int> testingID = Enumerable.Range(0, db.PCAtesting_dt.Count()).ToList();

                //cross validation
                List<List<int>> trainingFoldId = new List<List<int>>();
                List<List<int>> testingFoldId = new List<List<int>>();

                var ran = new Random(2);
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

                
            //find nearest neigbores
//            Dictionary<int, List<int>> trainingKnnToMe = new Dictionary<int,List<int>>();
//            Dictionary<int, List<int>> trainingKnnFromMe = new Dictionary<int, List<int>>();

//            getKNN(trainingID, db.PCAtraining_dt, ref trainingKnnToMe, ref trainingKnnFromMe);
            
            
            //PARALLEL FOR ANALIZE
                //if (Form1.rumPrallel)
                //{
                //    Parallel.For(0, mc.recArr.Count, i =>
                //    {
                //        analizer Analizer = new analizer(MainFolderName + "\\" + mc.recArr[i].getFullName(), MainGrid, db, mc.recArr[i]);
                //        Analizer.analize(trainingID, testingID, BB);
                //    }); 
                //}
                //else
                //{
                for (int i = 0; i < mc.recArr.Count; i++)
                {
                    analizer Analizer = new analizer(MainFolderName + "\\" + mc.recArr[i].getShortName(), MainGrid, db, mc.recArr[i]);
                    if (!croosValidCB.Checked)
                        Analizer.analize(trainingID, testingID, BB);
                    else
                        Analizer.analize(trainingFoldId[i], testingFoldId[i], BB);//cross validation
                    Console.Beep();

                }
                //} 
            //}
            //end tmp loop

            btnScript.BackColor = Color.Green;
        }

        private void getKNN(List<int> IdList, double[][] dt, ref Dictionary<int, List<int>> KnnToMe, ref Dictionary<int, List<int>> KnnFromMe)
        {
            //SET DICTIONARIES
            for(int i = 0; i < IdList.Count; i++)
            {
                KnnToMe.Add(IdList[i],new List<int>());
                KnnFromMe.Add(IdList[i],new List<int>());
            }
            
            double dist = 0;
            //for each ID find closest (Euclidean measure) ID
            for (int i = 0; i < IdList.Count; i++)
            {
                double minDist = double.MaxValue;
                int minID = -1;
                for (int j = i + 1; j < IdList.Count; j++)
                {
                    dist = getDistIJ(dt, IdList[i], IdList[j]);
                    if(dist < minDist)
                    {
                        minDist = dist;
                        minID = IdList[j];                   
                    }
                }
                KnnToMe[IdList[i]].Add(minID);
                KnnFromMe[minID].Add(IdList[i]);
            }

        }

        private double getDistIJ(double[][] dt, int indexI, int indexJ)
        {
            //consider missing values
            int length = dt[0].Count();
            double dist = 0;
            for(int k=0; k < length; k++)
                dist += (dt[indexI][k] - dt[indexJ][k]) * (dt[indexI][k] - dt[indexJ][k]);
            return dist;
        }

        //THE LARGEST GROUP IS TRAINING
        private void createCrossValid(int Kfolds, List<int> trainingID, List<List<int>> trainingFoldId, List<List<int>> testingFoldId)
        {
            //ADD LISTS
            for (int i = 0; i < Kfolds; i++)
            {
                trainingFoldId.Add(new List<int>());
                testingFoldId.Add(new List<int>());
            }

            int Npoints = trainingID.Count / Kfolds;
            //ADD POINTS ID
            int upper_bound = Npoints;
            int lower_bound = -1;
            int counter =-1;
            for (int i = 0; i < trainingID.Count; i++)
            {
                if (i % Npoints == 0)
                {
                    counter++;//should happen Kfolds times
                    //if (i == (Kfolds * Npoints))
                    //    counter--;
                }
                
                for (int j = 0; j < Kfolds; j++)
                {
                    if(j==counter)
                        testingFoldId[j].Add(trainingID[i]);
                    else
                        trainingFoldId[j].Add(trainingID[i]);
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
            if (u_config.runOneTreeCB == "1")
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
            u_config.runOneTreeCB = runOneTreeCB.Checked ? "1" : "0";
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

    }
}


//while (rdForm.method_calced == false)
//{
//    System.Threading.Thread.Sleep(250); // pause for 250 mili seconds;
//};
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DataSetsSparsity;
using System.IO;

namespace wf
{
    static public class Program
    {
        static void Main(string[] args)
        {
            userConfig.readConfig("config.txt");
            //Run();
            userConfig.logger.Close();
            //var directories = Directory.GetDirectories(@"C:\Users\Oren E\Documents\Visual Studio 2015\Projects\wf\procDataSets\classification");
            //var directories = Directory.GetDirectories(@"C:\Users\Oren E\Documents\Visual Studio 2015\Projects\wf\dataSets\cifar10\layer4");
            var directories = Directory.GetDirectories(@"C:\Users\Oren E\Google Drive\phd\spiralTest");     
            //var directories = Directory.GetDirectories(@"C: \Users\Oren E\Documents\Visual Studio 2015\Projects\wf\procDataSets\regression");
            foreach (string dir in directories)
            {
                userConfig.readConfig("config.txt");
                userConfig.dbPath = dir;
                userConfig.resultsPath = dir + "\\results";
                if (!Directory.Exists(userConfig.resultsPath))
                    Directory.CreateDirectory(userConfig.resultsPath);
                //save config in results path
                File.Copy("config.txt", userConfig.resultsPath + "\\" + "config.txt", true);
                Run();
                userConfig.logger.Close();//close log file
            }
        }
        public static void remoteRun(string configfile)
        {
            userConfig.readConfig(configfile);
            if (!Directory.Exists(userConfig.resultsPath))
                Directory.CreateDirectory(userConfig.resultsPath);
            File.Copy(configfile, userConfig.resultsPath + "\\" + "config.txt", true);
            Run();

        }
        public static double[][] boundingBox;
        public static List<List<double>> MainGrid;
        public static string MainFolderName; //THE DIR OF THE ROOT FOLDER
        public static string[] seperator = { " ", ";", "/t", "/n", "," };

        public static userConfig u_config = new userConfig();

        public static void printtable(List<int>[] table, string filename)
        {
            StreamWriter sw = new StreamWriter(filename, false);

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

        public static void printList(List<double> lst, string filename)
        {
            StreamWriter sw = new StreamWriter(filename, false);

            for (int i = 0; i < lst.Count(); i++)
            {
                sw.WriteLine(lst[i]);
            }

            sw.Close();
        }

        static public void printConstWavelets2File(List<GeoWave> decision_GeoWaveArr, string filename)
        {
            StreamWriter sw = new StreamWriter(filename, false);
            int dataDim = boundingBox[0].Count();
            int labelDim = decision_GeoWaveArr[0].MeanValue.Count();

            //PRINT METADATA
            sw.WriteLine("dimension," + dataDim);
            sw.WriteLine("labelDimension," + labelDim);
            sw.WriteLine("StartReading");

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

                line += decision_GeoWaveArr[i].norm + "; " + decision_GeoWaveArr[i].parentID.ToString() + "; ";

                line += decision_GeoWaveArr[i].dimIndex.ToString() + "; " + decision_GeoWaveArr[i].MaingridValue.ToString() + "; ";//SPLITTED

                line += decision_GeoWaveArr[i].dimIndexSplitter.ToString() + "; " + decision_GeoWaveArr[i].splitValue.ToString();//SPLITTER

                sw.WriteLine(line);
            }
            sw.Close();
        }

        static public void printWaveletsProperties(List<GeoWave> decision_GeoWaveArr, string filename)
        {
            StreamWriter sw = new StreamWriter(filename, false);
            int dataDim = boundingBox[0].Count();
            int labelDim = decision_GeoWaveArr[0].MeanValue.Count();

            sw.WriteLine("norm, level, Npoints, dimSplit, MainGridIndexSplit", "MaingridValue");

            for (int i = 0; i < decision_GeoWaveArr.Count; i++)
            {
                sw.WriteLine(decision_GeoWaveArr[i].norm + ", " + decision_GeoWaveArr[i].level + ", " + decision_GeoWaveArr[i].pointsIdArray.Count()
                                                         + ", " + decision_GeoWaveArr[i].dimIndex + ", " + decision_GeoWaveArr[i].Maingridindex
                                                         + ", " + decision_GeoWaveArr[i].MaingridValue);
            }

            sw.Close();
        }

        public static bool IsBoxSingular(int[][] Box, int dim)
        {
            for (int i = 0; i < dim; i++)
            {
                if (Box[1][i] < Box[0][i])
                    return true;
            }

            if (Enumerable.SequenceEqual(Box[0], Box[1]))
                return true;

            return false;
        }

        static private void Run()
        {
            //READ DATA
            DB db = new DB();
            db.training_label = db.getDataTable(userConfig.dbPath + "\\trainingLabel.txt");
            if (userConfig.setInhyperCube)
                db.training_dt = db.getcleanDb(userConfig.dbPath + "\\trainingData.txt", db.training_label.Count());
            else
                db.training_dt = db.getDataTable(userConfig.dbPath + "\\trainingData.txt");
            if (userConfig.useCV)
            {
                db.testing_dt = db.training_dt;
                db.testing_label = db.training_label;
            }
            else
            {
                db.testing_label = db.getDataTable(userConfig.dbPath + "\\testingLabel.txt");
                if (userConfig.setInhyperCube)
                    db.testing_dt = db.getcleanDb(userConfig.dbPath + "\\testingData.txt", db.testing_label.Count());
                else
                    db.testing_dt = db.getDataTable(userConfig.dbPath + "\\testingData.txt");
            }
            db.validation_dt = db.testing_dt;
            db.validation_label = db.testing_label;

            //set features
            setDim(db.training_dt[0].Count());

            //BOUNDING BOX AND GRID  
            db.DBtraining_GridIndex_dt = new long[db.training_dt.Count()][];
            for (int i = 0; i < db.training_dt.Count(); i++)
                db.DBtraining_GridIndex_dt[i] = new long[db.training_dt[i].Count()];

            boundingBox = db.getboundingBox(db.training_dt);
            MainGrid = db.getMainGrid(db.training_dt, boundingBox, ref db.DBtraining_GridIndex_dt);

            ////List<recordConfig> recArr = new List<recordConfig>();
            //string[] folds = null;
            //if (userConfig.useCV)
            //    folds = new string[userConfig.nCv];
            //else


            if (userConfig.useCV && userConfig.nCv ==0)
                userConfig.logger.WriteLine("Num of Cross validation folders wasn't provided");

            //SET ID ARRAY LIST
            List<int> trainingID = Enumerable.Range(0, db.training_dt.Count()).ToList();
            List<List<int>> trainingFoldId = new List<List<int>>();
            List<List<int>> testingFoldId = new List<List<int>>();
            List<List<int>> validatingFoldId = new List<List<int>>();
            var ran = new Random(2);
            List<int> training_rand = trainingID.OrderBy(x => ran.Next()).ToList().GetRange(0, trainingID.Count);
            if (userConfig.useCV)
            {
                createCrossValid(userConfig.nCv, training_rand, trainingFoldId, testingFoldId);
                splitTrain2Valid(trainingFoldId, validatingFoldId);
            }


            //debug - makeSureNoIntersection(trainingFoldId[0], validatingFoldId[0], testingFoldId[0]);
                

            //BOUNDING INTERVALS
            int[][] BB = new int[2][];
            BB[0] = new int[boundingBox[0].Count()];
            BB[1] = new int[boundingBox[0].Count()];
            for (int i = 0; i < boundingBox[0].Count(); i++)
            {
                BB[1][i] = MainGrid[i].Count() - 1;//set last index in each dim
            }
            List<int> testingID = Enumerable.Range(0, db.testing_dt.Count()).ToList();
            List<int> validatingID = Enumerable.Range(0, db.testing_dt.Count()).ToList();
            int n_iter = (userConfig.useCV) ? userConfig.nCv : 1;
            for (int i = 0; i < n_iter; i++)
            {
                if (!userConfig.useCV)
                {
                    analizer Analizer = new analizer(userConfig.resultsPath, MainGrid, db);
                    Analizer.analize(trainingID, testingID, validatingID, BB);
                }

                else
                {
                    string fold_path = userConfig.resultsPath + "\\" + i.ToString();
                    if (!Directory.Exists(fold_path))
                        Directory.CreateDirectory(fold_path);
                    analizer Analizer = new analizer(fold_path, MainGrid, db);
                    Analizer.analize(trainingFoldId[i], testingFoldId[i], validatingFoldId[i], BB);//cross validation
                }
                    
            }
        }

        private static void makeSureNoIntersection(List<int> list1, List<int> list2, List<int> list3)
        {
            int veryBad = 0;
            for (int i = 1; i < list1.Count(); i++)
                for (int j = 1; j < list2.Count(); j++)
                {
                    if (list1[i] == list2[j])
                        veryBad++;
                }
        }

        private static void setDim(int dim)
        {
            if (userConfig.nFeaturesStr == "all")
                userConfig.nFeatures = dim;
            else if (userConfig.nFeaturesStr == "sqrt")
                userConfig.nFeatures = (int)Math.Ceiling((Convert.ToDouble(Math.Sqrt(dim))));
            else if (userConfig.nFeaturesStr == "div")
                userConfig.nFeatures = (int)Math.Ceiling((Convert.ToDouble(dim / 3)));
            else
                userConfig.nFeatures = int.Parse(userConfig.nFeaturesStr);
        }

        //THE LARGEST GROUP IS TRAINING
        static private void createCrossValid(int Kfolds, List<int> trainingID, List<List<int>> trainingFoldId, List<List<int>> testingFoldId)
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
            int counter = -1;
            for (int i = 0; i < trainingID.Count; i++)
            {
                if (i % Npoints == 0)
                {
                    counter++;//should happen Kfolds times
                }

                for (int j = 0; j < Kfolds; j++)
                {
                    if (j == counter)
                        testingFoldId[j].Add(trainingID[i]);
                    else
                        trainingFoldId[j].Add(trainingID[i]);
                }
            }
        }

        static private void splitTrain2Valid(List<List<int>> trainingFoldId, List<List<int>> validatingFoldId)
        {
            int Kfolds = trainingFoldId.Count();
            
            for (int j = 0; j < Kfolds; j++)
            {
                validatingFoldId.Add(new List<int>());
                //take 10%
                int tenner = trainingFoldId[j].Count() / 10;
                for (int i = 0; i < tenner; i++)
                {
                    validatingFoldId[j].Add(trainingFoldId[j][i]);
                    trainingFoldId[j].RemoveAt(i);
                }
            }
        }

        public static void applyFor(int begin, int size, Action<int> body)
        {
            if (userConfig.useParallel) Parallel.For(begin, size, body);
            else regularDelegateFor(begin, size, body);
        }

        private static void regularDelegateFor(int begin, int size, Action<int> body)
        {
            for (int i = begin; i < size; i++)
            {
                body.Invoke(i);
            }
        }

        public static void LinearRegression(double[] xVals, double[] yVals,
                                    int inclusiveStart, int exclusiveEnd,
                                    out double rsquared, out double yintercept,
                                    out double slope)
        {
            //Debug.Assert(xVals.Length == yVals.Length);
            double sumOfX = 0;
            double sumOfY = 0;
            double sumOfXSq = 0;
            double sumOfYSq = 0;
            double ssX = 0;
            double ssY = 0;
            double sumCodeviates = 0;
            double sCo = 0;
            double count = exclusiveEnd - inclusiveStart;

            for (int ctr = inclusiveStart; ctr < exclusiveEnd; ctr++)
            {
                double x = xVals[ctr];
                double y = yVals[ctr];
                sumCodeviates += x * y;
                sumOfX += x;
                sumOfY += y;
                sumOfXSq += x * x;
                sumOfYSq += y * y;
            }
            ssX = sumOfXSq - ((sumOfX * sumOfX) / count);
            ssY = sumOfYSq - ((sumOfY * sumOfY) / count);
            double RNumerator = (count * sumCodeviates) - (sumOfX * sumOfY);
            double RDenom = (count * sumOfXSq - (sumOfX * sumOfX))
             * (count * sumOfYSq - (sumOfY * sumOfY));
            sCo = sumCodeviates - ((sumOfX * sumOfY) / count);

            double meanX = sumOfX / count;
            double meanY = sumOfY / count;
            double dblR = RNumerator / Math.Sqrt(RDenom);
            rsquared = dblR * dblR;
            yintercept = meanY - ((sCo / ssX) * meanX);
            slope = sCo / ssX;
        }
    }
}

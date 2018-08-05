
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace utilities
{
    class Program
    {
        static void Main(string[] args)
        {
            createMissLabeling(@"C:\Users\Oren E\Google Drive\phd\spiralTest\clean", @"C:\Users\Oren E\Google Drive\phd\spiralTest\noise20");
            //var directories = Directory.GetDirectories(@"C:\Users\Oren E\Documents\Visual Studio 2015\Projects\wf\dataSets");
            //var directories = Directory.GetDirectories(@"C:\Users\Oren E\Documents\Visual Studio 2015\Projects\wf\dataSets\cifar10\layer4");
            //var directories = Directory.GetDirectories(@"C:\Users\Administrator\Documents\cifar10\embedding_many_epocs_total"); 
            //foreach (string dir in directories)
            //{
            //    //string dbPathOut = dir.Replace("layer3", "proclayer3");
            //    //System.IO.Directory.CreateDirectory(dbPathOut);
            //    //cleanDb(dir, dir, false);
            //    createMultiClass(dir, dir, 10);

            //}
                ////string dbPathIn = @"C:\Users\Oren E\Documents\Visual Studio 2015\Projects\wf\dataSets\wine";
                ////string dbPathOut = @"C:\Users\Oren E\Documents\Visual Studio 2015\Projects\wf\procDataSets\wine";
                //if (!System.IO.Directory.Exists(dbPathOut))
                //System.IO.Directory.CreateDirectory(dbPathOut);
                //cleanDb(dbPathIn, dbPathOut);
        }

        private static void createMissLabeling(string dbPathIn, string dbPathOut)
        {
            string filename = dbPathIn + "\\trainingLabel.txt";
            string filenameOut = dbPathOut + "\\trainingLabel.txt";
            StreamReader reader = new StreamReader(File.OpenRead(filename));
            StreamWriter sw = new StreamWriter(filenameOut, false);
            var ran = new Random(0);
            while (!reader.EndOfStream)
            {
                string line = reader.ReadLine();
                int num = ran.Next(0, 10);
                if (num < 2)
                    line = (line == "0") ? "1" : "0";
                sw.WriteLine(line);
            }
            reader.Close();
            sw.Close();
        }

        private static void createMultiClass(string dbPathIn, object dbPathOut,int Nclass)
        {
            string filename = dbPathIn + "\\labels_total.csv";
            string filenameOut = dbPathOut + "\\trainingLabel.txt";
            StreamReader reader = new StreamReader(File.OpenRead(filename));
            StreamWriter sw = new StreamWriter(filenameOut, false);

            while (!reader.EndOfStream)
            {
                int[] arrInt = new int[Nclass];
                string line = reader.ReadLine();
                string[] values = line.Split(seperator, StringSplitOptions.RemoveEmptyEntries);
                decimal d = Decimal.Parse(values[0], System.Globalization.NumberStyles.Float);
                int index = Convert.ToInt32(d);
                arrInt[index] = 1;
                string result = string.Join(",", arrInt);
                sw.WriteLine(result);
            }
            reader.Close();
            sw.Close();
        }

        private static void cleanDb(string dbPathIn, string dbPathOut,bool cleanLabels)
        {

            //double[][] training_dt = getDataTable(dbPathIn + "\\trainingData.txt");
            //double[][] training_label = getDataTable(dbPathIn + "\\trainingLabel.txt");
            double[][] training_dt = getDataTable(dbPathIn + "\\input.csv");
            double[][] training_label = getDataTable(dbPathIn + "\\labels_total.csv");
            //FOR MULTICLASS USE: 
            double[] minsD = new double[training_dt[0].Count()];
            double[] maxsD = new double[training_dt[0].Count()];
            for (int i = 0; i < training_dt[0].Count(); i++)
            {
                minsD[i] = double.MaxValue;
                maxsD[i] = double.MinValue;
            }
            double minL = double.MaxValue;
            double maxL = double.MinValue;
            for (int i = 0; i < training_dt.Count(); i++)
            {
                for (int j = 0; j < training_dt[0].Count(); j++)
                {
                    minsD[j] = (training_dt[i][j] < minsD[j]) ? training_dt[i][j] : minsD[j];
                    maxsD[j] = (training_dt[i][j] > maxsD[j]) ? training_dt[i][j] : maxsD[j];
                }
                minL = (training_label[i][0] < minL) ? training_label[i][0] : minL;
                maxL = (training_label[i][0] > maxL) ? training_label[i][0] : maxL;
            }

            for (int i = 0; i < training_dt.Count(); i++)
            {
                for (int j = 0; j < training_dt[0].Count(); j++)
                {
                    training_dt[i][j] = (training_dt[i][j] - minsD[j]) / (maxsD[j] - minsD[j]);
                }
                training_label[i][0] = (training_label[i][0] - minL) / (maxL - minL);
            }
            if (!Directory.Exists(dbPathOut))
                Directory.CreateDirectory(dbPathOut);
            printtable(training_dt, dbPathOut + "\\trainingData.txt");
            if (cleanLabels)
                printtable(training_label, dbPathOut + "\\trainingLabel.txt");
        }

        private static double[][] getcleanDb(string dbPathIn)
        {
            double[][] training_dt = getDataTable(dbPathIn + "\\input.csv");
            //FOR MULTICLASS USE: 
            double[] minsD = new double[training_dt[0].Count()];
            double[] maxsD = new double[training_dt[0].Count()];
            for (int i = 0; i < training_dt[0].Count(); i++)
            {
                minsD[i] = double.MaxValue;
                maxsD[i] = double.MinValue;
            }
            for (int i = 0; i < training_dt.Count(); i++)
            {
                for (int j = 0; j < training_dt[0].Count(); j++)
                {
                    minsD[j] = (training_dt[i][j] < minsD[j]) ? training_dt[i][j] : minsD[j];
                    maxsD[j] = (training_dt[i][j] > maxsD[j]) ? training_dt[i][j] : maxsD[j];
                }
            }

            for (int i = 0; i < training_dt.Count(); i++)
            {
                for (int j = 0; j < training_dt[0].Count(); j++)
                {
                    training_dt[i][j] = (training_dt[i][j] - minsD[j]) / (maxsD[j] - minsD[j]);
                }
            }
            return training_dt;
        }

        static public string[] seperator = { " ", ";", "/t", "/n", "," };



        static public double[][] getDataTable(string filename)
        {
            StreamReader reader;
            long lineCount = 0;

            if (!File.Exists(filename))//IF NO VALID EXISTS - TRY WITH TEST
                filename = filename.Replace("Valid", "testing");
            if (!File.Exists(filename))//IF NO TESTING EXISTS - TRY WITH TRAINING
                filename = filename.Replace("testing", "training");


            lineCount = countLines(filename);
            reader = new StreamReader(File.OpenRead(filename));
            //lineCount = File.ReadAllLines(filename).Length;


            //GET THE FIRST LINE 
            string line = reader.ReadLine();
            string[] values = line.Split(seperator, StringSplitOptions.RemoveEmptyEntries);

            //IF NO VALUES ALERT
            if (values.Count() < 1)
                return null;

            double[][] dt = new double[lineCount][];
            dt[0] = new double[values.Count()];
            for (int j = 0; j < values.Count(); j++)
                dt[0][j] = double.Parse(values[j]);

            //SET VALUES TO TABLE
            int counter = 1;
            while (!reader.EndOfStream)
            {
                line = reader.ReadLine();
                values = line.Split(seperator, StringSplitOptions.RemoveEmptyEntries);
                dt[counter] = new double[values.Count()];
                for (int j = 0; j < values.Count(); j++)
                    dt[counter][j] = double.Parse(values[j]);
                counter++;
            }

            reader.Close();

            return dt;
        }

        private static long countLines(string filename)
        {
            int counter = 0;
            StreamReader reader = new StreamReader(File.OpenRead(filename));
            string line = "";

            while (!reader.EndOfStream)
            {
                line = reader.ReadLine();
                counter++;
            }

            reader.Close();
            return counter;
        }

        public static void printtable(double[][] table, string filename)
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
    }
}

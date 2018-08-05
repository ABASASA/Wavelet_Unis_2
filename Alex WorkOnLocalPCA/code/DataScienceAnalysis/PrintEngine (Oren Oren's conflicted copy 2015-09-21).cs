using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using Amazon.S3.IO;

namespace DataScienceAnalysis
{
    public static class PrintEngine
    {
        public static void printSplitByComponentHistogram(List<GeoWave>[] rfTreeArr, string analysisFolderName)
        {
            recordConfig rc = rfTreeArr.First().First().rc;
            //relevant only for local pca split
            if (rc.split_type != 5)
            {
                return;
            }
            StreamWriter sw = new StreamWriter(analysisFolderName + "\\localPCAsplitHistogram.txt", false); 
            //get config from first tree and first node
            int[] splitHist =new int[rc.dim];
            foreach (GeoWave node in (from tree in rfTreeArr from node in tree where node.dimIndex != -1 select node))
            {
                splitHist[node.dimIndex]++;
            }
            foreach (int splitDimCounter in splitHist)
            {
                sw.Write(splitDimCounter + " ");
            }
            sw.Close();
        }
        public static void printForestProperties(List<GeoWave>[] rfTreeArr, string analysisFolderName)
        {
            if (Form1.u_config.saveTressCB != "1") return;
            if (!Directory.Exists(analysisFolderName + "\\archive"))  Directory.CreateDirectory(analysisFolderName + "\\archive");
            for (int i = 0; i < rfTreeArr.Count(); i++)
            {
                PrintEngine.printWaveletsProperties(rfTreeArr[i], analysisFolderName + "\\archive\\waveletsPropertiesTree_" + i.ToString() + ".txt");
            }
        }
       public static  void printWaveletsProperties(List<GeoWave> decisionGeoWaveArr, string filename)
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

            recordConfig rc = decisionGeoWaveArr[0].rc;
            //different log for local pca split
            if (rc.split_type == 5)
            {
                printPcaWaveletsProperties(decisionGeoWaveArr, sw);
                return;
            }

            sw.WriteLine("norm, level, Npoints, volume, dimSplit, MainGridIndexSplit");

            foreach (GeoWave t in decisionGeoWaveArr)
            {
                double volume = 1;
                //if (pnt[i] < Form1.MainGrid[i][BoxOfIndeces[0][i]] || pnt[i] > Form1.MainGrid[i][BoxOfIndeces[1][i]])
                for (int j = 0; j < t.boubdingBox[0].Count(); j++)
                    volume *= (Form1.MainGrid[j][t.boubdingBox[1][j]] - Form1.MainGrid[j][t.boubdingBox[0][j]]);

                sw.WriteLine(t.norm + ", " + t.level + ", "
                             + t.pointsIdArray.Count() + ", " + volume
                             + ", " + t.dimIndex + ", " + t.Maingridindex
                             + ", " + t.MaingridValue);
            }

            sw.Close();
        }

        static public void printPcaWaveletsProperties(List<GeoWave> geoWaveArr, StreamWriter sw)
        {
            const int dg = 4; //debug round digits
            sw.WriteLine("norm\t\tlevel\t\tNpoints\t\tpcaDim\t\tdimSplit\t\tpcaUpperSplitValue");
            foreach (GeoWave n in geoWaveArr)
            {

                sw.WriteLine(Math.Round(n.norm, dg) + "\t\t" + n.level + "\t\t "
                             + n.pointsIdArray.Count() + "\t\t" + n.pcaDim
                             + "\t\t " + n.dimIndex + "\t\t " + Math.Round(n.pcaUpperSplitValue, dg));
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

            for (int i = 0; i < table.Count(); i++)
            {
                string line = "";
                for (int j = 0; j < table[i].Count(); j++)
                {
                    line += table[i][j].ToString(CultureInfo.InvariantCulture) + " ";
                }
                sw.WriteLine(line);
            }

            sw.Close();
        }
//************************************** PRINT FUNCTIONS WITH OUT REFERENCES ****************************************************
        public static void printErrorsToFile(string filename, double l2, double l1, double l0, double testSize)
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
            writer.WriteLine("l2 estimation error: " + l2.ToString(CultureInfo.InvariantCulture));
            writer.WriteLine("l1 estimation error: " + l1.ToString(CultureInfo.InvariantCulture));
            writer.WriteLine("num of miss labels: " + l0.ToString(CultureInfo.InvariantCulture));
            writer.WriteLine("num of tests: " + testSize.ToString(CultureInfo.InvariantCulture));
            writer.WriteLine("sucess rate : " + (1 - (l0 / testSize)).ToString(CultureInfo.InvariantCulture));
            writer.Close();
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

            for (int i = 0; i < intArr.Count(); i++)
            {
                string line = "";
                for (int j = 0; j < table[i].Count(); j++)
                {
                    line += table[intArr[i]][j].ToString(CultureInfo.InvariantCulture) + " ";
                }
                sw.WriteLine(line);
            }

            sw.Close();
        }
        static public void printConstWavelets2File(List<GeoWave> decisionGeoWaveArr, string filename)
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
            int dataDim = decisionGeoWaveArr[0].rc.dim;
            int labelDim = decisionGeoWaveArr[0].MeanValue.Count();

            //save metadata

            foreach (GeoWave t in decisionGeoWaveArr)
            {
                string line = t.ID.ToString() + "; " + t.child0.ToString() + "; " + t.child1.ToString() + "; ";
                for (int j = 0; j < dataDim; j++)
                {
                    line += t.boubdingBox[0][j].ToString() + "; " + t.boubdingBox[1][j].ToString() + "; "
                            + Form1.MainGrid[j][t.boubdingBox[0][j]].ToString(CultureInfo.InvariantCulture) + "; " +
                            Form1.MainGrid[j][t.boubdingBox[1][j]].ToString(CultureInfo.InvariantCulture) + "; ";
                }
                line += t.level + "; ";

                for (int j = 0; j < labelDim; j++)
                {
                    line += t.MeanValue[j].ToString(CultureInfo.InvariantCulture) + "; ";
                }

                line += t.norm + "; " + t.parentID.ToString();

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

            for (int i = 0; i < table.Count(); i++)
            {
                var line = "";
                for (int j = 0; j < table[i].Count(); j++)
                {
                    line += table[i][j].ToString() + " ";
                }
                sw.WriteLine(line);
            }

            sw.Close();
        }


        static public void printLevelWaveletNorm(List<GeoWave> decisionGeoWaveArr, string filename)
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
            /*  int dataDim = decisionGeoWaveArr[0].rc.dim;
              int labelDim = decisionGeoWaveArr[0].MeanValue.Count();*/

            foreach (GeoWave t in decisionGeoWaveArr)
                sw.WriteLine(t.level + ", " + t.norm);

            sw.Close();
        }


        public static List<double> getBoostingNormThresholdList(string filename)
        {
            if (!Form1.UseS3 && !File.Exists(filename))
            {
                MessageBox.Show(@"the file " + Path.GetFileName(filename) + @" doesnt exist  in " + Path.GetFullPath(filename));
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
            List<double> NormArry = new List<double>();
            if (line != null)
            {
                string[] values = line.Split(Form1.seperator, StringSplitOptions.RemoveEmptyEntries);

                for (int i = 0; i < values.Count(); i++)
                {
                    NormArry.Add(double.Parse(values[i]));
                }

                sr.Close();

            }
            return NormArry;
        }
        public static List<GeoWave> getConstWaveletsFromFile(string filename, recordConfig rc)
        {
            if (!Form1.UseS3 && !File.Exists(filename))//this func was not debugged after modification
            {
                MessageBox.Show(@"the file " + Path.GetFileName(filename) + @" doesnt exist  in " + Path.GetFullPath(filename));
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
            string line;
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
                    MessageBox.Show(@"the file " + Path.GetFileName(filename) + @" already exist in " + Path.GetFullPath(filename) + @" might have bad input !");
            }

            //read values
            List<GeoWave> gwArr = new List<GeoWave>();
            while (!sr.EndOfStream)
            {
                GeoWave gw = new GeoWave(dimension, labelDimension, rc);
                line = sr.ReadLine();
                if (line != null) values = line.Split(Form1.seperator, StringSplitOptions.RemoveEmptyEntries);
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

    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

using System.Threading;

namespace DataSetsSparsity
{
    public class userConfig
    {
        static public string[] seperator = { ";", "/t", "/n", "\t", "\n", "," };
        static public void readConfig(string txtfile)
        { 
            if(!File.Exists(txtfile))
                return;
            StreamReader sr = new StreamReader(File.OpenRead(txtfile));

            string[] values = { "" };
            string line = "";

            while (!sr.EndOfStream)
            {
                line = sr.ReadLine();
                //values = line.Split("=".ToArray(), StringSplitOptions.RemoveEmptyEntries);
                values = line.Split(seperator, StringSplitOptions.RemoveEmptyEntries);

                if (values[0] == "useCV")
                    useCV = Convert.ToBoolean(values[1]);
                else if (values[0] == "dbPath")
                    dbPath = values[1];
                else if (values[0] == "resultsPath")
                    resultsPath = values[1];
                else if (values[0] == "setInhyperCube")
                    setInhyperCube = Convert.ToBoolean(values[1]);
                else if (values[0] == "findMterms")
                    findMterms = Convert.ToBoolean(values[1]);
                else if (values[0] == "useParallel")
                    useParallel = Convert.ToBoolean(values[1]);
                else if (values[0] == "dbPath")
                    dbPath = values[1];
                else if (values[0] == "resultsPath")
                    resultsPath = values[1];
                else if (values[0] == "approxThresh")
                    approxThresh = Convert.ToDouble(values[1]);
                else if (values[0] == "minNodeSize")
                    minNodeSize = Convert.ToInt32(values[1]);
                else if (values[0] == "partitionType")
                    partitionType = values[1];
                else if (values[0] == "boundLevelDepth")
                    boundLevelDepth = Convert.ToInt32(values[1]);
                else if (values[0] == "nTrees")
                    nTrees = Convert.ToInt32(values[1]);
                else if (values[0] == "nFeaturesStr")
                    nFeaturesStr = values[1];
                else if (values[0] == "bagginPercent")
                    bagginPercent = Convert.ToDouble(values[1]);
                else if (values[0] == "saveTrees")
                    saveTrees = Convert.ToBoolean(values[1]);
                else if (values[0] == "testRF")
                    testRF = Convert.ToBoolean(values[1]);
                else if (values[0] == "evaluateSmoothness")
                    evaluateSmoothness = Convert.ToBoolean(values[1]);
                else if (values[0] == "testWf")
                    testWf = Convert.ToBoolean(values[1]);
                else if (values[0] == "errTypeTest")
                    errTypeTest = values[1];
                else if (values[0] == "fixThreshold")
                    fixThreshold = Convert.ToDouble(values[1]);
                else if (values[0] == "hopping")
                    hopping = Convert.ToInt32(values[1]);
                else if (values[0] == "m_terms")
                    m_terms = Convert.ToInt32(values[1]);
                else if (values[0] == "setClassification")
                    setClassification = Convert.ToBoolean(values[1]);
                else if (values[0] == "nonLinearHopping")
                    nonLinearHopping = Convert.ToBoolean(values[1]);
                else if (values[0] == "nCv")
                    nCv = Convert.ToInt32(values[1]);
                else if (values[0] == "useContNorms")
                    useContNorms = Convert.ToBoolean(values[1]);
                
            }
            sr.Close();
        }

        //public void printConfig(string fileName, string outFile)
        //{ 
        //    StreamWriter sw;
        //    if(outFile == null)
        //    {
        //        string dir = Path.GetDirectoryName(fileName);
        //        if (!System.IO.Directory.Exists(dir))
        //            System.IO.Directory.CreateDirectory(dir);

        //        sw = new StreamWriter(fileName, false);
        //    }
        //    else
        //        sw = new StreamWriter(outFile.OpenWrite());
            
        //    sw.WriteLine("useCV" + "=" + useCV);
        //    sw.WriteLine("croosValidTB" + "=" + croosValidTB);
        //    sw.WriteLine("setInhyperCube" + "=" + setInhyperCube);
        //    sw.WriteLine("useRF" + "=" + useRF);
        //    sw.WriteLine("useParallel" + "=" + useParallel);
        //    sw.WriteLine("dbPath" + "=" + dbPath);
        //    sw.WriteLine("resultsPath" + "=" + resultsPath);
        //    sw.WriteLine("approxThresh" + "=" + approxThresh);
        //    sw.WriteLine("minNodeSize" + "=" + minNodeSize);
        //    sw.WriteLine("partitionType" + "=" + partitionType);
        //    sw.WriteLine("splitTypeTB" + "=" + splitTypeTB);
        //    sw.WriteLine("boundLevelDepth" + "=" + boundLevelDepth);
        //    sw.WriteLine("errTypeTest" + "=" + errTypeTest);
        //    sw.WriteLine("trainingPercentTB" + "=" + trainingPercentTB);
        //    sw.WriteLine("nTrees" + "=" + nTrees);
        //    sw.WriteLine("Nfeatures" + "=" + Nfeatures);
        //    sw.WriteLine("bagginPercent" + "=" + bagginPercent);
        //    sw.WriteLine("saveTrees" + "=" + saveTrees);
        //    sw.WriteLine("runOneTreeCB" + "=" + runOneTreeCB);
        //    sw.WriteLine("testRF" + "=" + testRF);   
        //    sw.WriteLine("evaluateSmoothness" + "=" + evaluateSmoothness);
        //    sw.WriteLine("testWf" + "=" + testWf);
        //    sw.WriteLine("BaggingWithRepCB" + "=" + BaggingWithRepCB); 
        //    sw.WriteLine("boundDepthTB" + "=" + boundDepthTB);
        //    sw.WriteLine("fixThreshold" + "=" + fixThreshold);
        //    sw.WriteLine("thresholdWaveletsTB" + "=" + thresholdWaveletsTB); 
        //    sw.WriteLine("setClassification" + "=" + setClassification);

        //    sw.Close();
        //}

        static public bool useCV;
        static public bool setInhyperCube;
        static public bool findMterms;
        static public bool useParallel;
        static public string dbPath;
        static public string resultsPath;
        static public double approxThresh;
        static public int minNodeSize;
        static public string partitionType;
        static public int boundLevelDepth;
        static public string errTypeTest;
        static public int nTrees;
        static public int nFeatures;
        static public double bagginPercent;
        static public bool saveTrees;
        static public bool testRF;
        static public bool evaluateSmoothness;
        static public bool testWf;
        static public double fixThreshold;
        static public int hopping;
        static public int m_terms;
        static public bool setClassification;
        static public bool nonLinearHopping;
        //static public bool rumPrallel;      
        static public bool runRFPrunning;
        static public string MainFolderName;
        static public StreamWriter logger = new StreamWriter("log.txt", false);
        static public int nCv;
        static public string nFeaturesStr;
        static public bool useContNorms;
    }
 }
 
 
 
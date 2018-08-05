using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

using System.Threading;

namespace DataSetsSparsity
{
    public class userConfigGUI
    {
        public void readConfig(string txtfile)
        { 
            if(!File.Exists(txtfile))
                return;
            StreamReader sr = new StreamReader(File.OpenRead(txtfile));

            string[] seperator = { ";", "/t", "/n", "\t", "\n", "," };

            string[] values = { "" };
            string line = "";

            while (!sr.EndOfStream)
            {
                line = sr.ReadLine();
                //values = line.Split(",".ToArray(), StringSplitOptions.RemoveEmptyEntries);
                values = line.Split(seperator, StringSplitOptions.RemoveEmptyEntries);
                
                if (values[0] == "useCV")
                    useCV = values[1];
                if (values[0] == "useParallel")
                    useParallel = values[1];
                else if (values[0] == "nCv")
                    nCv = values[1];
                else if (values[0] == "setInhyperCube")
                    setInhyperCube = values[1];
                else if (values[0] == "nonLinearHopping")
                    nonLinearHopping = values[1];
                else if (values[0] == "dbPath")
                    dbPath = values[1];
                else if (values[0] == "resultsPath")
                    resultsPath = values[1];
                else if (values[0] == "approxThresh")
                    approxThresh = values[1];
                else if (values[0] == "minNodeSize")
                    minNodeSize = values[1];
                else if (values[0] == "partitionType")
                    partitionType = values[1];
                else if (values[0] == "useContNorms")
                    useContNorms = values[1];
                else if (values[0] == "boundLevelDepth")
                    boundLevelDepth = values[1];
                else if (values[0] == "nTrees")
                    nTrees = values[1];
                else if (values[0] == "nFeaturesStr")
                    nFeaturesStr = values[1];
                else if (values[0] == "bagginPercent")
                    bagginPercent = values[1];
                else if (values[0] == "saveTrees")
                    saveTrees = values[1];
                else if (values[0] == "testRF")
                    testRF = values[1];
                else if (values[0] == "evaluateSmoothness")
                    evaluateSmoothness = values[1];
                else if (values[0] == "testWf")
                    testWf = values[1];
                //else if (values[0] == "BaggingWithRepCB")
                //    BaggingWithRepCB = values[1];
                else if (values[0] == "errTypeTest")
                    errTypeTest = values[1];
                //else if (values[0] == "boundDepthTB")
                //    boundDepthTB = values[1];
                //else if (values[0] == "thresholdWaveletsCB")
                //    thresholdWaveletsCB = values[1];
                else if (values[0] == "fixThreshold")
                    fixThreshold = values[1];
                else if (values[0] == "hopping")
                    hopping = values[1];
                else if (values[0] == "m_terms")
                    m_terms = values[1];
                else if (values[0] == "setClassification")
                    setClassification = values[1];

            }
            sr.Close();
        }
        public void printConfig(string fileName, string outFile)
        { 
            StreamWriter sw;
            if(outFile == null)
            {
                string dir = Path.GetDirectoryName(fileName);
                if (!System.IO.Directory.Exists(dir))
                    System.IO.Directory.CreateDirectory(dir);

                sw = new StreamWriter(fileName, false);
            }
            else
                sw = new StreamWriter(outFile);
            
            sw.WriteLine("useCV" + "," + useCV);
            sw.WriteLine("useParallel" + "," + useParallel);            
            sw.WriteLine("nCv" + "," + nCv);
            sw.WriteLine("setInhyperCube" + "," + setInhyperCube);
            sw.WriteLine("nonLinearHopping" + "," + nonLinearHopping);
            sw.WriteLine("dbPath" + "," + dbPath);
            sw.WriteLine("resultsPath" + "," + resultsPath);
            sw.WriteLine("approxThresh" + "," + approxThresh);
            sw.WriteLine("minNodeSize" + "," + minNodeSize);
            sw.WriteLine("partitionType" + "," + partitionType);
            sw.WriteLine("useContNorms" + "," + useContNorms);
            sw.WriteLine("boundLevelDepth" + "," + boundLevelDepth);
            sw.WriteLine("errTypeTest" + "," + errTypeTest);
            sw.WriteLine("trainingPercentTB" + "," + trainingPercentTB);
            sw.WriteLine("nTrees" + "," + nTrees);
            sw.WriteLine("nFeaturesStr" + "," + nFeaturesStr);
            sw.WriteLine("bagginPercent" + "," + bagginPercent);
            sw.WriteLine("saveTrees" + "," + saveTrees);
            sw.WriteLine("runOneTreeCB" + "," + runOneTreeCB);
            sw.WriteLine("testRF" + "," + testRF);   
            sw.WriteLine("evaluateSmoothness" + "," + evaluateSmoothness);
            sw.WriteLine("testWf" + "," + testWf);
            //sw.WriteLine("BaggingWithRepCB" + "," + BaggingWithRepCB); 
            //sw.WriteLine("boundDepthTB" + "," + boundDepthTB);
            //sw.WriteLine("thresholdWaveletsCB" + "," + thresholdWaveletsCB);
            sw.WriteLine("fixThreshold" + "," + fixThreshold);
            sw.WriteLine("hopping" + "," + hopping);
            sw.WriteLine("m_terms" + "," + m_terms);
            sw.WriteLine("setClassification" + "," + setClassification);

            sw.Close();
        }

        public string useCV;
        public string useParallel;
        public string nCv;
        public string setInhyperCube;
        public string nonLinearHopping;
        public string dbPath;
        public string resultsPath;
        public string approxThresh;
        public string minNodeSize;
        public string partitionType;
        public string useContNorms;
        public string boundLevelDepth;
        public string errTypeTest;
        public string trainingPercentTB;
        public string nTrees;
        public string nFeaturesStr;
        public string bagginPercent;
        public string saveTrees;
        public string runOneTreeCB;
        public string testRF;
        public string evaluateSmoothness;
        public string testWf;
        public string BaggingWithRepCB;
        public string boundDepthTB;   
        //public string thresholdWaveletsCB;
        public string fixThreshold;
        public string hopping;
        public string m_terms;
        public string setClassification; 
     }
 }
 
 
 
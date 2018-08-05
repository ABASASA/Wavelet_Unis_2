using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

using System.Threading;
using Amazon;
using Amazon.Runtime;
using Amazon.S3;
using Amazon.S3.Model;
using Amazon.S3.IO;

namespace DataScienceAnalysis
{
    public class userConfig
    {
        public void readConfig(string txtfile)
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
                values = line.Split("=".ToArray(), StringSplitOptions.None);

                if (values[0] == "croosValidCB")
                    croosValidCB = values[1];
                else if (values[0] == "croosValidTB")
                    croosValidTB = values[1];
                else if (values[0] == "usePCA")
                    usePCA = values[1];
                else if (values[0] == "runRFProoningCB")
                    runRFProoningCB = values[1];
                else if (values[0] == "runBoostingLearningRateCB")
                    runBoostingLearningRateCB = values[1];
                else if (values[0] == "runBoostingProoningCB")
                    runBoostingProoningCB = values[1];
                else if (values[0] == "runProoningCB")
                    runProoningCB = values[1];
                else if (values[0] == "runRfCB")
                    runRfCB = values[1];
                else if (values[0] == "runBoostingCB")
                    runBoostingCB = values[1];
                else if (values[0] == "rumPrallelCB")
                    rumPrallelCB = values[1];
                else if (values[0] == "UseS3CB")
                    UseS3CB = values[1];
                else if (values[0] == "bucketTB")
                    bucketTB = values[1];
                else if (values[0] == "DBTB")
                    DBTB = values[1];
                else if (values[0] == "ResultsTB")
                    ResultsTB = values[1];
                else if (values[0] == "NboostTB")
                    NboostTB = values[1];
                else if (values[0] == "boostingLamda0TB")
                    boostingLamda0TB = values[1];
                else if (values[0] == "NfirstwaveletsBoostingTB")
                    NfirstwaveletsBoostingTB = values[1];
                else if (values[0] == "NfirstPruninginBoostingTB")
                    NfirstPruninginBoostingTB = values[1];
                else if (values[0] == "NboostingLearningRateTB")
                    NboostingLearningRateTB = values[1];
                else if (values[0] == "boostingKfuncPercentTB")
                    boostingKfuncPercentTB = values[1];
                else if (values[0] == "NfeaturesTB")
                    NfeaturesTB = values[1];
                else if (values[0] == "approxThreshTB")
                    approxThreshTB = values[1];
                else if (values[0] == "minNodeSizeTB")
                    minNodeSizeTB = values[1];
                else if (values[0] == "partitionTypeTB")
                    partitionTypeTB = values[1];
                else if (values[0] == "splitTypeTB")
                    splitTypeTB = values[1];
                else if (values[0] == "boundLevelTB")
                    boundLevelTB = values[1];
                else if (values[0] == "pruningEstimationRange0TB")
                    pruningEstimationRange0TB = values[1];
                else if (values[0] == "waveletsEstimationRange0TB")
                    waveletsEstimationRange0TB = values[1];
                else if (values[0] == "errTypeEstimationTB")
                    errTypeEstimationTB = values[1];
                else if (values[0] == "waveletsSkipEstimationTB")
                    waveletsSkipEstimationTB = values[1];
                else if (values[0] == "waveletsPercentEstimationTB")
                    waveletsPercentEstimationTB = values[1];
                else if (values[0] == "trainingPercentTB")
                    trainingPercentTB = values[1];
                else if (values[0] == "NloopsTB")
                    NloopsTB = values[1];
                else if (values[0] == "pruningEstimationRange1TB")
                    pruningEstimationRange1TB = values[1];
                else if (values[0] == "waveletsEstimationRange1TB")
                    waveletsEstimationRange1TB = values[1];
                else if (values[0] == "RFpruningEstimationRange1TB")
                    RFpruningEstimationRange1TB = values[1];
                else if (values[0] == "NrfTB")
                    NrfTB = values[1];
                else if (values[0] == "RFwaveletsEstimationRange1TB")
                    RFwaveletsEstimationRange1TB = values[1];
                else if (values[0] == "RFpruningEstimationRange0TB")
                    RFpruningEstimationRange0TB = values[1];
                else if (values[0] == "NfeaturesrfTB")
                    NfeaturesrfTB = values[1];
                else if (values[0] == "RFwaveletsEstimationRange0TB")
                    RFwaveletsEstimationRange0TB = values[1];
                else if (values[0] == "bagginPercentTB")
                    bagginPercentTB = values[1];
                else if (values[0] == "saveTressCB")
                    saveTressCB = values[1];
                else if (values[0] == "runOneTreeCB")
                    runOneTreeCB = values[1];
                else if (values[0] == "estimateRFonTrainingCB")
                    estimateRFonTrainingCB = values[1];
                else if (values[0] == "runOneTreeOnTtrainingCB")
                    runOneTreeOnTtrainingCB = values[1];
                else if (values[0] == "estimateRFnoVotingCB")
                    estimateRFnoVotingCB = values[1];
                else if (values[0] == "estimateRFwaveletsCB")
                    estimateRFwaveletsCB = values[1];
                else if (values[0] == "BaggingWithRepCB")
                    BaggingWithRepCB = values[1]; 
                else if (values[0] == "sparseRfCB")
                    sparseRfCB = values[1];
                else if (values[0] == "sparseRfTB")
                    sparseRfTB = values[1];
                else if (values[0] == "boundDepthTB")
                    boundDepthTB = values[1];

            }
            sr.Close();
        }
        public void printConfig(string fileName, S3FileInfo outFile)
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
                sw = new StreamWriter(outFile.OpenWrite());
            
            sw.WriteLine("croosValidCB" + "=" + croosValidCB);
            sw.WriteLine("croosValidTB" + "=" + croosValidTB);
            sw.WriteLine("usePCA" + "=" + usePCA);
            sw.WriteLine("runRFProoningCB" + "=" + runRFProoningCB);
            sw.WriteLine("runBoostingLearningRateCB" + "=" + runBoostingLearningRateCB);
            sw.WriteLine("runBoostingProoningCB" + "=" + runBoostingProoningCB);
            sw.WriteLine("runProoningCB" + "=" + runProoningCB);
            sw.WriteLine("runRfCB" + "=" + runRfCB);
            sw.WriteLine("runBoostingCB" + "=" + runBoostingCB);
            sw.WriteLine("rumPrallelCB" + "=" + rumPrallelCB);
            sw.WriteLine("UseS3CB" + "=" + UseS3CB);
            sw.WriteLine("bucketTB" + "=" + bucketTB);
            sw.WriteLine("DBTB" + "=" + DBTB);
            sw.WriteLine("ResultsTB" + "=" + ResultsTB);
            sw.WriteLine("NboostTB" + "=" + NboostTB);
            sw.WriteLine("boostingLamda0TB" + "=" + boostingLamda0TB);
            sw.WriteLine("NfirstwaveletsBoostingTB" + "=" + NfirstwaveletsBoostingTB);
            sw.WriteLine("NfirstPruninginBoostingTB" + "=" + NfirstPruninginBoostingTB);
            sw.WriteLine("NboostingLearningRateTB" + "=" + NboostingLearningRateTB);
            sw.WriteLine("boostingKfuncPercentTB" + "=" + boostingKfuncPercentTB);
            sw.WriteLine("NfeaturesTB" + "=" + NfeaturesTB);
            sw.WriteLine("approxThreshTB" + "=" + approxThreshTB);
            sw.WriteLine("minNodeSizeTB" + "=" + minNodeSizeTB);
            sw.WriteLine("partitionTypeTB" + "=" + partitionTypeTB);
            sw.WriteLine("splitTypeTB" + "=" + splitTypeTB);
            sw.WriteLine("boundLevelTB" + "=" + boundLevelTB);
            sw.WriteLine("pruningEstimationRange0TB" + "=" + pruningEstimationRange0TB);
            sw.WriteLine("waveletsEstimationRange0TB" + "=" + waveletsEstimationRange0TB);
            sw.WriteLine("errTypeEstimationTB" + "=" + errTypeEstimationTB);
            sw.WriteLine("waveletsSkipEstimationTB" + "=" + waveletsSkipEstimationTB);
            sw.WriteLine("waveletsPercentEstimationTB" + "=" + waveletsPercentEstimationTB);
            sw.WriteLine("trainingPercentTB" + "=" + trainingPercentTB);
            sw.WriteLine("NloopsTB" + "=" + NloopsTB);
            sw.WriteLine("pruningEstimationRange1TB" + "=" + pruningEstimationRange1TB);
            sw.WriteLine("waveletsEstimationRange1TB" + "=" + waveletsEstimationRange1TB);
            sw.WriteLine("RFpruningEstimationRange1TB" + "=" + RFpruningEstimationRange1TB);
            sw.WriteLine("NrfTB" + "=" + NrfTB);
            sw.WriteLine("RFwaveletsEstimationRange1TB" + "=" + RFwaveletsEstimationRange1TB);
            sw.WriteLine("RFpruningEstimationRange0TB" + "=" + RFpruningEstimationRange0TB);
            sw.WriteLine("NfeaturesrfTB" + "=" + NfeaturesrfTB);
            sw.WriteLine("RFwaveletsEstimationRange0TB" + "=" + RFwaveletsEstimationRange0TB);
            sw.WriteLine("bagginPercentTB" + "=" + bagginPercentTB);
            sw.WriteLine("saveTressCB" + "=" + saveTressCB);
            sw.WriteLine("runOneTreeCB" + "=" + runOneTreeCB);
            sw.WriteLine("estimateRFonTrainingCB" + "=" + estimateRFonTrainingCB);   
            sw.WriteLine("runOneTreeOnTtrainingCB" + "=" + runOneTreeOnTtrainingCB);
            sw.WriteLine("estimateRFnoVotingCB" + "=" + estimateRFnoVotingCB);
            sw.WriteLine("estimateRFwaveletsCB" + "=" + estimateRFwaveletsCB);
            sw.WriteLine("BaggingWithRepCB" + "=" + BaggingWithRepCB); 
            sw.WriteLine("sparseRfCB" + "=" + sparseRfCB);
            sw.WriteLine("sparseRfTB" + "=" + sparseRfTB); 
            sw.WriteLine("boundDepthTB" + "=" + boundDepthTB);
            sw.Close();
        }

        public string croosValidCB;
        public string croosValidTB;
        public string usePCA;
        public string runRFProoningCB;
        public string runBoostingLearningRateCB;
        public string runBoostingProoningCB;
        public string runProoningCB;
        public string runRfCB;
        public string runBoostingCB;
        public string rumPrallelCB;
        public string UseS3CB;
        public string bucketTB;
        public string DBTB;
        public string ResultsTB;
        public string NboostTB;
        public string boostingLamda0TB;
        public string NfirstwaveletsBoostingTB;
        public string NfirstPruninginBoostingTB;
        public string NboostingLearningRateTB;
        public string boostingKfuncPercentTB;
        public string NfeaturesTB;
        public string approxThreshTB;
        public string minNodeSizeTB;
        public string partitionTypeTB;
        public string splitTypeTB;
        public string boundLevelTB;
        public string pruningEstimationRange0TB;
        public string waveletsEstimationRange0TB;
        public string errTypeEstimationTB;
        public string waveletsSkipEstimationTB;
        public string waveletsPercentEstimationTB;
        public string trainingPercentTB;
        public string NloopsTB;
        public string pruningEstimationRange1TB;
        public string waveletsEstimationRange1TB;
        public string RFpruningEstimationRange1TB;
        public string NrfTB;
        public string RFwaveletsEstimationRange1TB;
        public string RFpruningEstimationRange0TB;
        public string NfeaturesrfTB;
        public string RFwaveletsEstimationRange0TB;
        public string bagginPercentTB;
        public string saveTressCB;
        public string runOneTreeCB;
        public string estimateRFonTrainingCB;
        public string runOneTreeOnTtrainingCB;
        public string estimateRFnoVotingCB;
        public string estimateRFwaveletsCB;
        public string BaggingWithRepCB;
        public string sparseRfCB;
        public string sparseRfTB;
        public string boundDepthTB;   
     }
 }
 
 
 
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.IO;
using System.Threading.Tasks;
using System.Threading;
using wf;

namespace DataSetsSparsity
{
    public partial class Form1 : Form
    {
        //CONSTRUCTOR
        public Form1()
        {
            InitializeComponent();

            //READ AND SET PROPERTIES
            u_config.readConfig(@"..\..\..\..\wf\bin\Debug\config.txt");
            setfromConfig();
        }

        //PARAMS
        public static double[][] boundingBox;
        public static List<List<double>> MainGrid;
        public static string MainFolderName; //THE DIR OF THE ROOT FOLDER
        public static string[] seperator = { " ", ";", "/t", "/n", "," };
        //public static bool runRf;
        //public static bool runProoning;
        public static bool runRFPrunning;
        public static userConfigGUI u_config = new userConfigGUI();

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

        private static void printHeaderLine(StreamWriter sw,int dataDim, int labelDim)
        {
            string line = "id,waveletNorm,";
            for (int j = 0; j < dataDim; j++)
            {
                line += "beginDim" + j.ToString() + ",endDim" + j.ToString()+",";
            }

            for (int j = 0; j < labelDim; j++)
            {
                line += "val" + j.ToString() + ",";
            }
            line += "parentVal,volume";
            sw.WriteLine(line);
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

        
        private void btnScript_Click(object sender, EventArgs e)
        {
            set2Config();
            string confFile = @"..\..\..\..\wf\bin\Debug\config.txt";
            u_config.printConfig(confFile, null);
            wf.Program.remoteRun(confFile);
            btnScript.BackColor = Color.Green;
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

        private void setfromConfig()
        {
            if (u_config.useCV == "true")
                useCV.Checked = true;
            if (u_config.useParallel == "true")
                useParallel.Checked = true;
            
            //if (u_config.thresholdWaveletsCB == "1")
            //    thresholdWaveletsCB.Checked = true;
            if (u_config.setClassification == "true")
                setClassification.Checked = true;
            if (u_config.setInhyperCube == "true")
                setInhyperCube.Checked = true;
            if (u_config.nonLinearHopping == "true")
                useRF.Checked = true;
            if (u_config.saveTrees == "true")
                saveTrees.Checked = true;
            if (u_config.testRF == "true")
                testRF.Checked = true;
            if (u_config.evaluateSmoothness == "true")
                evaluateSmoothness.Checked = true;
            if (u_config.testWf == "true")
                testWf.Checked = true;
            //if (u_config.BaggingWithRepCB == "1")
            //    setClassification.Checked = true;
            nCv.Text = u_config.nCv;
            dbPath.Text = u_config.dbPath;
            resultsPath.Text = u_config.resultsPath;
            approxThresh.Text = u_config.approxThresh;
            minNodeSize.Text = u_config.minNodeSize;
            partitionType.Text = u_config.partitionType;
            useContNorms.Text = u_config.useContNorms;
            boundLevelDepth.Text = u_config.boundLevelDepth;
            errTypeTest.Text = u_config.errTypeTest;
            nTrees.Text = u_config.nTrees;
            nFeaturesStr.Text = u_config.nFeaturesStr;
            bagginPercent.Text = u_config.bagginPercent;
            //boundDepthTB.Text = u_config.boundDepthTB;
            fixThreshold.Text = u_config.fixThreshold;
            hopping.Text = u_config.hopping;
            m_terms.Text = u_config.m_terms;
    }

        private void set2Config()
        {
            u_config.useCV = useCV.Checked ? "true" : "false";
            u_config.useParallel = useParallel.Checked ? "true" : "false";            
            u_config.nCv = nCv.Text;
            //u_config.thresholdWaveletsCB = thresholdWaveletsCB.Checked ? "1" : "0";
            u_config.fixThreshold = fixThreshold.Text;
            u_config.hopping = hopping.Text;
            u_config.m_terms = m_terms.Text;

            u_config.setInhyperCube = setInhyperCube.Checked ? "true" : "false";
            u_config.nonLinearHopping = useRF.Checked ? "true" : "false";
            u_config.testRF = testRF.Checked ? "true" : "false";
            u_config.dbPath = dbPath.Text;
            u_config.resultsPath = resultsPath.Text;
            u_config.approxThresh = approxThresh.Text;
            u_config.minNodeSize = minNodeSize.Text;
            u_config.partitionType = partitionType.Text;
            u_config.useContNorms = useContNorms.Text;
            u_config.boundLevelDepth = boundLevelDepth.Text;
            u_config.errTypeTest = errTypeTest.Text;
            u_config.nTrees = nTrees.Text;
            u_config.nFeaturesStr = nFeaturesStr.Text;
            u_config.bagginPercent = bagginPercent.Text;
            //u_config.boundDepthTB = boundDepthTB.Text;
            u_config.saveTrees = saveTrees.Checked ? "true" : "false";
            u_config.evaluateSmoothness = evaluateSmoothness.Checked ? "true" : "false";
            u_config.testWf = testWf.Checked ? "true" : "false";
            //u_config.BaggingWithRepCB = setClassification.Checked ? "1" : "0";
        }

        
        private static void regularDelegateFor(int begin, int size, Action<int> body)
        {
            for (int i = begin; i < size; i++)
            {
                body.Invoke(i);
            }
        }

        private void textBox2_TextChanged(object sender, EventArgs e)
        {

        }
    }
}

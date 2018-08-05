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

namespace DataScienceAnalysis
{
    public partial class Form1 : Form
    {
        //CONSTRUCTOR
        public Form1()
        {
            InitializeComponent();
        }

        #region parameters

        public string MainFolderName; //THE DIR OF THE ROOT FOLDER
        
        DataTable training_dt = new DataTable();        //original input training data (table)
        DataTable testing_dt = new DataTable();         //original input testing data (table)
        DataTable training_label_dt = new DataTable();  //original input training labels (table)
        DataTable testing_label_dt = new DataTable();   //original input testing labels (table)

        DataTable training_data_low_dim_dt = new DataTable();        //training data with lower dimantion (table)
        
        string method;                              // method of dim reduction
        PrincipalComponentCollection Components;    //in spectral method of dim reduction - eigenvectors in OO container Components[0].eig...
        double[] eigenvalues;                       // in spectral method of dim reduction - eigenvalues
        double[,] traininglData_lower_dim;          //training data after rotation
        int dimSize;                                //dimSize of larger PCA
        ManageInputData tmpMngInput;                //tmp in order to run delegated function

        //TRAINING MATRIX 

        string[] dataTable_names = { "trainingData", "trainingLabel", "testingData", "testingLabel" };

        string[] table_seperator = { " ", ";", "/t", "/n", ","};

        DialogResult dlgResult = new DialogResult();
        ManageInputData MngInput;

        #endregion
        
        //READ_DATA_INTO_DT
        private void read_data_into_dt(string data_table_name, string datafileName)
        {
            StreamReader reader = new StreamReader(File.OpenRead(datafileName));
            
            //GET THE FIRST LINE 
            string line = reader.ReadLine();
            string[] values = line.Split(table_seperator, StringSplitOptions.RemoveEmptyEntries);

            //IF NO VALUES ALERT
            if (values.Count() < 1)
                MessageBox.Show("the file " + datafileName + " is probably empty");

            //SET DT BY ITS NAME
            DataTable dt;
            if (data_table_name == dataTable_names[0])
                dt = training_dt;
            else if (data_table_name == dataTable_names[1])
                dt = training_label_dt;
            else if (data_table_name == dataTable_names[2])
                dt = testing_dt;
            else if (data_table_name == dataTable_names[3])
                dt = testing_label_dt;
            else
            {
                dt = new DataTable();
                MessageBox.Show("should Not get here");
            } 
            
            
            //if data table already exists - create new 
            if (dt.Columns.Count > 0)
                dt = new DataTable();
            
            //SET COLUMNS TO TRAINING_DT with names (0,1,2,...)
            for (int i = 0; i < values.Count(); i++)
                dt.Columns.Add(i.ToString());

            //ADD FIRST DATA LINE
            dt.Rows.Add(values);

            //SET VALUES TO TABLE
            while (!reader.EndOfStream)
            {
                line = reader.ReadLine();
                values = line.Split(table_seperator, StringSplitOptions.RemoveEmptyEntries);

                //add first data line
                dt.Rows.Add(values);
            }        
        }

        //READ_DATA OF ALL TABLES
        private void read_data(ManageInputData MngInput)
        {
            read_data_into_dt(dataTable_names[0], MngInput.training_data_file_name);
            read_data_into_dt(dataTable_names[1], MngInput.training_label_file_name);
            read_data_into_dt(dataTable_names[2], MngInput.testing_data_file_name);
            read_data_into_dt(dataTable_names[3], MngInput.testing_label_file_name);        
        }
        
        //USE PROGRESS BAR TO /READ_DATA
        private void read_data_DoWork(ProgressForm sender, DoWorkEventArgs e)
        {
            //get the provided argument as usual
            object myArgument = e.Argument;

            read_data(tmpMngInput);

            for (int i = 0; i < 100; i++)
            {
                //notify progress to the form
                sender.SetProgress(i, "Step " + i.ToString() + " / 100...");

                //...

                //check if the user clicked cancel
                if (sender.CancellationPending)
                {
                    e.Cancel = true;
                    return;
                }
            }
        }

        private void setInputFiles()
        {

            tmpMngInput = MngInput;

            //USE PROGRESS BAR TO READ_DATA
            ProgressForm form = new ProgressForm();
            form.DoWork += new ProgressForm.DoWorkEventHandler(read_data_DoWork);
            DialogResult result = form.ShowDialog();

            if (MngInput.tabel_are_set == true)
            {
                btnReduceDim.Enabled = true;
                btnCreateDecisionTree.Enabled = true;
                btnLoadDecisionTree.Enabled = true;
            }        
        }
        
        //GET INPUT FILE NAMES
        private void btnInputFiles_Click(object sender, EventArgs e)
        {
            //USE FORM TO GET FILENAMES
            MngInput = new ManageInputData();
            dlgResult  = MngInput.ShowDialog();

            setInputFiles();
        }

        private void SetDataLowerDim(DataTable outputData, double[,] inputData, int dimSize)
        {
            //IF DATA TABLE ALREADY EXISTS - CREATE NEW 
            if (outputData.Columns.Count > 0)
                outputData = new DataTable();

            //SET COLUMNS TO TRAINING_DT with names (0,1,2,...)
            for (int i = 0; i < dimSize; i++)
                outputData.Columns.Add(i.ToString());

            for (int i = 0; i < inputData.GetLength(0); i++) 
            {
                DataRow dr = outputData.NewRow(); 
                
                for (int j = 0; j < outputData.Columns.Count; j++)
                    dr[j] = inputData[i,j];
                outputData.Rows.Add(dr);
            }           
        }

        private void ReduceDim(reduceDimantion rdForm)
        {
            //IF WE REDUCED DIMANTION
            if (rdForm.method_calced == true)
            {
                //SET VALUES FROM FORM
                eigenvalues = rdForm.eigenvalues;
                traininglData_lower_dim = rdForm.finalData;
                method = rdForm.method;
                dimSize = rdForm.dimSize;
                Components = rdForm.Components;

                //set training data with lower dimantion
                SetDataLowerDim(training_data_low_dim_dt, traininglData_lower_dim, dimSize);
            }            
        }
        
        //REDUCE DIMANTION
        private void btnReduceDim_Click(object sender, EventArgs e)
        {
            //USE FOR (REDUCEDIMANTION) TOREDUCE DIM
            reduceDimantion rdForm = new reduceDimantion(training_dt);
            rdForm.ShowDialog();

            ReduceDim(rdForm);

        }

        //CREATE NEW
        private void recordToolStripMenuItem_Click(object sender, EventArgs e)
        {
            folderBrowserDialog1.Description = "Select a folder to place your new Record";
            folderBrowserDialog1.SelectedPath = "D:\\Phd\\Shai\\code\\DataScienceAnalysis\\ada_test";
            DialogResult dlg = folderBrowserDialog1.ShowDialog();
            MainFolderName = folderBrowserDialog1.SelectedPath;
            btnInputFiles.Enabled = true;
        }

        //OPEN
        private void recordToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            folderBrowserDialog1.Description = "Select a Record folder";
            folderBrowserDialog1.SelectedPath = "D:\\Phd\\Shai\\code\\DataScienceAnalysis\\ada_test";
            DialogResult dlg = folderBrowserDialog1.ShowDialog();
            MainFolderName = folderBrowserDialog1.SelectedPath;
            btnInputFiles.Enabled = true;
        }

        public static void setdataRow(DataRow dr, double[] arr)
        {
            for (int i = 0; i < arr.Count(); i++)
            {
                arr[i] = double.Parse(dr[i].ToString());
            }
        }

        //SRC IS GREATER OR EQUAL TO DST
        public static void setMatrix(double[,] dst, double[,] src)
        {
            for (int i = 0; i < dst.GetLength(0); i++)
            {
                for (int j = 0; j < dst.GetLength(1); j++)
                {
                    dst[i, j] = src[i, j];
                }
            }
        }
     
        public void CreateDecisionTree()
        {
            if (training_data_low_dim_dt.Columns.Count < 1)
                MessageBox.Show("bad input - empty training data");
            else if (training_label_dt.Columns.Count < 1)
                MessageBox.Show("bad input - empty training labels");
            else
            {
                DecisionTreeForm decision_tree_form = new DecisionTreeForm(training_data_low_dim_dt, training_label_dt);
                decision_tree_form.ShowDialog();
            }        
        }

        private void btnCreateDecisionTree_Click(object sender, EventArgs e)
        {
            CreateDecisionTree();
        }

        private void btnScript_Click(object sender, EventArgs e)
        {
            if (cbScript.Text == "Script A")
            {
                MainFolderName = "D:\\Phd\\Shai\\code\\DataScienceAnalysis\\ada_test";

                //USE FORM TO GET FILENAMES
                MngInput = new ManageInputData();
                MngInput.SetAndReadFiles();
                setInputFiles();

                
                reduceDimantion rdForm = new reduceDimantion(training_dt);
                rdForm.DimCalc();
                rdForm.SetDimSize();
                ReduceDim(rdForm);

                
                if (training_data_low_dim_dt.Columns.Count < 1)
                    MessageBox.Show("bad input - empty training data");
                else if (training_label_dt.Columns.Count < 1)
                    MessageBox.Show("bad input - empty training labels");
                else
                {
                    DecisionTreeForm decision_tree_form = new DecisionTreeForm(training_data_low_dim_dt, training_label_dt);
                    decision_tree_form.Run();
                }

            }
        }
    }
}


//while (rdForm.method_calced == false)
//{
//    System.Threading.Thread.Sleep(250); // pause for 250 mili seconds;
//};
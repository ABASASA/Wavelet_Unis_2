using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using Accord.Statistics.Analysis;

namespace DataScienceAnalysis
{
    public partial class reduceDimantion : Form
    {
        public reduceDimantion(DataTable dt_input)
        {
            InitializeComponent();
            dt = dt_input;
            method_calced = false;
        }

        #region parameters
        
        DataTable dt;
        public double[] eigenvalues;
        public double[,] finalData;
        public string method;
        public bool method_calced;
        public int dimSize;
        public PrincipalComponentCollection Components;
        
        #endregion

        //SET MATRIX FROM DATATABLE
        private void SetMatrixFromDataTable(double[,] matrix, DataTable dt)
        {
            //SANITY CHECK
            if (dt.Rows.Count < 1 || dt.Columns.Count < 1)
                MessageBox.Show("data table " + dt.ToString() + " is empty ...");

            //SET VALUES
            for (int i = 0; i < dt.Rows.Count; i++)
            {
                for (int j = 0; j < dt.Columns.Count; j++)
                {
                    matrix[i, j] = double.Parse(dt.Rows[i][j].ToString());
                }
            }
        }

        //DO PCA
        private void PCA_DATA()
        {
            //GET DATA MATRIX IN THE FORMATE ACCORD CAN USE (DOUBLE [,])
            double[,] training_matrix = new double[dt.Rows.Count, dt.Columns.Count]; ;
            SetMatrixFromDataTable(training_matrix, dt);

            //Create the Principal Component Analysis
            var pca = new PrincipalComponentAnalysis(training_matrix);

            //PCA COMPUTE
            pca.Compute();
            
            //SET DATA
            finalData = pca.Transform(training_matrix);
            eigenvalues = pca.Eigenvalues;
            Components= pca.Components;
        }

        //USE PROGRESS BAR TO DO PCA
        void PCA_DoWork(ProgressForm sender, DoWorkEventArgs e)
        {
            //GET THE PROVIDED ARGUMENT AS USUAL
            object myArgument = e.Argument;

            PCA_DATA();

            for (int i = 0; i < 100; i++)
            {
                //notify progress to the form
                sender.SetProgress(i, "Step " + i.ToString() + " / 100...");

                //check if the user clicked cancel
                if (sender.CancellationPending)
                {
                    e.Cancel = true;
                    return;
                }
            }
        }

        //SHOW CHART
        private void showChart()
        {       

            for (int i = 0; i < eigenvalues.Count(); i++)
                chart_eigvalues.Series["eigenValues"].Points.AddY(eigenvalues[i]);
            chart_eigvalues.Show();

            //chart_eigvalues.ChartAreas[0].AxisX.Interval = 1;

            //chart_eigvalues.Series["eigenValues"].ChartType = SeriesChartType.FastLine;

            //chart_eigvalues.ChartAreas[0].AxisY.ScaleView.MinSize = 0;
            //chart_eigvalues.ChartAreas[0].CursorY.Interval = 0.001;

            ////Chart chart = this.Chart;
            //AxisXY axis = new AxisXY(chart_eigvalues);

            //double[] y = new double[] {8, 3, 5, 2, 9};
            //Data data1 = new Data(axis, y);
            //data1.DataType = Data.DATA_TYPE_LINE;
            //data1.LineColor = Color.Blue;
        }

        public void DimCalc()
        {
            //IF PCA
            if (cbDimReductMethod.Text.ToLower() == "PCA".ToLower())
            {
                //USE PROGRESS BAR TO CALC PCA
                ProgressForm form = new ProgressForm();
                form.DoWork += new ProgressForm.DoWorkEventHandler(PCA_DoWork);
                DialogResult result = form.ShowDialog();
                method = cbDimReductMethod.Text;
                method_calced = true;
                btnSetDimSize.Enabled = true;
                showChart();

                //PCA_DATA();
            }        
        
        }
        
        //REDUCE DIM CLICK
        private void btnDimCalc_Click(object sender, EventArgs e)
        {
            DimCalc();
        }

        public void SetDimSize()
        {
            if (tbDimSize.Text == "")
                return;
            dimSize = Int32.Parse(tbDimSize.Text);

            if (method_calced == true)//if it was set to true (when we calced eig vec)
                method_calced = true;
            else
                method_calced = false;

            Close();        
        }
        
        //SET DIM AND CLOSE FORM DIALOG
        private void btnSetDimSize_Click(object sender, EventArgs e)
        {
            SetDimSize();
        }
    
    
    }
}

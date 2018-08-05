namespace DataScienceAnalysis
{
    partial class reduceDimantion
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea2 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.Legend legend2 = new System.Windows.Forms.DataVisualization.Charting.Legend();
            System.Windows.Forms.DataVisualization.Charting.Series series2 = new System.Windows.Forms.DataVisualization.Charting.Series();
            System.Windows.Forms.DataVisualization.Charting.Title title2 = new System.Windows.Forms.DataVisualization.Charting.Title();
            this.lblDimReductionMethod = new System.Windows.Forms.Label();
            this.chart_eigvalues = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.cbDimReductMethod = new System.Windows.Forms.ComboBox();
            this.lblSetDimSize = new System.Windows.Forms.Label();
            this.tbDimSize = new System.Windows.Forms.TextBox();
            this.btnSetDimSize = new System.Windows.Forms.Button();
            this.btnDimCalc = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.chart_eigvalues)).BeginInit();
            this.SuspendLayout();
            // 
            // lblDimReductionMethod
            // 
            this.lblDimReductionMethod.AutoSize = true;
            this.lblDimReductionMethod.Location = new System.Drawing.Point(34, 32);
            this.lblDimReductionMethod.Name = "lblDimReductionMethod";
            this.lblDimReductionMethod.Size = new System.Drawing.Size(110, 13);
            this.lblDimReductionMethod.TabIndex = 24;
            this.lblDimReductionMethod.Text = "Dim reduction method";
            // 
            // chart_eigvalues
            // 
            chartArea2.AlignmentOrientation = System.Windows.Forms.DataVisualization.Charting.AreaAlignmentOrientations.Horizontal;
            chartArea2.AxisX2.IsMarginVisible = false;
            chartArea2.Name = "ChartArea1";
            this.chart_eigvalues.ChartAreas.Add(chartArea2);
            this.chart_eigvalues.ImeMode = System.Windows.Forms.ImeMode.Off;
            legend2.Enabled = false;
            legend2.Name = "Legend1";
            this.chart_eigvalues.Legends.Add(legend2);
            this.chart_eigvalues.Location = new System.Drawing.Point(12, 134);
            this.chart_eigvalues.Name = "chart_eigvalues";
            series2.ChartArea = "ChartArea1";
            series2.ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.Stock;
            series2.IsVisibleInLegend = false;
            series2.IsXValueIndexed = true;
            series2.LabelBackColor = System.Drawing.Color.GreenYellow;
            series2.Legend = "Legend1";
            series2.Name = "eigenValues";
            series2.YAxisType = System.Windows.Forms.DataVisualization.Charting.AxisType.Secondary;
            series2.YValuesPerPoint = 4;
            this.chart_eigvalues.Series.Add(series2);
            this.chart_eigvalues.Size = new System.Drawing.Size(814, 262);
            this.chart_eigvalues.TabIndex = 23;
            this.chart_eigvalues.Text = "sorted eigvalues";
            title2.Name = "Spectral Analysis";
            this.chart_eigvalues.Titles.Add(title2);
            // 
            // cbDimReductMethod
            // 
            this.cbDimReductMethod.FormattingEnabled = true;
            this.cbDimReductMethod.Items.AddRange(new object[] {
            "PCA",
            "diffusion maps",
            "Non"});
            this.cbDimReductMethod.Location = new System.Drawing.Point(150, 32);
            this.cbDimReductMethod.Name = "cbDimReductMethod";
            this.cbDimReductMethod.Size = new System.Drawing.Size(113, 21);
            this.cbDimReductMethod.TabIndex = 22;
            this.cbDimReductMethod.Text = "PCA";
            // 
            // lblSetDimSize
            // 
            this.lblSetDimSize.AutoSize = true;
            this.lblSetDimSize.Location = new System.Drawing.Point(34, 74);
            this.lblSetDimSize.Name = "lblSetDimSize";
            this.lblSetDimSize.Size = new System.Drawing.Size(63, 13);
            this.lblSetDimSize.TabIndex = 21;
            this.lblSetDimSize.Text = "Set dim size";
            // 
            // tbDimSize
            // 
            this.tbDimSize.Location = new System.Drawing.Point(150, 71);
            this.tbDimSize.Name = "tbDimSize";
            this.tbDimSize.Size = new System.Drawing.Size(60, 20);
            this.tbDimSize.TabIndex = 20;
            this.tbDimSize.Text = "6";
            // 
            // btnSetDimSize
            // 
            this.btnSetDimSize.Enabled = false;
            this.btnSetDimSize.Location = new System.Drawing.Point(291, 69);
            this.btnSetDimSize.Name = "btnSetDimSize";
            this.btnSetDimSize.Size = new System.Drawing.Size(60, 23);
            this.btnSetDimSize.TabIndex = 25;
            this.btnSetDimSize.Text = "Set";
            this.btnSetDimSize.UseVisualStyleBackColor = true;
            this.btnSetDimSize.Click += new System.EventHandler(this.btnSetDimSize_Click);
            // 
            // btnDimCalc
            // 
            this.btnDimCalc.Location = new System.Drawing.Point(291, 32);
            this.btnDimCalc.Name = "btnDimCalc";
            this.btnDimCalc.Size = new System.Drawing.Size(107, 24);
            this.btnDimCalc.TabIndex = 26;
            this.btnDimCalc.Text = "Calculate";
            this.btnDimCalc.UseVisualStyleBackColor = true;
            this.btnDimCalc.Click += new System.EventHandler(this.btnDimCalc_Click);
            // 
            // reduceDimantion
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(838, 436);
            this.Controls.Add(this.btnDimCalc);
            this.Controls.Add(this.btnSetDimSize);
            this.Controls.Add(this.lblDimReductionMethod);
            this.Controls.Add(this.chart_eigvalues);
            this.Controls.Add(this.cbDimReductMethod);
            this.Controls.Add(this.lblSetDimSize);
            this.Controls.Add(this.tbDimSize);
            this.Name = "reduceDimantion";
            this.Text = "reduceDimantion";
            ((System.ComponentModel.ISupportInitialize)(this.chart_eigvalues)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label lblDimReductionMethod;
        private System.Windows.Forms.DataVisualization.Charting.Chart chart_eigvalues;
        private System.Windows.Forms.ComboBox cbDimReductMethod;
        private System.Windows.Forms.Label lblSetDimSize;
        private System.Windows.Forms.TextBox tbDimSize;
        private System.Windows.Forms.Button btnSetDimSize;
        private System.Windows.Forms.Button btnDimCalc;
    }
}
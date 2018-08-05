namespace DataScienceAnalysis
{
    partial class ManageInputData
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
            this.btnTrainingLabel = new System.Windows.Forms.Button();
            this.btnTestinglabel = new System.Windows.Forms.Button();
            this.btnTestingData = new System.Windows.Forms.Button();
            this.btnTrainingData = new System.Windows.Forms.Button();
            this.tb_training_data = new System.Windows.Forms.TextBox();
            this.lblSetAndReadFiles = new System.Windows.Forms.Button();
            this.tb_testing_data = new System.Windows.Forms.TextBox();
            this.tb_testing_label = new System.Windows.Forms.TextBox();
            this.tb_training_label = new System.Windows.Forms.TextBox();
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            this.SuspendLayout();
            // 
            // btnTrainingLabel
            // 
            this.btnTrainingLabel.Location = new System.Drawing.Point(32, 83);
            this.btnTrainingLabel.Name = "btnTrainingLabel";
            this.btnTrainingLabel.Size = new System.Drawing.Size(90, 43);
            this.btnTrainingLabel.TabIndex = 21;
            this.btnTrainingLabel.Text = "read training label file";
            this.btnTrainingLabel.UseVisualStyleBackColor = true;
            this.btnTrainingLabel.Click += new System.EventHandler(this.btnTrainingLabel_Click);
            // 
            // btnTestinglabel
            // 
            this.btnTestinglabel.Location = new System.Drawing.Point(32, 181);
            this.btnTestinglabel.Name = "btnTestinglabel";
            this.btnTestinglabel.Size = new System.Drawing.Size(90, 43);
            this.btnTestinglabel.TabIndex = 20;
            this.btnTestinglabel.Text = "read testing label file";
            this.btnTestinglabel.UseVisualStyleBackColor = true;
            this.btnTestinglabel.Click += new System.EventHandler(this.btnTestinglabel_Click);
            // 
            // btnTestingData
            // 
            this.btnTestingData.Location = new System.Drawing.Point(32, 132);
            this.btnTestingData.Name = "btnTestingData";
            this.btnTestingData.Size = new System.Drawing.Size(90, 43);
            this.btnTestingData.TabIndex = 19;
            this.btnTestingData.Text = "read testing data file";
            this.btnTestingData.UseVisualStyleBackColor = true;
            this.btnTestingData.Click += new System.EventHandler(this.btnTestingData_Click);
            // 
            // btnTrainingData
            // 
            this.btnTrainingData.Location = new System.Drawing.Point(32, 34);
            this.btnTrainingData.Name = "btnTrainingData";
            this.btnTrainingData.Size = new System.Drawing.Size(90, 43);
            this.btnTrainingData.TabIndex = 18;
            this.btnTrainingData.Text = "read training data file";
            this.btnTrainingData.UseVisualStyleBackColor = true;
            this.btnTrainingData.Click += new System.EventHandler(this.btnTrainingData_Click);
            // 
            // tb_training_data
            // 
            this.tb_training_data.Location = new System.Drawing.Point(145, 46);
            this.tb_training_data.Name = "tb_training_data";
            this.tb_training_data.Size = new System.Drawing.Size(258, 20);
            this.tb_training_data.TabIndex = 23;
            // 
            // lblSetAndReadFiles
            // 
            this.lblSetAndReadFiles.Location = new System.Drawing.Point(481, 69);
            this.lblSetAndReadFiles.Name = "lblSetAndReadFiles";
            this.lblSetAndReadFiles.Size = new System.Drawing.Size(90, 43);
            this.lblSetAndReadFiles.TabIndex = 24;
            this.lblSetAndReadFiles.Text = "Set Data Files";
            this.lblSetAndReadFiles.UseVisualStyleBackColor = true;
            this.lblSetAndReadFiles.Click += new System.EventHandler(this.lblSetAndReadFiles_Click);
            // 
            // tb_testing_data
            // 
            this.tb_testing_data.Location = new System.Drawing.Point(145, 144);
            this.tb_testing_data.Name = "tb_testing_data";
            this.tb_testing_data.Size = new System.Drawing.Size(258, 20);
            this.tb_testing_data.TabIndex = 25;
            // 
            // tb_testing_label
            // 
            this.tb_testing_label.Location = new System.Drawing.Point(145, 193);
            this.tb_testing_label.Name = "tb_testing_label";
            this.tb_testing_label.Size = new System.Drawing.Size(258, 20);
            this.tb_testing_label.TabIndex = 26;
            // 
            // tb_training_label
            // 
            this.tb_training_label.Location = new System.Drawing.Point(145, 92);
            this.tb_training_label.Name = "tb_training_label";
            this.tb_training_label.Size = new System.Drawing.Size(258, 20);
            this.tb_training_label.TabIndex = 28;
            // 
            // openFileDialog1
            // 
            this.openFileDialog1.FileName = "openFileDialog1";
            // 
            // ManageInputData
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(611, 350);
            this.Controls.Add(this.tb_training_label);
            this.Controls.Add(this.tb_testing_label);
            this.Controls.Add(this.tb_testing_data);
            this.Controls.Add(this.lblSetAndReadFiles);
            this.Controls.Add(this.tb_training_data);
            this.Controls.Add(this.btnTrainingLabel);
            this.Controls.Add(this.btnTestinglabel);
            this.Controls.Add(this.btnTestingData);
            this.Controls.Add(this.btnTrainingData);
            this.Name = "ManageInputData";
            this.Text = "ManageInputData";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button btnTrainingLabel;
        private System.Windows.Forms.Button btnTestinglabel;
        private System.Windows.Forms.Button btnTestingData;
        private System.Windows.Forms.Button btnTrainingData;
        private System.Windows.Forms.TextBox tb_training_data;
        private System.Windows.Forms.Button lblSetAndReadFiles;
        private System.Windows.Forms.TextBox tb_testing_data;
        private System.Windows.Forms.TextBox tb_testing_label;
        private System.Windows.Forms.TextBox tb_training_label;
        private System.Windows.Forms.OpenFileDialog openFileDialog1;
    }
}
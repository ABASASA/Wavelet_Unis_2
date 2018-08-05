namespace DataSetsSparsity
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        /// <summary>
        /// Clean up any resources being used.
        private System.ComponentModel.IContainer components = null;

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
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            this.folderBrowserDialog1 = new System.Windows.Forms.FolderBrowserDialog();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.label5 = new System.Windows.Forms.Label();
            this.hopping = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.m_terms = new System.Windows.Forms.TextBox();
            this.useParallel = new System.Windows.Forms.CheckBox();
            this.label3 = new System.Windows.Forms.Label();
            this.setClassification = new System.Windows.Forms.CheckBox();
            this.fixThreshold = new System.Windows.Forms.TextBox();
            this.testWf = new System.Windows.Forms.CheckBox();
            this.evaluateSmoothness = new System.Windows.Forms.CheckBox();
            this.testRF = new System.Windows.Forms.CheckBox();
            this.saveTrees = new System.Windows.Forms.CheckBox();
            this.useCV = new System.Windows.Forms.CheckBox();
            this.setInhyperCube = new System.Windows.Forms.CheckBox();
            this.nCv = new System.Windows.Forms.TextBox();
            this.useRF = new System.Windows.Forms.CheckBox();
            this.btnScript = new System.Windows.Forms.Button();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.label2 = new System.Windows.Forms.Label();
            this.dbPath = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.resultsPath = new System.Windows.Forms.TextBox();
            this.label10 = new System.Windows.Forms.Label();
            this.label7 = new System.Windows.Forms.Label();
            this.approxThresh = new System.Windows.Forms.TextBox();
            this.label8 = new System.Windows.Forms.Label();
            this.minNodeSize = new System.Windows.Forms.TextBox();
            this.label9 = new System.Windows.Forms.Label();
            this.partitionType = new System.Windows.Forms.TextBox();
            this.useContNorms = new System.Windows.Forms.TextBox();
            this.boundLevelDepth = new System.Windows.Forms.TextBox();
            this.label24 = new System.Windows.Forms.Label();
            this.label26 = new System.Windows.Forms.Label();
            this.groupBox4 = new System.Windows.Forms.GroupBox();
            this.nTrees = new System.Windows.Forms.TextBox();
            this.label19 = new System.Windows.Forms.Label();
            this.nFeaturesStr = new System.Windows.Forms.TextBox();
            this.label35 = new System.Windows.Forms.Label();
            this.bagginPercent = new System.Windows.Forms.TextBox();
            this.label23 = new System.Windows.Forms.Label();
            this.errTypeTest = new System.Windows.Forms.TextBox();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.groupBox4.SuspendLayout();
            this.SuspendLayout();
            // 
            // openFileDialog1
            // 
            this.openFileDialog1.FileName = "openFileDialog1";
            // 
            // groupBox1
            // 
            this.groupBox1.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.groupBox1.Controls.Add(this.label5);
            this.groupBox1.Controls.Add(this.hopping);
            this.groupBox1.Controls.Add(this.label4);
            this.groupBox1.Controls.Add(this.m_terms);
            this.groupBox1.Controls.Add(this.useParallel);
            this.groupBox1.Controls.Add(this.label3);
            this.groupBox1.Controls.Add(this.setClassification);
            this.groupBox1.Controls.Add(this.fixThreshold);
            this.groupBox1.Controls.Add(this.testWf);
            this.groupBox1.Controls.Add(this.evaluateSmoothness);
            this.groupBox1.Controls.Add(this.testRF);
            this.groupBox1.Controls.Add(this.saveTrees);
            this.groupBox1.Controls.Add(this.useCV);
            this.groupBox1.Controls.Add(this.setInhyperCube);
            this.groupBox1.Controls.Add(this.nCv);
            this.groupBox1.Controls.Add(this.useRF);
            this.groupBox1.Location = new System.Drawing.Point(12, 110);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(299, 267);
            this.groupBox1.TabIndex = 20;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Script Config";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(3, 146);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(73, 13);
            this.label5.TabIndex = 73;
            this.label5.Text = "linear hopping";
            // 
            // hopping
            // 
            this.hopping.Location = new System.Drawing.Point(90, 143);
            this.hopping.Name = "hopping";
            this.hopping.Size = new System.Drawing.Size(60, 20);
            this.hopping.TabIndex = 74;
            this.hopping.TextChanged += new System.EventHandler(this.textBox2_TextChanged);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(9, 225);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(44, 13);
            this.label4.TabIndex = 71;
            this.label4.Text = "M terms";
            // 
            // m_terms
            // 
            this.m_terms.Location = new System.Drawing.Point(111, 223);
            this.m_terms.Name = "m_terms";
            this.m_terms.Size = new System.Drawing.Size(60, 20);
            this.m_terms.TabIndex = 72;
            // 
            // useParallel
            // 
            this.useParallel.AutoSize = true;
            this.useParallel.Location = new System.Drawing.Point(6, 28);
            this.useParallel.Name = "useParallel";
            this.useParallel.Size = new System.Drawing.Size(83, 17);
            this.useParallel.TabIndex = 70;
            this.useParallel.Text = "Run Parallel";
            this.useParallel.UseVisualStyleBackColor = true;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(6, 201);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(103, 13);
            this.label3.TabIndex = 31;
            this.label3.Text = "Threshold WF norm:";
            // 
            // setClassification
            // 
            this.setClassification.AutoSize = true;
            this.setClassification.Location = new System.Drawing.Point(6, 121);
            this.setClassification.Name = "setClassification";
            this.setClassification.Size = new System.Drawing.Size(203, 17);
            this.setClassification.TabIndex = 66;
            this.setClassification.Text = "use Classification (regression is defult)";
            this.setClassification.UseVisualStyleBackColor = true;
            // 
            // fixThreshold
            // 
            this.fixThreshold.Location = new System.Drawing.Point(111, 197);
            this.fixThreshold.Name = "fixThreshold";
            this.fixThreshold.Size = new System.Drawing.Size(60, 20);
            this.fixThreshold.TabIndex = 69;
            // 
            // testWf
            // 
            this.testWf.AutoSize = true;
            this.testWf.Location = new System.Drawing.Point(130, 51);
            this.testWf.Name = "testWf";
            this.testWf.Size = new System.Drawing.Size(139, 17);
            this.testWf.TabIndex = 64;
            this.testWf.Text = "estimate RF of wavelets";
            this.testWf.UseVisualStyleBackColor = true;
            // 
            // evaluateSmoothness
            // 
            this.evaluateSmoothness.AutoSize = true;
            this.evaluateSmoothness.Location = new System.Drawing.Point(6, 75);
            this.evaluateSmoothness.Name = "evaluateSmoothness";
            this.evaluateSmoothness.Size = new System.Drawing.Size(126, 17);
            this.evaluateSmoothness.TabIndex = 64;
            this.evaluateSmoothness.Text = "estimate Smoothness";
            this.evaluateSmoothness.UseVisualStyleBackColor = true;
            // 
            // testRF
            // 
            this.testRF.AutoSize = true;
            this.testRF.Location = new System.Drawing.Point(130, 31);
            this.testRF.Name = "testRF";
            this.testRF.Size = new System.Drawing.Size(151, 17);
            this.testRF.TabIndex = 63;
            this.testRF.Text = "estimate RF (no Wavelets)";
            this.testRF.UseVisualStyleBackColor = true;
            // 
            // saveTrees
            // 
            this.saveTrees.AutoSize = true;
            this.saveTrees.Location = new System.Drawing.Point(130, 72);
            this.saveTrees.Name = "saveTrees";
            this.saveTrees.Size = new System.Drawing.Size(124, 17);
            this.saveTrees.TabIndex = 61;
            this.saveTrees.Text = "save trees in archive";
            this.saveTrees.UseVisualStyleBackColor = true;
            // 
            // useCV
            // 
            this.useCV.AutoSize = true;
            this.useCV.Location = new System.Drawing.Point(6, 98);
            this.useCV.Name = "useCV";
            this.useCV.Size = new System.Drawing.Size(122, 17);
            this.useCV.TabIndex = 60;
            this.useCV.Text = "Fold cross validation";
            this.useCV.UseVisualStyleBackColor = true;
            // 
            // setInhyperCube
            // 
            this.setInhyperCube.AutoSize = true;
            this.setInhyperCube.Location = new System.Drawing.Point(6, 51);
            this.setInhyperCube.Name = "setInhyperCube";
            this.setInhyperCube.Size = new System.Drawing.Size(109, 17);
            this.setInhyperCube.TabIndex = 7;
            this.setInhyperCube.Text = "Set in hyper cube";
            this.setInhyperCube.UseVisualStyleBackColor = true;
            // 
            // nCv
            // 
            this.nCv.Location = new System.Drawing.Point(130, 95);
            this.nCv.Name = "nCv";
            this.nCv.Size = new System.Drawing.Size(60, 20);
            this.nCv.TabIndex = 59;
            // 
            // useRF
            // 
            this.useRF.AutoSize = true;
            this.useRF.Location = new System.Drawing.Point(6, 169);
            this.useRF.Name = "useRF";
            this.useRF.Size = new System.Drawing.Size(113, 17);
            this.useRF.TabIndex = 3;
            this.useRF.Text = "non linear hopping";
            this.useRF.UseVisualStyleBackColor = true;
            // 
            // btnScript
            // 
            this.btnScript.Location = new System.Drawing.Point(368, 294);
            this.btnScript.Name = "btnScript";
            this.btnScript.Size = new System.Drawing.Size(158, 47);
            this.btnScript.TabIndex = 21;
            this.btnScript.Text = "Run Script !!!";
            this.btnScript.UseVisualStyleBackColor = true;
            this.btnScript.Click += new System.EventHandler(this.btnScript_Click);
            // 
            // groupBox2
            // 
            this.groupBox2.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.groupBox2.Controls.Add(this.label2);
            this.groupBox2.Controls.Add(this.dbPath);
            this.groupBox2.Controls.Add(this.label1);
            this.groupBox2.Controls.Add(this.resultsPath);
            this.groupBox2.Location = new System.Drawing.Point(12, 18);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(299, 79);
            this.groupBox2.TabIndex = 21;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "I.O Config";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(7, 24);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(46, 13);
            this.label2.TabIndex = 11;
            this.label2.Text = "DB path";
            // 
            // dbPath
            // 
            this.dbPath.Location = new System.Drawing.Point(91, 21);
            this.dbPath.Name = "dbPath";
            this.dbPath.Size = new System.Drawing.Size(194, 20);
            this.dbPath.TabIndex = 10;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(6, 53);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(66, 13);
            this.label1.TabIndex = 9;
            this.label1.Text = "Results path";
            // 
            // resultsPath
            // 
            this.resultsPath.Location = new System.Drawing.Point(90, 50);
            this.resultsPath.Name = "resultsPath";
            this.resultsPath.Size = new System.Drawing.Size(194, 20);
            this.resultsPath.TabIndex = 8;
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(5, 76);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(74, 13);
            this.label10.TabIndex = 28;
            this.label10.Text = "% RF Bagging";
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(6, 101);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(123, 13);
            this.label7.TabIndex = 16;
            this.label7.Text = "Approximation Threshold";
            // 
            // approxThresh
            // 
            this.approxThresh.Location = new System.Drawing.Point(146, 97);
            this.approxThresh.Name = "approxThresh";
            this.approxThresh.Size = new System.Drawing.Size(48, 20);
            this.approxThresh.TabIndex = 15;
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(6, 154);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(76, 13);
            this.label8.TabIndex = 20;
            this.label8.Text = "Min Node Size";
            // 
            // minNodeSize
            // 
            this.minNodeSize.Location = new System.Drawing.Point(147, 150);
            this.minNodeSize.Name = "minNodeSize";
            this.minNodeSize.Size = new System.Drawing.Size(48, 20);
            this.minNodeSize.TabIndex = 19;
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(6, 130);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(67, 13);
            this.label9.TabIndex = 18;
            this.label9.Text = "wavelets Lp ";
            // 
            // partitionType
            // 
            this.partitionType.Location = new System.Drawing.Point(147, 123);
            this.partitionType.Name = "partitionType";
            this.partitionType.Size = new System.Drawing.Size(48, 20);
            this.partitionType.TabIndex = 17;
            // 
            // useContNorms
            // 
            this.useContNorms.Location = new System.Drawing.Point(147, 202);
            this.useContNorms.Name = "useContNorms";
            this.useContNorms.Size = new System.Drawing.Size(48, 20);
            this.useContNorms.TabIndex = 23;
            // 
            // boundLevelDepth
            // 
            this.boundLevelDepth.Location = new System.Drawing.Point(147, 176);
            this.boundLevelDepth.Name = "boundLevelDepth";
            this.boundLevelDepth.Size = new System.Drawing.Size(48, 20);
            this.boundLevelDepth.TabIndex = 21;
            // 
            // label24
            // 
            this.label24.AutoSize = true;
            this.label24.Location = new System.Drawing.Point(6, 205);
            this.label24.Name = "label24";
            this.label24.Size = new System.Drawing.Size(118, 13);
            this.label24.TabIndex = 48;
            this.label24.Text = "Non discrete WF norms";
            // 
            // label26
            // 
            this.label26.AutoSize = true;
            this.label26.Location = new System.Drawing.Point(8, 180);
            this.label26.Name = "label26";
            this.label26.Size = new System.Drawing.Size(134, 13);
            this.label26.TabIndex = 44;
            this.label26.Text = "Bound Level (in estimation)";
            // 
            // groupBox4
            // 
            this.groupBox4.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.groupBox4.Controls.Add(this.nTrees);
            this.groupBox4.Controls.Add(this.label19);
            this.groupBox4.Controls.Add(this.nFeaturesStr);
            this.groupBox4.Controls.Add(this.approxThresh);
            this.groupBox4.Controls.Add(this.label10);
            this.groupBox4.Controls.Add(this.label7);
            this.groupBox4.Controls.Add(this.label35);
            this.groupBox4.Controls.Add(this.bagginPercent);
            this.groupBox4.Controls.Add(this.partitionType);
            this.groupBox4.Controls.Add(this.label9);
            this.groupBox4.Controls.Add(this.minNodeSize);
            this.groupBox4.Controls.Add(this.label8);
            this.groupBox4.Controls.Add(this.label23);
            this.groupBox4.Controls.Add(this.errTypeTest);
            this.groupBox4.Controls.Add(this.label24);
            this.groupBox4.Controls.Add(this.boundLevelDepth);
            this.groupBox4.Controls.Add(this.useContNorms);
            this.groupBox4.Controls.Add(this.label26);
            this.groupBox4.Location = new System.Drawing.Point(332, 18);
            this.groupBox4.Name = "groupBox4";
            this.groupBox4.Size = new System.Drawing.Size(218, 259);
            this.groupBox4.TabIndex = 21;
            this.groupBox4.TabStop = false;
            this.groupBox4.Text = "Parameters settings";
            // 
            // nTrees
            // 
            this.nTrees.Location = new System.Drawing.Point(148, 24);
            this.nTrees.Name = "nTrees";
            this.nTrees.Size = new System.Drawing.Size(48, 20);
            this.nTrees.TabIndex = 55;
            // 
            // label19
            // 
            this.label19.AutoSize = true;
            this.label19.Location = new System.Drawing.Point(8, 48);
            this.label19.Name = "label19";
            this.label19.Size = new System.Drawing.Size(73, 13);
            this.label19.TabIndex = 46;
            this.label19.Text = "N features RF";
            // 
            // nFeaturesStr
            // 
            this.nFeaturesStr.Location = new System.Drawing.Point(148, 49);
            this.nFeaturesStr.Name = "nFeaturesStr";
            this.nFeaturesStr.Size = new System.Drawing.Size(48, 20);
            this.nFeaturesStr.TabIndex = 45;
            // 
            // label35
            // 
            this.label35.AutoSize = true;
            this.label35.Location = new System.Drawing.Point(8, 27);
            this.label35.Name = "label35";
            this.label35.Size = new System.Drawing.Size(46, 13);
            this.label35.TabIndex = 30;
            this.label35.Text = "RF Num";
            // 
            // bagginPercent
            // 
            this.bagginPercent.Location = new System.Drawing.Point(146, 72);
            this.bagginPercent.Name = "bagginPercent";
            this.bagginPercent.Size = new System.Drawing.Size(48, 20);
            this.bagginPercent.TabIndex = 29;
            // 
            // label23
            // 
            this.label23.AutoSize = true;
            this.label23.Location = new System.Drawing.Point(6, 232);
            this.label23.Name = "label23";
            this.label23.Size = new System.Drawing.Size(117, 13);
            this.label23.TabIndex = 50;
            this.label23.Text = "Error Type in estimation";
            // 
            // errTypeTest
            // 
            this.errTypeTest.Location = new System.Drawing.Point(147, 228);
            this.errTypeTest.Name = "errTypeTest";
            this.errTypeTest.Size = new System.Drawing.Size(48, 20);
            this.errTypeTest.TabIndex = 49;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.AutoScroll = true;
            this.BackColor = System.Drawing.SystemColors.InactiveCaption;
            this.ClientSize = new System.Drawing.Size(595, 389);
            this.Controls.Add(this.groupBox4);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.btnScript);
            this.Controls.Add(this.groupBox1);
            this.Name = "Form1";
            this.Text = "Wavelets decomposition";
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            this.groupBox4.ResumeLayout(false);
            this.groupBox4.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.OpenFileDialog openFileDialog1;
        private System.Windows.Forms.FolderBrowserDialog folderBrowserDialog1;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.Button btnScript;
        private System.Windows.Forms.CheckBox setInhyperCube;
        private System.Windows.Forms.CheckBox useRF;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox resultsPath;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox dbPath;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.TextBox approxThresh;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.TextBox useContNorms;
        private System.Windows.Forms.TextBox boundLevelDepth;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.TextBox minNodeSize;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.TextBox partitionType;
        private System.Windows.Forms.Label label24;
        private System.Windows.Forms.Label label26;
        private System.Windows.Forms.GroupBox groupBox4;
        private System.Windows.Forms.TextBox nTrees;
        private System.Windows.Forms.Label label19;
        private System.Windows.Forms.TextBox nFeaturesStr;
        private System.Windows.Forms.Label label35;
        private System.Windows.Forms.TextBox bagginPercent;
        private System.Windows.Forms.CheckBox useCV;
        private System.Windows.Forms.TextBox nCv;
        private System.Windows.Forms.CheckBox saveTrees;
        private System.Windows.Forms.CheckBox testRF;
        private System.Windows.Forms.CheckBox evaluateSmoothness;
        private System.Windows.Forms.CheckBox testWf;
        private System.Windows.Forms.CheckBox setClassification;
        private System.Windows.Forms.TextBox fixThreshold;
        private System.Windows.Forms.Label label23;
        private System.Windows.Forms.TextBox errTypeTest;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.CheckBox useParallel;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox hopping;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox m_terms;
    }
}


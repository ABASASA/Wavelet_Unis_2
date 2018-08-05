namespace DataScienceAnalysis
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
            this.estimateRFwaveletsCB = new System.Windows.Forms.CheckBox();
            this.runOneTreeOnTtrainingCB = new System.Windows.Forms.CheckBox();
            this.estimateRFnoVotingCB = new System.Windows.Forms.CheckBox();
            this.estimateRFonTrainingCB = new System.Windows.Forms.CheckBox();
            this.runOneTreeCB = new System.Windows.Forms.CheckBox();
            this.saveTressCB = new System.Windows.Forms.CheckBox();
            this.croosValidCB = new System.Windows.Forms.CheckBox();
            this.usePCA = new System.Windows.Forms.CheckBox();
            this.runRFProoningCB = new System.Windows.Forms.CheckBox();
            this.croosValidTB = new System.Windows.Forms.TextBox();
            this.runBoostingLearningRateCB = new System.Windows.Forms.CheckBox();
            this.runBoostingProoningCB = new System.Windows.Forms.CheckBox();
            this.runProoningCB = new System.Windows.Forms.CheckBox();
            this.runRfCB = new System.Windows.Forms.CheckBox();
            this.runBoostingCB = new System.Windows.Forms.CheckBox();
            this.rumPrallelCB = new System.Windows.Forms.CheckBox();
            this.UseS3CB = new System.Windows.Forms.CheckBox();
            this.btnScript = new System.Windows.Forms.Button();
            this.bucketTB = new System.Windows.Forms.TextBox();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.DBnameCB = new System.Windows.Forms.ComboBox();
            this.label5 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.DBTB = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.ResultsTB = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.NboostTB = new System.Windows.Forms.TextBox();
            this.label28 = new System.Windows.Forms.Label();
            this.boostingLamda0TB = new System.Windows.Forms.TextBox();
            this.label29 = new System.Windows.Forms.Label();
            this.NfirstwaveletsBoostingTB = new System.Windows.Forms.TextBox();
            this.label32 = new System.Windows.Forms.Label();
            this.NfirstPruninginBoostingTB = new System.Windows.Forms.TextBox();
            this.label34 = new System.Windows.Forms.Label();
            this.NboostingLearningRateTB = new System.Windows.Forms.TextBox();
            this.label11 = new System.Windows.Forms.Label();
            this.boostingKfuncPercentTB = new System.Windows.Forms.TextBox();
            this.label33 = new System.Windows.Forms.Label();
            this.label10 = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.NfeaturesTB = new System.Windows.Forms.TextBox();
            this.label7 = new System.Windows.Forms.Label();
            this.approxThreshTB = new System.Windows.Forms.TextBox();
            this.label8 = new System.Windows.Forms.Label();
            this.minNodeSizeTB = new System.Windows.Forms.TextBox();
            this.label9 = new System.Windows.Forms.Label();
            this.partitionTypeTB = new System.Windows.Forms.TextBox();
            this.label12 = new System.Windows.Forms.Label();
            this.splitTypeTB = new System.Windows.Forms.TextBox();
            this.label13 = new System.Windows.Forms.Label();
            this.boundLevelTB = new System.Windows.Forms.TextBox();
            this.label21 = new System.Windows.Forms.Label();
            this.pruningEstimationRange0TB = new System.Windows.Forms.TextBox();
            this.label22 = new System.Windows.Forms.Label();
            this.waveletsEstimationRange0TB = new System.Windows.Forms.TextBox();
            this.label23 = new System.Windows.Forms.Label();
            this.errTypeEstimationTB = new System.Windows.Forms.TextBox();
            this.label24 = new System.Windows.Forms.Label();
            this.waveletsSkipEstimationTB = new System.Windows.Forms.TextBox();
            this.label26 = new System.Windows.Forms.Label();
            this.waveletsPercentEstimationTB = new System.Windows.Forms.TextBox();
            this.label27 = new System.Windows.Forms.Label();
            this.trainingPercentTB = new System.Windows.Forms.TextBox();
            this.groupBox4 = new System.Windows.Forms.GroupBox();
            this.boundDepthTB = new System.Windows.Forms.TextBox();
            this.label25 = new System.Windows.Forms.Label();
            this.label30 = new System.Windows.Forms.Label();
            this.NloopsTB = new System.Windows.Forms.TextBox();
            this.label15 = new System.Windows.Forms.Label();
            this.label14 = new System.Windows.Forms.Label();
            this.pruningEstimationRange1TB = new System.Windows.Forms.TextBox();
            this.waveletsEstimationRange1TB = new System.Windows.Forms.TextBox();
            this.groupBox5 = new System.Windows.Forms.GroupBox();
            this.sparseRfCB = new System.Windows.Forms.CheckBox();
            this.sparseRfTB = new System.Windows.Forms.TextBox();
            this.BaggingWithRepCB = new System.Windows.Forms.CheckBox();
            this.label17 = new System.Windows.Forms.Label();
            this.label16 = new System.Windows.Forms.Label();
            this.label20 = new System.Windows.Forms.Label();
            this.label18 = new System.Windows.Forms.Label();
            this.RFpruningEstimationRange1TB = new System.Windows.Forms.TextBox();
            this.NrfTB = new System.Windows.Forms.TextBox();
            this.RFwaveletsEstimationRange1TB = new System.Windows.Forms.TextBox();
            this.label19 = new System.Windows.Forms.Label();
            this.RFpruningEstimationRange0TB = new System.Windows.Forms.TextBox();
            this.NfeaturesrfTB = new System.Windows.Forms.TextBox();
            this.RFwaveletsEstimationRange0TB = new System.Windows.Forms.TextBox();
            this.label35 = new System.Windows.Forms.Label();
            this.bagginPercentTB = new System.Windows.Forms.TextBox();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.groupBox3.SuspendLayout();
            this.groupBox4.SuspendLayout();
            this.groupBox5.SuspendLayout();
            this.SuspendLayout();
            // 
            // openFileDialog1
            // 
            this.openFileDialog1.FileName = "openFileDialog1";
            // 
            // groupBox1
            // 
            this.groupBox1.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.groupBox1.Controls.Add(this.estimateRFwaveletsCB);
            this.groupBox1.Controls.Add(this.runOneTreeOnTtrainingCB);
            this.groupBox1.Controls.Add(this.estimateRFnoVotingCB);
            this.groupBox1.Controls.Add(this.estimateRFonTrainingCB);
            this.groupBox1.Controls.Add(this.runOneTreeCB);
            this.groupBox1.Controls.Add(this.saveTressCB);
            this.groupBox1.Controls.Add(this.croosValidCB);
            this.groupBox1.Controls.Add(this.usePCA);
            this.groupBox1.Controls.Add(this.runRFProoningCB);
            this.groupBox1.Controls.Add(this.croosValidTB);
            this.groupBox1.Controls.Add(this.runBoostingLearningRateCB);
            this.groupBox1.Controls.Add(this.runBoostingProoningCB);
            this.groupBox1.Controls.Add(this.runProoningCB);
            this.groupBox1.Controls.Add(this.runRfCB);
            this.groupBox1.Controls.Add(this.runBoostingCB);
            this.groupBox1.Controls.Add(this.rumPrallelCB);
            this.groupBox1.Location = new System.Drawing.Point(12, 166);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(299, 234);
            this.groupBox1.TabIndex = 20;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Script Config";
            // 
            // estimateRFwaveletsCB
            // 
            this.estimateRFwaveletsCB.AutoSize = true;
            this.estimateRFwaveletsCB.Location = new System.Drawing.Point(6, 210);
            this.estimateRFwaveletsCB.Name = "estimateRFwaveletsCB";
            this.estimateRFwaveletsCB.Size = new System.Drawing.Size(139, 17);
            this.estimateRFwaveletsCB.TabIndex = 64;
            this.estimateRFwaveletsCB.Text = "estimate RF of wavelets";
            this.estimateRFwaveletsCB.UseVisualStyleBackColor = true;
            // 
            // runOneTreeOnTtrainingCB
            // 
            this.runOneTreeOnTtrainingCB.AutoSize = true;
            this.runOneTreeOnTtrainingCB.Location = new System.Drawing.Point(6, 187);
            this.runOneTreeOnTtrainingCB.Name = "runOneTreeOnTtrainingCB";
            this.runOneTreeOnTtrainingCB.Size = new System.Drawing.Size(159, 17);
            this.runOneTreeOnTtrainingCB.TabIndex = 65;
            this.runOneTreeOnTtrainingCB.Text = "estimate one tree on training";
            this.runOneTreeOnTtrainingCB.UseVisualStyleBackColor = true;
            // 
            // estimateRFnoVotingCB
            // 
            this.estimateRFnoVotingCB.AutoSize = true;
            this.estimateRFnoVotingCB.Location = new System.Drawing.Point(130, 163);
            this.estimateRFnoVotingCB.Name = "estimateRFnoVotingCB";
            this.estimateRFnoVotingCB.Size = new System.Drawing.Size(152, 17);
            this.estimateRFnoVotingCB.TabIndex = 64;
            this.estimateRFnoVotingCB.Text = "estimate RF without Voting";
            this.estimateRFnoVotingCB.UseVisualStyleBackColor = true;
            // 
            // estimateRFonTrainingCB
            // 
            this.estimateRFonTrainingCB.AutoSize = true;
            this.estimateRFonTrainingCB.Location = new System.Drawing.Point(130, 140);
            this.estimateRFonTrainingCB.Name = "estimateRFonTrainingCB";
            this.estimateRFonTrainingCB.Size = new System.Drawing.Size(149, 17);
            this.estimateRFonTrainingCB.TabIndex = 63;
            this.estimateRFonTrainingCB.Text = "estimate RF on trainng set";
            this.estimateRFonTrainingCB.UseVisualStyleBackColor = true;
            // 
            // runOneTreeCB
            // 
            this.runOneTreeCB.AutoSize = true;
            this.runOneTreeCB.Location = new System.Drawing.Point(6, 163);
            this.runOneTreeCB.Name = "runOneTreeCB";
            this.runOneTreeCB.Size = new System.Drawing.Size(107, 17);
            this.runOneTreeCB.TabIndex = 62;
            this.runOneTreeCB.Text = "estimate one tree";
            this.runOneTreeCB.UseVisualStyleBackColor = true;
            // 
            // saveTressCB
            // 
            this.saveTressCB.AutoSize = true;
            this.saveTressCB.Location = new System.Drawing.Point(6, 140);
            this.saveTressCB.Name = "saveTressCB";
            this.saveTressCB.Size = new System.Drawing.Size(124, 17);
            this.saveTressCB.TabIndex = 61;
            this.saveTressCB.Text = "save trees in archive";
            this.saveTressCB.UseVisualStyleBackColor = true;
            // 
            // croosValidCB
            // 
            this.croosValidCB.AutoSize = true;
            this.croosValidCB.Location = new System.Drawing.Point(6, 116);
            this.croosValidCB.Name = "croosValidCB";
            this.croosValidCB.Size = new System.Drawing.Size(122, 17);
            this.croosValidCB.TabIndex = 60;
            this.croosValidCB.Text = "Fold cross validation";
            this.croosValidCB.UseVisualStyleBackColor = true;
            // 
            // usePCA
            // 
            this.usePCA.AutoSize = true;
            this.usePCA.Location = new System.Drawing.Point(6, 50);
            this.usePCA.Name = "usePCA";
            this.usePCA.Size = new System.Drawing.Size(108, 17);
            this.usePCA.TabIndex = 1;
            this.usePCA.Text = "Use PCA on data";
            this.usePCA.UseVisualStyleBackColor = true;
            // 
            // runRFProoningCB
            // 
            this.runRFProoningCB.AutoSize = true;
            this.runRFProoningCB.Location = new System.Drawing.Point(130, 28);
            this.runRFProoningCB.Name = "runRFProoningCB";
            this.runRFProoningCB.Size = new System.Drawing.Size(97, 17);
            this.runRFProoningCB.TabIndex = 7;
            this.runRFProoningCB.Text = "runRFProoning";
            this.runRFProoningCB.UseVisualStyleBackColor = true;
            // 
            // croosValidTB
            // 
            this.croosValidTB.Location = new System.Drawing.Point(130, 114);
            this.croosValidTB.Name = "croosValidTB";
            this.croosValidTB.Size = new System.Drawing.Size(37, 20);
            this.croosValidTB.TabIndex = 59;
            // 
            // runBoostingLearningRateCB
            // 
            this.runBoostingLearningRateCB.AutoSize = true;
            this.runBoostingLearningRateCB.Location = new System.Drawing.Point(130, 91);
            this.runBoostingLearningRateCB.Name = "runBoostingLearningRateCB";
            this.runBoostingLearningRateCB.Size = new System.Drawing.Size(146, 17);
            this.runBoostingLearningRateCB.TabIndex = 6;
            this.runBoostingLearningRateCB.Text = "runBoostingLearningRate";
            this.runBoostingLearningRateCB.UseVisualStyleBackColor = true;
            // 
            // runBoostingProoningCB
            // 
            this.runBoostingProoningCB.AutoSize = true;
            this.runBoostingProoningCB.Location = new System.Drawing.Point(130, 70);
            this.runBoostingProoningCB.Name = "runBoostingProoningCB";
            this.runBoostingProoningCB.Size = new System.Drawing.Size(124, 17);
            this.runBoostingProoningCB.TabIndex = 5;
            this.runBoostingProoningCB.Text = "runBoostingProoning";
            this.runBoostingProoningCB.UseVisualStyleBackColor = true;
            // 
            // runProoningCB
            // 
            this.runProoningCB.AutoSize = true;
            this.runProoningCB.Location = new System.Drawing.Point(6, 70);
            this.runProoningCB.Name = "runProoningCB";
            this.runProoningCB.Size = new System.Drawing.Size(83, 17);
            this.runProoningCB.TabIndex = 4;
            this.runProoningCB.Text = "runProoning";
            this.runProoningCB.UseVisualStyleBackColor = true;
            // 
            // runRfCB
            // 
            this.runRfCB.AutoSize = true;
            this.runRfCB.Location = new System.Drawing.Point(6, 91);
            this.runRfCB.Name = "runRfCB";
            this.runRfCB.Size = new System.Drawing.Size(52, 17);
            this.runRfCB.TabIndex = 3;
            this.runRfCB.Text = "runRf";
            this.runRfCB.UseVisualStyleBackColor = true;
            // 
            // runBoostingCB
            // 
            this.runBoostingCB.AutoSize = true;
            this.runBoostingCB.Location = new System.Drawing.Point(130, 50);
            this.runBoostingCB.Name = "runBoostingCB";
            this.runBoostingCB.Size = new System.Drawing.Size(82, 17);
            this.runBoostingCB.TabIndex = 2;
            this.runBoostingCB.Text = "runBoosting";
            this.runBoostingCB.UseVisualStyleBackColor = true;
            // 
            // rumPrallelCB
            // 
            this.rumPrallelCB.AutoSize = true;
            this.rumPrallelCB.Location = new System.Drawing.Point(6, 28);
            this.rumPrallelCB.Name = "rumPrallelCB";
            this.rumPrallelCB.Size = new System.Drawing.Size(71, 17);
            this.rumPrallelCB.TabIndex = 1;
            this.rumPrallelCB.Text = "rumPrallel";
            this.rumPrallelCB.UseVisualStyleBackColor = true;
            // 
            // UseS3CB
            // 
            this.UseS3CB.AutoSize = true;
            this.UseS3CB.Location = new System.Drawing.Point(9, 23);
            this.UseS3CB.Name = "UseS3CB";
            this.UseS3CB.Size = new System.Drawing.Size(58, 17);
            this.UseS3CB.TabIndex = 0;
            this.UseS3CB.Text = "UseS3";
            this.UseS3CB.UseVisualStyleBackColor = true;
            // 
            // btnScript
            // 
            this.btnScript.Location = new System.Drawing.Point(551, 413);
            this.btnScript.Name = "btnScript";
            this.btnScript.Size = new System.Drawing.Size(158, 47);
            this.btnScript.TabIndex = 21;
            this.btnScript.Text = "Run Script !!!";
            this.btnScript.UseVisualStyleBackColor = true;
            this.btnScript.Click += new System.EventHandler(this.btnScript_Click);
            // 
            // bucketTB
            // 
            this.bucketTB.Location = new System.Drawing.Point(163, 20);
            this.bucketTB.Name = "bucketTB";
            this.bucketTB.Size = new System.Drawing.Size(121, 20);
            this.bucketTB.TabIndex = 8;
            // 
            // groupBox2
            // 
            this.groupBox2.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.groupBox2.Controls.Add(this.DBnameCB);
            this.groupBox2.Controls.Add(this.label5);
            this.groupBox2.Controls.Add(this.label3);
            this.groupBox2.Controls.Add(this.bucketTB);
            this.groupBox2.Controls.Add(this.label2);
            this.groupBox2.Controls.Add(this.DBTB);
            this.groupBox2.Controls.Add(this.label1);
            this.groupBox2.Controls.Add(this.ResultsTB);
            this.groupBox2.Controls.Add(this.UseS3CB);
            this.groupBox2.Location = new System.Drawing.Point(12, 18);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(299, 146);
            this.groupBox2.TabIndex = 21;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "I.O Config";
            // 
            // DBnameCB
            // 
            this.DBnameCB.FormattingEnabled = true;
            this.DBnameCB.Items.AddRange(new object[] {
            "red wine"});
            this.DBnameCB.Location = new System.Drawing.Point(91, 52);
            this.DBnameCB.Name = "DBnameCB";
            this.DBnameCB.Size = new System.Drawing.Size(121, 21);
            this.DBnameCB.TabIndex = 14;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(7, 52);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(51, 13);
            this.label5.TabIndex = 13;
            this.label5.Text = "DB name";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(87, 25);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(70, 13);
            this.label3.TabIndex = 12;
            this.label3.Text = "Bucket name";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(7, 81);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(46, 13);
            this.label2.TabIndex = 11;
            this.label2.Text = "DB path";
            // 
            // DBTB
            // 
            this.DBTB.Location = new System.Drawing.Point(91, 78);
            this.DBTB.Name = "DBTB";
            this.DBTB.Size = new System.Drawing.Size(194, 20);
            this.DBTB.TabIndex = 10;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(6, 110);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(66, 13);
            this.label1.TabIndex = 9;
            this.label1.Text = "Results path";
            // 
            // ResultsTB
            // 
            this.ResultsTB.Location = new System.Drawing.Point(90, 107);
            this.ResultsTB.Name = "ResultsTB";
            this.ResultsTB.Size = new System.Drawing.Size(194, 20);
            this.ResultsTB.TabIndex = 8;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(19, 430);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(151, 13);
            this.label4.TabIndex = 22;
            this.label4.Text = "support at orenelis@gmail.com";
            // 
            // groupBox3
            // 
            this.groupBox3.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.groupBox3.Controls.Add(this.NboostTB);
            this.groupBox3.Controls.Add(this.label28);
            this.groupBox3.Controls.Add(this.boostingLamda0TB);
            this.groupBox3.Controls.Add(this.label29);
            this.groupBox3.Controls.Add(this.NfirstwaveletsBoostingTB);
            this.groupBox3.Controls.Add(this.label32);
            this.groupBox3.Controls.Add(this.NfirstPruninginBoostingTB);
            this.groupBox3.Controls.Add(this.label34);
            this.groupBox3.Controls.Add(this.NboostingLearningRateTB);
            this.groupBox3.Controls.Add(this.label11);
            this.groupBox3.Controls.Add(this.boostingKfuncPercentTB);
            this.groupBox3.Controls.Add(this.label33);
            this.groupBox3.Location = new System.Drawing.Point(587, 234);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Size = new System.Drawing.Size(272, 166);
            this.groupBox3.TabIndex = 21;
            this.groupBox3.TabStop = false;
            this.groupBox3.Text = "Parameters Setting";
            // 
            // NboostTB
            // 
            this.NboostTB.Location = new System.Drawing.Point(164, 17);
            this.NboostTB.Name = "NboostTB";
            this.NboostTB.Size = new System.Drawing.Size(48, 20);
            this.NboostTB.TabIndex = 41;
            // 
            // label28
            // 
            this.label28.AutoSize = true;
            this.label28.Location = new System.Drawing.Point(10, 142);
            this.label28.Name = "label28";
            this.label28.Size = new System.Drawing.Size(125, 13);
            this.label28.TabIndex = 40;
            this.label28.Text = "N Boosting learning Rate";
            // 
            // boostingLamda0TB
            // 
            this.boostingLamda0TB.Location = new System.Drawing.Point(165, 117);
            this.boostingLamda0TB.Name = "boostingLamda0TB";
            this.boostingLamda0TB.Size = new System.Drawing.Size(48, 20);
            this.boostingLamda0TB.TabIndex = 39;
            // 
            // label29
            // 
            this.label29.AutoSize = true;
            this.label29.Location = new System.Drawing.Point(7, 92);
            this.label29.Name = "label29";
            this.label29.Size = new System.Drawing.Size(115, 13);
            this.label29.TabIndex = 38;
            this.label29.Text = "N wavelets in Boosting";
            // 
            // NfirstwaveletsBoostingTB
            // 
            this.NfirstwaveletsBoostingTB.Location = new System.Drawing.Point(165, 92);
            this.NfirstwaveletsBoostingTB.Name = "NfirstwaveletsBoostingTB";
            this.NfirstwaveletsBoostingTB.Size = new System.Drawing.Size(48, 20);
            this.NfirstwaveletsBoostingTB.TabIndex = 37;
            // 
            // label32
            // 
            this.label32.AutoSize = true;
            this.label32.Location = new System.Drawing.Point(9, 120);
            this.label32.Name = "label32";
            this.label32.Size = new System.Drawing.Size(88, 13);
            this.label32.TabIndex = 36;
            this.label32.Text = "Boosting lamda 0";
            // 
            // NfirstPruninginBoostingTB
            // 
            this.NfirstPruninginBoostingTB.Location = new System.Drawing.Point(165, 67);
            this.NfirstPruninginBoostingTB.Name = "NfirstPruninginBoostingTB";
            this.NfirstPruninginBoostingTB.Size = new System.Drawing.Size(48, 20);
            this.NfirstPruninginBoostingTB.TabIndex = 27;
            // 
            // label34
            // 
            this.label34.AutoSize = true;
            this.label34.Location = new System.Drawing.Point(10, 24);
            this.label34.Name = "label34";
            this.label34.Size = new System.Drawing.Size(58, 13);
            this.label34.TabIndex = 32;
            this.label34.Text = "N boosting";
            // 
            // NboostingLearningRateTB
            // 
            this.NboostingLearningRateTB.Location = new System.Drawing.Point(166, 139);
            this.NboostingLearningRateTB.Name = "NboostingLearningRateTB";
            this.NboostingLearningRateTB.Size = new System.Drawing.Size(48, 20);
            this.NboostingLearningRateTB.TabIndex = 35;
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.Location = new System.Drawing.Point(7, 44);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(103, 13);
            this.label11.TabIndex = 26;
            this.label11.Text = "% in boosting K func";
            // 
            // boostingKfuncPercentTB
            // 
            this.boostingKfuncPercentTB.Location = new System.Drawing.Point(165, 40);
            this.boostingKfuncPercentTB.Name = "boostingKfuncPercentTB";
            this.boostingKfuncPercentTB.Size = new System.Drawing.Size(48, 20);
            this.boostingKfuncPercentTB.TabIndex = 25;
            // 
            // label33
            // 
            this.label33.AutoSize = true;
            this.label33.Location = new System.Drawing.Point(8, 70);
            this.label33.Name = "label33";
            this.label33.Size = new System.Drawing.Size(128, 13);
            this.label33.TabIndex = 34;
            this.label33.Text = "N first Pruning in Boosting";
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(6, 78);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(74, 13);
            this.label10.TabIndex = 28;
            this.label10.Text = "% RF Bagging";
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(6, 47);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(56, 13);
            this.label6.TabIndex = 14;
            this.label6.Text = "N features";
            // 
            // NfeaturesTB
            // 
            this.NfeaturesTB.Location = new System.Drawing.Point(144, 44);
            this.NfeaturesTB.Name = "NfeaturesTB";
            this.NfeaturesTB.Size = new System.Drawing.Size(48, 20);
            this.NfeaturesTB.TabIndex = 13;
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(4, 73);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(123, 13);
            this.label7.TabIndex = 16;
            this.label7.Text = "Approximation Threshold";
            // 
            // approxThreshTB
            // 
            this.approxThreshTB.Location = new System.Drawing.Point(144, 69);
            this.approxThreshTB.Name = "approxThreshTB";
            this.approxThreshTB.Size = new System.Drawing.Size(48, 20);
            this.approxThreshTB.TabIndex = 15;
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(4, 126);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(76, 13);
            this.label8.TabIndex = 20;
            this.label8.Text = "Min Node Size";
            // 
            // minNodeSizeTB
            // 
            this.minNodeSizeTB.Location = new System.Drawing.Point(145, 122);
            this.minNodeSizeTB.Name = "minNodeSizeTB";
            this.minNodeSizeTB.Size = new System.Drawing.Size(48, 20);
            this.minNodeSizeTB.TabIndex = 19;
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(4, 102);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(94, 13);
            this.label9.TabIndex = 18;
            this.label9.Text = "Partition ErrorType";
            // 
            // partitionTypeTB
            // 
            this.partitionTypeTB.Location = new System.Drawing.Point(145, 95);
            this.partitionTypeTB.Name = "partitionTypeTB";
            this.partitionTypeTB.Size = new System.Drawing.Size(48, 20);
            this.partitionTypeTB.TabIndex = 17;
            // 
            // label12
            // 
            this.label12.AutoSize = true;
            this.label12.Location = new System.Drawing.Point(6, 229);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(133, 13);
            this.label12.TabIndex = 24;
            this.label12.Text = "% of wavelets in estimation";
            // 
            // splitTypeTB
            // 
            this.splitTypeTB.Location = new System.Drawing.Point(145, 174);
            this.splitTypeTB.Name = "splitTypeTB";
            this.splitTypeTB.Size = new System.Drawing.Size(48, 20);
            this.splitTypeTB.TabIndex = 23;
            // 
            // label13
            // 
            this.label13.AutoSize = true;
            this.label13.Location = new System.Drawing.Point(-3, 256);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(134, 13);
            this.label13.TabIndex = 22;
            this.label13.Text = "Skip wavelets in estimation";
            // 
            // boundLevelTB
            // 
            this.boundLevelTB.Location = new System.Drawing.Point(145, 149);
            this.boundLevelTB.Name = "boundLevelTB";
            this.boundLevelTB.Size = new System.Drawing.Size(48, 20);
            this.boundLevelTB.TabIndex = 21;
            // 
            // label21
            // 
            this.label21.AutoSize = true;
            this.label21.Location = new System.Drawing.Point(7, 330);
            this.label21.Name = "label21";
            this.label21.Size = new System.Drawing.Size(127, 13);
            this.label21.TabIndex = 54;
            this.label21.Text = "pruning estimation Range";
            // 
            // pruningEstimationRange0TB
            // 
            this.pruningEstimationRange0TB.Location = new System.Drawing.Point(148, 327);
            this.pruningEstimationRange0TB.Name = "pruningEstimationRange0TB";
            this.pruningEstimationRange0TB.Size = new System.Drawing.Size(39, 20);
            this.pruningEstimationRange0TB.TabIndex = 53;
            // 
            // label22
            // 
            this.label22.AutoSize = true;
            this.label22.Location = new System.Drawing.Point(7, 305);
            this.label22.Name = "label22";
            this.label22.Size = new System.Drawing.Size(134, 13);
            this.label22.TabIndex = 52;
            this.label22.Text = "wavelets estimation Range";
            // 
            // waveletsEstimationRange0TB
            // 
            this.waveletsEstimationRange0TB.Location = new System.Drawing.Point(148, 301);
            this.waveletsEstimationRange0TB.Name = "waveletsEstimationRange0TB";
            this.waveletsEstimationRange0TB.Size = new System.Drawing.Size(39, 20);
            this.waveletsEstimationRange0TB.TabIndex = 51;
            // 
            // label23
            // 
            this.label23.AutoSize = true;
            this.label23.Location = new System.Drawing.Point(7, 277);
            this.label23.Name = "label23";
            this.label23.Size = new System.Drawing.Size(117, 13);
            this.label23.TabIndex = 50;
            this.label23.Text = "Error Type in estimation";
            // 
            // errTypeEstimationTB
            // 
            this.errTypeEstimationTB.Location = new System.Drawing.Point(149, 275);
            this.errTypeEstimationTB.Name = "errTypeEstimationTB";
            this.errTypeEstimationTB.Size = new System.Drawing.Size(45, 20);
            this.errTypeEstimationTB.TabIndex = 49;
            // 
            // label24
            // 
            this.label24.AutoSize = true;
            this.label24.Location = new System.Drawing.Point(4, 177);
            this.label24.Name = "label24";
            this.label24.Size = new System.Drawing.Size(50, 13);
            this.label24.TabIndex = 48;
            this.label24.Text = "Split type";
            // 
            // waveletsSkipEstimationTB
            // 
            this.waveletsSkipEstimationTB.Location = new System.Drawing.Point(147, 252);
            this.waveletsSkipEstimationTB.Name = "waveletsSkipEstimationTB";
            this.waveletsSkipEstimationTB.Size = new System.Drawing.Size(48, 20);
            this.waveletsSkipEstimationTB.TabIndex = 47;
            // 
            // label26
            // 
            this.label26.AutoSize = true;
            this.label26.Location = new System.Drawing.Point(6, 152);
            this.label26.Name = "label26";
            this.label26.Size = new System.Drawing.Size(134, 13);
            this.label26.TabIndex = 44;
            this.label26.Text = "Bound Level (in estimation)";
            // 
            // waveletsPercentEstimationTB
            // 
            this.waveletsPercentEstimationTB.Location = new System.Drawing.Point(147, 226);
            this.waveletsPercentEstimationTB.Name = "waveletsPercentEstimationTB";
            this.waveletsPercentEstimationTB.Size = new System.Drawing.Size(48, 20);
            this.waveletsPercentEstimationTB.TabIndex = 43;
            // 
            // label27
            // 
            this.label27.AutoSize = true;
            this.label27.Location = new System.Drawing.Point(3, 202);
            this.label27.Name = "label27";
            this.label27.Size = new System.Drawing.Size(88, 13);
            this.label27.TabIndex = 42;
            this.label27.Text = "% training to take";
            // 
            // trainingPercentTB
            // 
            this.trainingPercentTB.Location = new System.Drawing.Point(146, 199);
            this.trainingPercentTB.Name = "trainingPercentTB";
            this.trainingPercentTB.Size = new System.Drawing.Size(48, 20);
            this.trainingPercentTB.TabIndex = 41;
            // 
            // groupBox4
            // 
            this.groupBox4.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.groupBox4.Controls.Add(this.boundDepthTB);
            this.groupBox4.Controls.Add(this.label25);
            this.groupBox4.Controls.Add(this.label30);
            this.groupBox4.Controls.Add(this.NloopsTB);
            this.groupBox4.Controls.Add(this.label15);
            this.groupBox4.Controls.Add(this.label14);
            this.groupBox4.Controls.Add(this.pruningEstimationRange1TB);
            this.groupBox4.Controls.Add(this.waveletsEstimationRange1TB);
            this.groupBox4.Controls.Add(this.label6);
            this.groupBox4.Controls.Add(this.NfeaturesTB);
            this.groupBox4.Controls.Add(this.label21);
            this.groupBox4.Controls.Add(this.pruningEstimationRange0TB);
            this.groupBox4.Controls.Add(this.approxThreshTB);
            this.groupBox4.Controls.Add(this.label7);
            this.groupBox4.Controls.Add(this.label22);
            this.groupBox4.Controls.Add(this.label13);
            this.groupBox4.Controls.Add(this.partitionTypeTB);
            this.groupBox4.Controls.Add(this.label9);
            this.groupBox4.Controls.Add(this.waveletsEstimationRange0TB);
            this.groupBox4.Controls.Add(this.minNodeSizeTB);
            this.groupBox4.Controls.Add(this.label8);
            this.groupBox4.Controls.Add(this.label23);
            this.groupBox4.Controls.Add(this.errTypeEstimationTB);
            this.groupBox4.Controls.Add(this.label24);
            this.groupBox4.Controls.Add(this.waveletsSkipEstimationTB);
            this.groupBox4.Controls.Add(this.boundLevelTB);
            this.groupBox4.Controls.Add(this.splitTypeTB);
            this.groupBox4.Controls.Add(this.label12);
            this.groupBox4.Controls.Add(this.label26);
            this.groupBox4.Controls.Add(this.label27);
            this.groupBox4.Controls.Add(this.waveletsPercentEstimationTB);
            this.groupBox4.Controls.Add(this.trainingPercentTB);
            this.groupBox4.Location = new System.Drawing.Point(332, 18);
            this.groupBox4.Name = "groupBox4";
            this.groupBox4.Size = new System.Drawing.Size(249, 382);
            this.groupBox4.TabIndex = 21;
            this.groupBox4.TabStop = false;
            this.groupBox4.Text = "General Parameters";
            // 
            // boundDepthTB
            // 
            this.boundDepthTB.Location = new System.Drawing.Point(149, 354);
            this.boundDepthTB.Name = "boundDepthTB";
            this.boundDepthTB.Size = new System.Drawing.Size(48, 20);
            this.boundDepthTB.TabIndex = 62;
            // 
            // label25
            // 
            this.label25.AutoSize = true;
            this.label25.Location = new System.Drawing.Point(7, 358);
            this.label25.Name = "label25";
            this.label25.Size = new System.Drawing.Size(138, 13);
            this.label25.TabIndex = 63;
            this.label25.Text = "Bound depth (in generation)";
            // 
            // label30
            // 
            this.label30.AutoSize = true;
            this.label30.Location = new System.Drawing.Point(6, 23);
            this.label30.Name = "label30";
            this.label30.Size = new System.Drawing.Size(47, 13);
            this.label30.TabIndex = 61;
            this.label30.Text = "N Loops";
            // 
            // NloopsTB
            // 
            this.NloopsTB.Location = new System.Drawing.Point(144, 20);
            this.NloopsTB.Name = "NloopsTB";
            this.NloopsTB.Size = new System.Drawing.Size(48, 20);
            this.NloopsTB.TabIndex = 60;
            // 
            // label15
            // 
            this.label15.AutoSize = true;
            this.label15.Location = new System.Drawing.Point(191, 332);
            this.label15.Name = "label15";
            this.label15.Size = new System.Drawing.Size(10, 13);
            this.label15.TabIndex = 58;
            this.label15.Text = "-";
            // 
            // label14
            // 
            this.label14.AutoSize = true;
            this.label14.Location = new System.Drawing.Point(191, 305);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(10, 13);
            this.label14.TabIndex = 57;
            this.label14.Text = "-";
            // 
            // pruningEstimationRange1TB
            // 
            this.pruningEstimationRange1TB.Location = new System.Drawing.Point(202, 328);
            this.pruningEstimationRange1TB.Name = "pruningEstimationRange1TB";
            this.pruningEstimationRange1TB.Size = new System.Drawing.Size(39, 20);
            this.pruningEstimationRange1TB.TabIndex = 56;
            // 
            // waveletsEstimationRange1TB
            // 
            this.waveletsEstimationRange1TB.Location = new System.Drawing.Point(202, 302);
            this.waveletsEstimationRange1TB.Name = "waveletsEstimationRange1TB";
            this.waveletsEstimationRange1TB.Size = new System.Drawing.Size(39, 20);
            this.waveletsEstimationRange1TB.TabIndex = 55;
            // 
            // groupBox5
            // 
            this.groupBox5.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.groupBox5.Controls.Add(this.sparseRfCB);
            this.groupBox5.Controls.Add(this.sparseRfTB);
            this.groupBox5.Controls.Add(this.BaggingWithRepCB);
            this.groupBox5.Controls.Add(this.label17);
            this.groupBox5.Controls.Add(this.label16);
            this.groupBox5.Controls.Add(this.label20);
            this.groupBox5.Controls.Add(this.label18);
            this.groupBox5.Controls.Add(this.RFpruningEstimationRange1TB);
            this.groupBox5.Controls.Add(this.NrfTB);
            this.groupBox5.Controls.Add(this.RFwaveletsEstimationRange1TB);
            this.groupBox5.Controls.Add(this.label19);
            this.groupBox5.Controls.Add(this.RFpruningEstimationRange0TB);
            this.groupBox5.Controls.Add(this.NfeaturesrfTB);
            this.groupBox5.Controls.Add(this.RFwaveletsEstimationRange0TB);
            this.groupBox5.Controls.Add(this.label10);
            this.groupBox5.Controls.Add(this.label35);
            this.groupBox5.Controls.Add(this.bagginPercentTB);
            this.groupBox5.Location = new System.Drawing.Point(587, 18);
            this.groupBox5.Name = "groupBox5";
            this.groupBox5.Size = new System.Drawing.Size(272, 210);
            this.groupBox5.TabIndex = 57;
            this.groupBox5.TabStop = false;
            this.groupBox5.Text = "Parameters Setting";
            // 
            // sparseRfCB
            // 
            this.sparseRfCB.AutoSize = true;
            this.sparseRfCB.Location = new System.Drawing.Point(8, 102);
            this.sparseRfCB.Name = "sparseRfCB";
            this.sparseRfCB.Size = new System.Drawing.Size(152, 17);
            this.sparseRfCB.TabIndex = 68;
            this.sparseRfCB.Text = "sparse number of wavelets";
            this.sparseRfCB.UseVisualStyleBackColor = true;
            // 
            // sparseRfTB
            // 
            this.sparseRfTB.Location = new System.Drawing.Point(164, 102);
            this.sparseRfTB.Name = "sparseRfTB";
            this.sparseRfTB.Size = new System.Drawing.Size(50, 20);
            this.sparseRfTB.TabIndex = 66;
            // 
            // BaggingWithRepCB
            // 
            this.BaggingWithRepCB.AutoSize = true;
            this.BaggingWithRepCB.Location = new System.Drawing.Point(8, 126);
            this.BaggingWithRepCB.Name = "BaggingWithRepCB";
            this.BaggingWithRepCB.Size = new System.Drawing.Size(141, 17);
            this.BaggingWithRepCB.TabIndex = 66;
            this.BaggingWithRepCB.Text = "Bagging With repetitions";
            this.BaggingWithRepCB.UseVisualStyleBackColor = true;
            // 
            // label17
            // 
            this.label17.AutoSize = true;
            this.label17.Location = new System.Drawing.Point(209, 184);
            this.label17.Name = "label17";
            this.label17.Size = new System.Drawing.Size(10, 13);
            this.label17.TabIndex = 64;
            this.label17.Text = "-";
            // 
            // label16
            // 
            this.label16.AutoSize = true;
            this.label16.Location = new System.Drawing.Point(6, 186);
            this.label16.Name = "label16";
            this.label16.Size = new System.Drawing.Size(144, 13);
            this.label16.TabIndex = 57;
            this.label16.Text = "RF pruning estimation Range";
            // 
            // label20
            // 
            this.label20.AutoSize = true;
            this.label20.Location = new System.Drawing.Point(209, 157);
            this.label20.Name = "label20";
            this.label20.Size = new System.Drawing.Size(10, 13);
            this.label20.TabIndex = 63;
            this.label20.Text = "-";
            // 
            // label18
            // 
            this.label18.AutoSize = true;
            this.label18.Location = new System.Drawing.Point(9, 156);
            this.label18.Name = "label18";
            this.label18.Size = new System.Drawing.Size(151, 13);
            this.label18.TabIndex = 56;
            this.label18.Text = "RF wavelets estimation Range";
            // 
            // RFpruningEstimationRange1TB
            // 
            this.RFpruningEstimationRange1TB.Location = new System.Drawing.Point(220, 180);
            this.RFpruningEstimationRange1TB.Name = "RFpruningEstimationRange1TB";
            this.RFpruningEstimationRange1TB.Size = new System.Drawing.Size(39, 20);
            this.RFpruningEstimationRange1TB.TabIndex = 62;
            // 
            // NrfTB
            // 
            this.NrfTB.Location = new System.Drawing.Point(163, 17);
            this.NrfTB.Name = "NrfTB";
            this.NrfTB.Size = new System.Drawing.Size(48, 20);
            this.NrfTB.TabIndex = 55;
            // 
            // RFwaveletsEstimationRange1TB
            // 
            this.RFwaveletsEstimationRange1TB.Location = new System.Drawing.Point(220, 154);
            this.RFwaveletsEstimationRange1TB.Name = "RFwaveletsEstimationRange1TB";
            this.RFwaveletsEstimationRange1TB.Size = new System.Drawing.Size(39, 20);
            this.RFwaveletsEstimationRange1TB.TabIndex = 61;
            // 
            // label19
            // 
            this.label19.AutoSize = true;
            this.label19.Location = new System.Drawing.Point(6, 47);
            this.label19.Name = "label19";
            this.label19.Size = new System.Drawing.Size(73, 13);
            this.label19.TabIndex = 46;
            this.label19.Text = "N features RF";
            // 
            // RFpruningEstimationRange0TB
            // 
            this.RFpruningEstimationRange0TB.Location = new System.Drawing.Point(166, 179);
            this.RFpruningEstimationRange0TB.Name = "RFpruningEstimationRange0TB";
            this.RFpruningEstimationRange0TB.Size = new System.Drawing.Size(39, 20);
            this.RFpruningEstimationRange0TB.TabIndex = 60;
            // 
            // NfeaturesrfTB
            // 
            this.NfeaturesrfTB.Location = new System.Drawing.Point(164, 44);
            this.NfeaturesrfTB.Name = "NfeaturesrfTB";
            this.NfeaturesrfTB.Size = new System.Drawing.Size(48, 20);
            this.NfeaturesrfTB.TabIndex = 45;
            // 
            // RFwaveletsEstimationRange0TB
            // 
            this.RFwaveletsEstimationRange0TB.Location = new System.Drawing.Point(166, 153);
            this.RFwaveletsEstimationRange0TB.Name = "RFwaveletsEstimationRange0TB";
            this.RFwaveletsEstimationRange0TB.Size = new System.Drawing.Size(39, 20);
            this.RFwaveletsEstimationRange0TB.TabIndex = 59;
            // 
            // label35
            // 
            this.label35.AutoSize = true;
            this.label35.Location = new System.Drawing.Point(9, 20);
            this.label35.Name = "label35";
            this.label35.Size = new System.Drawing.Size(46, 13);
            this.label35.TabIndex = 30;
            this.label35.Text = "RF Num";
            // 
            // bagginPercentTB
            // 
            this.bagginPercentTB.Location = new System.Drawing.Point(165, 75);
            this.bagginPercentTB.Name = "bagginPercentTB";
            this.bagginPercentTB.Size = new System.Drawing.Size(48, 20);
            this.bagginPercentTB.TabIndex = 29;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.AutoScroll = true;
            this.ClientSize = new System.Drawing.Size(892, 470);
            this.Controls.Add(this.groupBox5);
            this.Controls.Add(this.groupBox4);
            this.Controls.Add(this.groupBox3);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.btnScript);
            this.Controls.Add(this.groupBox1);
            this.Name = "Form1";
            this.Text = "Wavelets decomposition";
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            this.groupBox3.ResumeLayout(false);
            this.groupBox3.PerformLayout();
            this.groupBox4.ResumeLayout(false);
            this.groupBox4.PerformLayout();
            this.groupBox5.ResumeLayout(false);
            this.groupBox5.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.OpenFileDialog openFileDialog1;
        private System.Windows.Forms.FolderBrowserDialog folderBrowserDialog1;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.Button btnScript;
        private System.Windows.Forms.CheckBox runRFProoningCB;
        private System.Windows.Forms.CheckBox runBoostingLearningRateCB;
        private System.Windows.Forms.CheckBox runBoostingProoningCB;
        private System.Windows.Forms.CheckBox runProoningCB;
        private System.Windows.Forms.CheckBox runRfCB;
        private System.Windows.Forms.CheckBox runBoostingCB;
        private System.Windows.Forms.CheckBox rumPrallelCB;
        private System.Windows.Forms.CheckBox UseS3CB;
        private System.Windows.Forms.TextBox bucketTB;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox ResultsTB;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox DBTB;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.GroupBox groupBox3;
        private System.Windows.Forms.CheckBox usePCA;
        private System.Windows.Forms.ComboBox DBnameCB;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.TextBox approxThreshTB;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.TextBox NfeaturesTB;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.TextBox NfirstPruninginBoostingTB;
        private System.Windows.Forms.Label label11;
        private System.Windows.Forms.TextBox boostingKfuncPercentTB;
        private System.Windows.Forms.Label label12;
        private System.Windows.Forms.TextBox splitTypeTB;
        private System.Windows.Forms.Label label13;
        private System.Windows.Forms.TextBox boundLevelTB;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.TextBox minNodeSizeTB;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.TextBox partitionTypeTB;
        private System.Windows.Forms.Label label21;
        private System.Windows.Forms.TextBox pruningEstimationRange0TB;
        private System.Windows.Forms.Label label22;
        private System.Windows.Forms.TextBox waveletsEstimationRange0TB;
        private System.Windows.Forms.Label label23;
        private System.Windows.Forms.TextBox errTypeEstimationTB;
        private System.Windows.Forms.Label label24;
        private System.Windows.Forms.TextBox waveletsSkipEstimationTB;
        private System.Windows.Forms.Label label26;
        private System.Windows.Forms.TextBox waveletsPercentEstimationTB;
        private System.Windows.Forms.Label label27;
        private System.Windows.Forms.TextBox trainingPercentTB;
        private System.Windows.Forms.Label label28;
        private System.Windows.Forms.TextBox boostingLamda0TB;
        private System.Windows.Forms.Label label29;
        private System.Windows.Forms.TextBox NfirstwaveletsBoostingTB;
        private System.Windows.Forms.GroupBox groupBox4;
        private System.Windows.Forms.GroupBox groupBox5;
        private System.Windows.Forms.Label label18;
        private System.Windows.Forms.TextBox NrfTB;
        private System.Windows.Forms.Label label19;
        private System.Windows.Forms.TextBox NfeaturesrfTB;
        private System.Windows.Forms.Label label32;
        private System.Windows.Forms.TextBox NboostingLearningRateTB;
        private System.Windows.Forms.Label label33;
        private System.Windows.Forms.Label label34;
        private System.Windows.Forms.Label label35;
        private System.Windows.Forms.TextBox bagginPercentTB;
        private System.Windows.Forms.CheckBox croosValidCB;
        private System.Windows.Forms.TextBox croosValidTB;
        private System.Windows.Forms.TextBox NboostTB;
        private System.Windows.Forms.Label label30;
        private System.Windows.Forms.TextBox NloopsTB;
        private System.Windows.Forms.Label label15;
        private System.Windows.Forms.Label label14;
        private System.Windows.Forms.TextBox pruningEstimationRange1TB;
        private System.Windows.Forms.TextBox waveletsEstimationRange1TB;
        private System.Windows.Forms.Label label17;
        private System.Windows.Forms.Label label16;
        private System.Windows.Forms.Label label20;
        private System.Windows.Forms.TextBox RFpruningEstimationRange1TB;
        private System.Windows.Forms.TextBox RFwaveletsEstimationRange1TB;
        private System.Windows.Forms.TextBox RFpruningEstimationRange0TB;
        private System.Windows.Forms.TextBox RFwaveletsEstimationRange0TB;
        private System.Windows.Forms.CheckBox saveTressCB;
        private System.Windows.Forms.CheckBox runOneTreeCB;
        private System.Windows.Forms.CheckBox estimateRFonTrainingCB;
        private System.Windows.Forms.CheckBox runOneTreeOnTtrainingCB;
        private System.Windows.Forms.CheckBox estimateRFnoVotingCB;
        private System.Windows.Forms.CheckBox estimateRFwaveletsCB;
        private System.Windows.Forms.CheckBox BaggingWithRepCB;
        private System.Windows.Forms.CheckBox sparseRfCB;
        private System.Windows.Forms.TextBox sparseRfTB;
        private System.Windows.Forms.TextBox boundDepthTB;
        private System.Windows.Forms.Label label25;
    }
}


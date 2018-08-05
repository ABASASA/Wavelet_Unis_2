namespace DataScienceAnalysis
{
    partial class DecisionTreeForm
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
            this.label3 = new System.Windows.Forms.Label();
            this.textBox3 = new System.Windows.Forms.TextBox();
            this.btnRun = new System.Windows.Forms.Button();
            this.label2 = new System.Windows.Forms.Label();
            this.textBox2 = new System.Windows.Forms.TextBox();
            this.cbApprox = new System.Windows.Forms.ComboBox();
            this.cbMethod = new System.Windows.Forms.ComboBox();
            this.tbResolution = new System.Windows.Forms.TextBox();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.tbApproxThreshol = new System.Windows.Forms.TextBox();
            this.lblApproxThreshol = new System.Windows.Forms.Label();
            this.lblMethod = new System.Windows.Forms.Label();
            this.lblApprox = new System.Windows.Forms.Label();
            this.lblDomainExtantion = new System.Windows.Forms.Label();
            this.tbDomainExt = new System.Windows.Forms.TextBox();
            this.lblResolution = new System.Windows.Forms.Label();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.label1 = new System.Windows.Forms.Label();
            this.textBox4 = new System.Windows.Forms.TextBox();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.groupBox3.SuspendLayout();
            this.SuspendLayout();
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(13, 51);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(88, 13);
            this.label3.TabIndex = 19;
            this.label3.Text = "number of boosts";
            // 
            // textBox3
            // 
            this.textBox3.Location = new System.Drawing.Point(114, 48);
            this.textBox3.Name = "textBox3";
            this.textBox3.Size = new System.Drawing.Size(68, 20);
            this.textBox3.TabIndex = 18;
            // 
            // btnRun
            // 
            this.btnRun.Location = new System.Drawing.Point(258, 236);
            this.btnRun.Name = "btnRun";
            this.btnRun.Size = new System.Drawing.Size(261, 99);
            this.btnRun.TabIndex = 17;
            this.btnRun.Text = "Run !!!";
            this.btnRun.UseVisualStyleBackColor = true;
            this.btnRun.Click += new System.EventHandler(this.btnRun_Click);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(16, 46);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(65, 13);
            this.label2.TabIndex = 16;
            this.label2.Text = "num of trees";
            // 
            // textBox2
            // 
            this.textBox2.Location = new System.Drawing.Point(87, 43);
            this.textBox2.Name = "textBox2";
            this.textBox2.Size = new System.Drawing.Size(43, 20);
            this.textBox2.TabIndex = 15;
            // 
            // cbApprox
            // 
            this.cbApprox.FormattingEnabled = true;
            this.cbApprox.Items.AddRange(new object[] {
            "Constants",
            "Linear"});
            this.cbApprox.Location = new System.Drawing.Point(88, 67);
            this.cbApprox.Name = "cbApprox";
            this.cbApprox.Size = new System.Drawing.Size(103, 21);
            this.cbApprox.TabIndex = 20;
            this.cbApprox.Text = "Constants";
            // 
            // cbMethod
            // 
            this.cbMethod.FormattingEnabled = true;
            this.cbMethod.Items.AddRange(new object[] {
            "Waveletes",
            "Geometric Waveletes"});
            this.cbMethod.Location = new System.Drawing.Point(88, 28);
            this.cbMethod.Name = "cbMethod";
            this.cbMethod.Size = new System.Drawing.Size(100, 21);
            this.cbMethod.TabIndex = 21;
            this.cbMethod.Text = "Waveletes";
            // 
            // tbResolution
            // 
            this.tbResolution.Location = new System.Drawing.Point(88, 114);
            this.tbResolution.Name = "tbResolution";
            this.tbResolution.Size = new System.Drawing.Size(52, 20);
            this.tbResolution.TabIndex = 23;
            this.tbResolution.Text = "100";
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.tbApproxThreshol);
            this.groupBox1.Controls.Add(this.lblApproxThreshol);
            this.groupBox1.Controls.Add(this.lblMethod);
            this.groupBox1.Controls.Add(this.lblApprox);
            this.groupBox1.Controls.Add(this.lblDomainExtantion);
            this.groupBox1.Controls.Add(this.tbDomainExt);
            this.groupBox1.Controls.Add(this.lblResolution);
            this.groupBox1.Controls.Add(this.cbApprox);
            this.groupBox1.Controls.Add(this.cbMethod);
            this.groupBox1.Controls.Add(this.tbResolution);
            this.groupBox1.Location = new System.Drawing.Point(23, 29);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(212, 239);
            this.groupBox1.TabIndex = 27;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Decomposition";
            // 
            // tbApproxThreshol
            // 
            this.tbApproxThreshol.Location = new System.Drawing.Point(88, 184);
            this.tbApproxThreshol.Name = "tbApproxThreshol";
            this.tbApproxThreshol.Size = new System.Drawing.Size(52, 20);
            this.tbApproxThreshol.TabIndex = 33;
            this.tbApproxThreshol.Text = "4";
            // 
            // lblApproxThreshol
            // 
            this.lblApproxThreshol.AccessibleDescription = "the extantion (precent) of the data obtained in training data";
            this.lblApproxThreshol.Location = new System.Drawing.Point(6, 184);
            this.lblApproxThreshol.Name = "lblApproxThreshol";
            this.lblApproxThreshol.Size = new System.Drawing.Size(76, 35);
            this.lblApproxThreshol.TabIndex = 32;
            this.lblApproxThreshol.Tag = "";
            this.lblApproxThreshol.Text = "Approximation threshold";
            // 
            // lblMethod
            // 
            this.lblMethod.AutoSize = true;
            this.lblMethod.Location = new System.Drawing.Point(6, 28);
            this.lblMethod.Name = "lblMethod";
            this.lblMethod.Size = new System.Drawing.Size(43, 13);
            this.lblMethod.TabIndex = 30;
            this.lblMethod.Text = "Method";
            // 
            // lblApprox
            // 
            this.lblApprox.Location = new System.Drawing.Point(6, 62);
            this.lblApprox.Name = "lblApprox";
            this.lblApprox.Size = new System.Drawing.Size(85, 38);
            this.lblApprox.TabIndex = 31;
            this.lblApprox.Text = "Approximation Order";
            // 
            // lblDomainExtantion
            // 
            this.lblDomainExtantion.AccessibleDescription = "the extantion (precent) of the data obtained in training data";
            this.lblDomainExtantion.Location = new System.Drawing.Point(6, 149);
            this.lblDomainExtantion.Name = "lblDomainExtantion";
            this.lblDomainExtantion.Size = new System.Drawing.Size(66, 35);
            this.lblDomainExtantion.TabIndex = 26;
            this.lblDomainExtantion.Tag = "";
            this.lblDomainExtantion.Text = "Domain extantion";
            this.lblDomainExtantion.MouseHover += new System.EventHandler(this.lblDomainExtantion_MouseHover);
            // 
            // tbDomainExt
            // 
            this.tbDomainExt.Location = new System.Drawing.Point(88, 149);
            this.tbDomainExt.Name = "tbDomainExt";
            this.tbDomainExt.Size = new System.Drawing.Size(52, 20);
            this.tbDomainExt.TabIndex = 25;
            this.tbDomainExt.Text = "0.1";
            // 
            // lblResolution
            // 
            this.lblResolution.AccessibleDescription = "number of points in each axe";
            this.lblResolution.Location = new System.Drawing.Point(6, 114);
            this.lblResolution.Name = "lblResolution";
            this.lblResolution.Size = new System.Drawing.Size(66, 35);
            this.lblResolution.TabIndex = 24;
            this.lblResolution.Tag = "";
            this.lblResolution.Text = "Resolution";
            this.lblResolution.MouseHover += new System.EventHandler(this.lblResolution_MouseHover);
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.label1);
            this.groupBox2.Controls.Add(this.textBox4);
            this.groupBox2.Controls.Add(this.label3);
            this.groupBox2.Controls.Add(this.textBox3);
            this.groupBox2.Location = new System.Drawing.Point(258, 29);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(200, 100);
            this.groupBox2.TabIndex = 28;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Boosting";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(13, 75);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(47, 13);
            this.label1.TabIndex = 21;
            this.label1.Text = "lambda0";
            // 
            // textBox4
            // 
            this.textBox4.Location = new System.Drawing.Point(114, 72);
            this.textBox4.Name = "textBox4";
            this.textBox4.Size = new System.Drawing.Size(68, 20);
            this.textBox4.TabIndex = 20;
            // 
            // groupBox3
            // 
            this.groupBox3.Controls.Add(this.label2);
            this.groupBox3.Controls.Add(this.textBox2);
            this.groupBox3.Location = new System.Drawing.Point(484, 29);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Size = new System.Drawing.Size(200, 100);
            this.groupBox3.TabIndex = 29;
            this.groupBox3.TabStop = false;
            this.groupBox3.Text = "Random Forest";
            // 
            // DecisionTreeForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(775, 395);
            this.Controls.Add(this.groupBox3);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.btnRun);
            this.Name = "DecisionTreeForm";
            this.Text = "DecisionTreeForm";
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            this.groupBox3.ResumeLayout(false);
            this.groupBox3.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox textBox3;
        private System.Windows.Forms.Button btnRun;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox textBox2;
        private System.Windows.Forms.ComboBox cbApprox;
        private System.Windows.Forms.ComboBox cbMethod;
        private System.Windows.Forms.TextBox tbResolution;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox textBox4;
        private System.Windows.Forms.GroupBox groupBox3;
        private System.Windows.Forms.Label lblResolution;
        private System.Windows.Forms.Label lblDomainExtantion;
        private System.Windows.Forms.TextBox tbDomainExt;
        private System.Windows.Forms.Label lblMethod;
        private System.Windows.Forms.Label lblApprox;
        private System.Windows.Forms.TextBox tbApproxThreshol;
        private System.Windows.Forms.Label lblApproxThreshol;
    }
}
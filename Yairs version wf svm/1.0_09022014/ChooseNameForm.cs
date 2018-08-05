using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace DataScienceAnalysis
{
    public partial class ChooseNameForm : Form
    {
        public ChooseNameForm()
        {
            InitializeComponent();
        }

        public string myName;

        private void btnChoose_Click(object sender, EventArgs e)
        {
            myName = tbName.Text;
            Close();
        }
    }
}

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.IO;

namespace DataScienceAnalysis
{
    public partial class ManageInputData : Form
    {
        public ManageInputData()
        {
            InitializeComponent();
            load_files_names(Filenames_File);
            tabel_are_set = false;
        }

        //FILE NAMES FILE (STATIC)
        string Filenames_File = "filenames_file.txt";

        //DATA NAMES
        public string training_data_file_name;
        public string training_label_file_name;
        public string testing_data_file_name;
        public string testing_label_file_name;

        public bool tabel_are_set;
        string[] string_seperator = { ","};

        //LOAD FILES NAMES (FOR CASHING)
        private void load_files_names(string filenames_file)
        {
            if (!File.Exists(filenames_file))
                return;

            //READ FROM FILENAMES  FILE
            StreamReader reader = new StreamReader(File.OpenRead(filenames_file));

            //get the first line 
            string line;
            string[] values;

            while (!reader.EndOfStream)
            {
                line = reader.ReadLine();
                values = line.Split(string_seperator, StringSplitOptions.RemoveEmptyEntries); 

                //VERIFY INPUT FILE IS OK AND SET TEXT BOX AND FILE NAMES ACORDING TO FILE
                if(values.Count() != 2)
                    MessageBox.Show("bad input file: " + filenames_file);
                else if(values[0] == "training_data")
                {
                    tb_training_data.Text = values[1];
                    training_data_file_name = values[1];
                }
                else if (values[0] == "training_label")
                {
                    tb_training_label.Text = values[1];
                    training_label_file_name = values[1];
                }
                else if (values[0] == "testing_data")
                {
                    tb_testing_data.Text = values[1];
                    testing_data_file_name = values[1];
                }
                else if (values[0] == "testing_label")
                {
                    tb_testing_label.Text = values[1];
                    testing_label_file_name = values[1];
                }
            }
            
            //CLOSE
            reader.Close();
        
        }

        //SAVE TEXT FROM TEXT BOXES INTO FILE (FOR CASHING)
        private void save_files_names2file(string filenames_file)
        {
            StreamWriter writer = new StreamWriter(filenames_file, false);

            if (tb_training_data.Text != "")
                writer.WriteLine("training_data," + tb_training_data.Text);
            if (tb_training_label.Text != "")
                writer.WriteLine("training_label," + tb_training_label.Text);
            if (tb_training_label.Text != "")
                writer.WriteLine("testing_data," + tb_testing_data.Text);
            if (tb_training_label.Text != "")
                writer.WriteLine("testing_label," + tb_testing_label.Text);
            writer.Close();      
        }

        //LOAD FILE PATHS CLICKS
        private void btnTrainingLabel_Click(object sender, EventArgs e)
        {
            openFileDialog1.Title = "Select a training label file";
            DialogResult dlg = openFileDialog1.ShowDialog();
            tb_training_label.Text = openFileDialog1.FileName;
        }

        private void btnTestingData_Click(object sender, EventArgs e)
        {
            openFileDialog1.Title = "Select a testing data file";
            DialogResult dlg = openFileDialog1.ShowDialog();
            tb_testing_data.Text = openFileDialog1.FileName;
        }

        private void btnTestinglabel_Click(object sender, EventArgs e)
        {
            openFileDialog1.Title = "Select a testing label file";
            DialogResult dlg = openFileDialog1.ShowDialog();
            tb_testing_label.Text = openFileDialog1.FileName;
        }

        private void btnTrainingData_Click(object sender, EventArgs e)
        {
            openFileDialog1.Title = "Select a training data file";
            DialogResult dlg = openFileDialog1.ShowDialog();
            tb_training_data.Text = openFileDialog1.FileName;
        }

        //SET CLICK
        private void lblSetAndReadFiles_Click(object sender, EventArgs e)
        {
            SetAndReadFiles();
            DialogResult = DialogResult.OK;
        }

        public void SetAndReadFiles()
        {
            //SAVE TO FILENAMES
            save_files_names2file(Filenames_File);

            //SET NAMES
            training_data_file_name = tb_training_data.Text;
            training_label_file_name = tb_training_label.Text;
            testing_data_file_name = tb_testing_data.Text;
            testing_label_file_name = tb_testing_label.Text;
            tabel_are_set = true;
            Close();        
        }

    }
}

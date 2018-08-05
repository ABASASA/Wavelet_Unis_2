using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Data;
using System.IO;
using System.Windows.Forms;
using System.Threading.Tasks;
//using Amazon.S3.IO;

namespace DataScienceAnalysis
{
    public class DB
    {
        public const int Lower = 0;
        public const int Upper = 1;
        public const int Value = 0;
        public const int Ind = 1;

        public double[][] training_dt;        //original input training data (table)
        public double[][] testing_dt;         //original input testing data (table)
        public double[][] validation_dt;         //original input - subset of the testing data (table)
        public double[][] training_label;  //original input training labels (table)
        public double[][] testing_label;   //original input testing labels (table)
        public double[][] validation_label;   //original input -  subset of the testing labels (table)

        public double[][] PCAtraining_dt;        //original input training data (table)
        public double[][] PCAtesting_dt;         //original input testing data (table)
        public double[][] PCAvalidation_dt;         //original input testing data (table)

        public long[][] PCAtraining_GridIndex_dt;        //for each PCAtraining_dt save the grid index point to its left (smaller grid point)

        public string[] seperator = { " ", ";", "\t", "/n", "," };


        public double[][] getDataTable(string filename)
        {
            StreamReader reader;
            long lineCount=0;
           /* if (Form1.UseS3)
            {*/
                /*string dir_name = Path.GetDirectoryName(filename);
                string file_name = Path.GetFileName(filename);

                S3DirectoryInfo s3dir = new S3DirectoryInfo(Form1.S3client, Form1.bucketName, dir_name);
                S3FileInfo artFile = s3dir.GetFile(file_name);

                
                if (artFile.Exists == false && file_name.Contains("Valid"))
                {
                    string tmp = file_name.Replace("Valid", "testing");
                    artFile = s3dir.GetFile(tmp);
                }

                bool faileEread = true;
                reader = null;
                while (faileEread)
                {
                    try 
                    {
                        reader = artFile.OpenText();
                        while (!reader.EndOfStream)
                        {
                            reader.ReadLine();
                            lineCount++;
                        }
                        reader.DiscardBufferedData();
                        reader.BaseStream.Seek(0, SeekOrigin.Begin);
                        reader.BaseStream.Position = 0;

                        faileEread = false;
                    }
                    catch 
                    {
                        faileEread = true;
                    }
                }

              */
            //}
           // else 
           // {
                if (!File.Exists(filename))//IF NO VALID EXISTS - TRY WITH TEST
                    filename = filename.Replace("Valid", "testing");
                reader = new StreamReader(File.OpenRead(filename));
                lineCount = File.ReadAllLines(filename).Length;
           // }


            //GET THE FIRST LINE 
            string line = reader.ReadLine();
            string[] values = line.Split(seperator, StringSplitOptions.RemoveEmptyEntries);

            //IF NO VALUES ALERT
            if (values.Count() < 1)
                return null;

            double[][] dt = new double[lineCount][];
            dt[0] = new double[values.Count()];
            for (int j = 0; j < values.Count(); j++)
                dt[0][j] = double.Parse(values[j]);

            //SET VALUES TO TABLE
            int counter = 1;
            while (!reader.EndOfStream)
            {
                line = reader.ReadLine();
                values = line.Split(seperator, StringSplitOptions.RemoveEmptyEntries);
                dt[counter] = new double[values.Count()];
                for (int j = 0; j < values.Count(); j++)
                    dt[counter][j] = double.Parse(values[j]);
                counter++;
            }

            reader.Close();

            return dt;
        }

        /*public double[][] getDataTableWithNan(string filename, ref Dictionary<Tuple<int, int>, bool> naTable)
        {            
            StreamReader reader;
            long lineCount = 0;
            List<double> artVal = new List<double>();
            List<int> emptyVal = new List<int>();
            string tmpline = "";

            #region first loop
            if (Form1.UseS3)
            {
                string dir_name = Path.GetDirectoryName(filename);
                string file_name = Path.GetFileName(filename);

                S3DirectoryInfo s3dir = new S3DirectoryInfo(Form1.S3client, Form1.bucketName, dir_name);
                S3FileInfo artFile = s3dir.GetFile(file_name);


                if (artFile.Exists == false && file_name.Contains("Valid"))
                {
                    string tmp = file_name.Replace("Valid", "testing");
                    artFile = s3dir.GetFile(tmp);
                }

                bool faileEread = true;
                reader = null;
                while (faileEread)
                {
                    try
                    {
                        reader = artFile.OpenText();
                        while (!reader.EndOfStream)
                        {
                            tmpline = reader.ReadLine();
                            lineCount++;

                            //***********************
                            //haqndle missing values
                            if (lineCount == 1)//first read
                            { 
                                string[] tmpvalues = tmpline.Split(seperator, StringSplitOptions.RemoveEmptyEntries);
                                for (int i = 0; i < tmpvalues.Count(); i++ )
                                {
                                    double tmpDouble;
                                    if (double.TryParse(tmpvalues[i], out tmpDouble))
                                        artVal.Add(tmpDouble);
                                    else
                                    {
                                        artVal.Add(0);
                                        emptyVal.Add(i);
                                    }
                                }
                            }
                            if (emptyVal.Count > 0 ) // not all artVal values are set - at least once...
                            {
                                string[] tmpvalues = tmpline.Split(seperator, StringSplitOptions.RemoveEmptyEntries);
                                for (int i = 0; i < emptyVal.Count(); i++)
                                {
                                    double tmpDouble;
                                    if (double.TryParse(tmpvalues[emptyVal[i]], out tmpDouble))
                                    {
                                        artVal[emptyVal[i]] = tmpDouble;
                                        emptyVal.RemoveAt(i);
                                    }
                                }                                
                            }
                            //***********************
                        }
                        reader.DiscardBufferedData();
                        reader.BaseStream.Seek(0, SeekOrigin.Begin);
                        reader.BaseStream.Position = 0;

                        faileEread = false;
                    }
                    catch
                    {
                        faileEread = true;
                    }
                }
            }
            else
            {
                if (!File.Exists(filename))//IF NO VALID EXISTS - TRY WITH TEST
                    filename = filename.Replace("Valid", "testing");
                reader = new StreamReader(File.OpenRead(filename));
                lineCount = File.ReadAllLines(filename).Length;

                StreamReader tmpreader = new StreamReader(File.OpenRead(filename));
                bool stoploop = false;
                bool firstLine = true;
                while (!tmpreader.EndOfStream && !stoploop)
                {
                    tmpline = tmpreader.ReadLine();
                    if (firstLine)//first read
                    {
                        firstLine = false;
                        string[] tmpvalues = tmpline.Split(seperator, StringSplitOptions.RemoveEmptyEntries);
                        for (int i = 0; i < tmpvalues.Count(); i++)
                        {
                            double tmpDouble;
                            if (double.TryParse(tmpvalues[i], out tmpDouble))
                                artVal.Add(tmpDouble);
                            else
                            {
                                artVal.Add(0);
                                emptyVal.Add(i);
                            }
                        }
                    }
                    if (emptyVal.Count > 0 ) // not all artVal values are set - at least once...
                    {
                        string[] tmpvalues = tmpline.Split(seperator, StringSplitOptions.RemoveEmptyEntries);
                        for (int i = 0; i < emptyVal.Count(); i++)
                        {
                            double tmpDouble;
                            if (double.TryParse(tmpvalues[emptyVal[i]], out tmpDouble))
                            {
                                artVal[emptyVal[i]] = tmpDouble;
                                emptyVal.RemoveAt(i);
                            }
                        }
                    }
                    //***********************
                    if (emptyVal.Count < 1)
                        stoploop = true;
                }
                tmpreader.Close();
            }
            #endregion
            if(emptyVal.Count > 0)
                MessageBox.Show("there is a column with only Nan values - stop and repair");


            //GET THE FIRST LINE 
            string line = reader.ReadLine();
            string[] values = line.Split(seperator, StringSplitOptions.RemoveEmptyEntries);

            //IF NO VALUES ALERT
            if (values.Count() < 1)
                return null;

            double[][] dt = new double[lineCount][];
            dt[0] = new double[values.Count()];
            for (int j = 0; j < values.Count(); j++)
            {
                double tmpDouble = 0;
                if (double.TryParse(values[j], out tmpDouble))
                    dt[0][j] = double.Parse(values[j]);
                else
                {
                    dt[0][j] = artVal[j];
                    naTable.Add(new Tuple<int, int>(0, j), true);
                }
            }

            //SET VALUES TO TABLE
            int counter = 1;
            while (!reader.EndOfStream)
            {
                line = reader.ReadLine();
                values = line.Split(seperator, StringSplitOptions.RemoveEmptyEntries);
                dt[counter] = new double[values.Count()];
                for (int j = 0; j < values.Count(); j++)
                {
                    double tmpDouble = 0;
                    if (double.TryParse(values[j], out tmpDouble))
                        dt[counter][j] = double.Parse(values[j]);
                    else 
                    {
                        dt[counter][j] = artVal[j];
                        naTable.Add(new Tuple<int,int>(counter,j), true);
                    }
                }
                counter++;
            }

            reader.Close();

            return dt;
        }*/

        public void WriteDataTable(double[][] dt, string datafileName)
        {
            StreamWriter writer = new StreamWriter(datafileName, false);

            int rows = dt.Count();
            int cols = dt[0].Count();

            //WRITE 
            string line = "";
            for (int i = 0; i < rows; i++)
            {
                line = "";
                for (int j = 0; j < cols; j++)
                    line += dt[i][j].ToString() + " ";
                writer.WriteLine(line);
            }
            writer.Close();
        }

        public double[][] getboundingBox(double[][] dt)//+10% for each side
        {
            int Nrow = dt.Count();
            int Ncol = dt[0].Count();

            double[][] BB = new double[2][];
            BB[0] = new double[Ncol];
            BB[1] = new double[Ncol];
            Helpers.applyFor(0, Ncol, i =>
            {
                BB[0][i] = Enumerable.Range(0, Nrow).Select(k => dt[k][i]).Min();
                BB[1][i] = Enumerable.Range(0, Nrow).Select(k => dt[k][i]).Max();
            });
            //EXPEND 10% - TBD CHOOSE FROM OUTSIDE
            for (int i = 0; i < Ncol ; i++)
            {
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (BB[0][i] == BB[1][i]) continue;
                double upper = BB[1][i];
                double lower = BB[0][i];

                BB[0][i] -= 0.1 * (upper - lower);
                BB[1][i] += 0.1 * (upper - lower);
            }

            return BB;
        }

        public List<List<double>> getMainGrid(double[][] dt, double[][] box, ref long[][] dt_grid_indexing)
        {
            List<List<double>> MainGrid = new List<List<double>>();
            int Nrow = dt.Count();
            int Ncol = dt[0].Count();

            for (int i = 0; i < Ncol; i++)
            {
                List<double> lst = new List<double> {box[0][i]};
                //insert first boundery to each list - I can add that if upper and lower bounds are equal - don't add..
                MainGrid.Add(lst);
            }
            for (int i = 0; i < Ncol; i++)
            {
                SetMaingrid(ref MainGrid, i, Nrow, dt, box, ref  dt_grid_indexing);
            }
            

            return MainGrid;
        }

        //private double getMinGap(double[][] dt, int colIndex)
        //{
        //    double esilon_machine = 0.000000000001;
        //    List<double> rowArr = new List<double>();
        //    for (int i = 0; i < dt.Count(); i++)
        //    {
        //        rowArr.Add(dt[i][colIndex]);
        //    }
        //    Array.Sort(rowArr.ToArray());
        //    rowArr = rowArr.OrderBy(o => o).ToList();//see if not sorted by norm already...

        //    double MinVal = double.MaxValue;
        //    for (int i = 0; i < rowArr.Count - 1; i++)
        //        if (Math.Abs(rowArr[i + 1] - rowArr[i]) < MinVal && Math.Abs(rowArr[i + 1] - rowArr[i]) > esilon_machine)
        //            MinVal = Math.Abs(rowArr[i + 1] - rowArr[i]);
        //    return MinVal;

        //}

        public void SetMaingrid(ref List<List<double>> MainGrid, int dim, int Nrow, double[][] dt, double[][] box, ref long[][] dt_grid_indexing)
        {
            //if feature is empty - dont set points to it ...
            if (box[Upper][dim] == box[Lower][dim])
                return;

            List<double[]> val_index_arr = new List<double[]>();
            for (int k = 0; k < dt.Count(); k++)
            {
                double[] pair = new double[2];
                pair[Value] = dt[k][dim];
                pair[Ind] = k;
                val_index_arr.Add(pair);
            }
            //sort by value pairs <value,index> at "dim" dimention
            val_index_arr = val_index_arr.OrderBy(t => t[Value]).ToList();

            double[][] sortedlist = dt.OrderBy(t => t[dim]).ToArray();
            
            //save index of smallest element (first elem in sorted list)
            long indexOfSmalest = Convert.ToInt64(val_index_arr[0][Ind]);
            dt_grid_indexing[indexOfSmalest][dim] = MainGrid[dim].Count-1;//index

            for (int j = 1; j < Nrow; j++)
            {
               
                if (sortedlist[j][dim] != sortedlist[j - 1][dim])
                {
                    //add median to grid at "dim" dimention
                    MainGrid[dim].Add(0.5 * (sortedlist[j - 1][dim] + sortedlist[j][dim])); 
                }
                //save position at the grid by row original id
                long jPointIndex = Convert.ToInt64(val_index_arr[j][Ind]);
                dt_grid_indexing[jPointIndex][dim] = MainGrid[dim].Count - 1;//index
            }

       
            MainGrid[dim].Add(box[1][dim]);   
            for (int j = MainGrid[dim].Count() - 1; j > 0; j--)
                if (MainGrid[dim][j] == MainGrid[dim][j - 1])
                {
                    MainGrid[dim].RemoveAt(j);
                }

        }

        //public static bool IsPntInsideBox(double[][] Box, double[] pnt)
        //{
        //    for (int i = 0; i < pnt.Count(); i++)
        //    {
        //        if (pnt[i] < Box[0][i] || pnt[i] > Box[1][i])
        //            return false;
        //    }
        //    return true;
        //}

        public static bool IsPntInsideBox(int[][] BoxOfIndeces, double[] pnt, int dim)
        {
            for (int i = 0; i < dim; i++)
            {
                if (pnt[i] == 55555.66666)//NA ELEMENT
                    continue;
                if (pnt[i] < Form1.MainGrid[i][BoxOfIndeces[0][i]] || pnt[i] > Form1.MainGrid[i][BoxOfIndeces[1][i]])
                    return false;
            }
            return true;
        }

        //public static void ProjectPntInsideBox(double[][] Box, ref double[] pnt)
        //{
        //    for (int i = 0; i < pnt.Count(); i++)
        //    {
        //        if (pnt[i] < Box[0][i])
        //            pnt[i] = Box[0][i];
        //        if (pnt[i] > Box[1][i])
        //            pnt[i] = Box[1][i];
        //    }
        //}

        public static void ProjectPntInsideBox(int[][] BoxOfIndeces, ref double[] pnt)
        {
            for (int i = 0; i < pnt.Count(); i++)
            {
                if (pnt[i] < Form1.MainGrid[i][BoxOfIndeces[0][i]])
                    pnt[i] = Form1.MainGrid[i][BoxOfIndeces[0][i]];
                if (pnt[i] > Form1.MainGrid[i][BoxOfIndeces[1][i]])
                    pnt[i] = Form1.MainGrid[i][BoxOfIndeces[1][i]];
            }
        }

       /* public double[][] getDataTableTMP(string filename)
        {
            StreamReader reader;
            long lineCount = 0;
            if (Form1.UseS3)
            {
                string dir_name = Path.GetDirectoryName(filename);
                string file_name = Path.GetFileName(filename);

                S3DirectoryInfo s3dir = new S3DirectoryInfo(Form1.S3client, Form1.bucketName, dir_name);
                S3FileInfo artFile = s3dir.GetFile(file_name);


                if (artFile.Exists == false && file_name.Contains("Valid"))
                {
                    string tmp = file_name.Replace("Valid", "testing");
                    artFile = s3dir.GetFile(tmp);
                }

                bool faileEread = true;
                reader = null;
                while (faileEread)
                {
                    try
                    {
                        reader = artFile.OpenText();
                        while (!reader.EndOfStream)
                        {
                            reader.ReadLine();
                            lineCount++;
                        }
                        reader.DiscardBufferedData();
                        reader.BaseStream.Seek(0, SeekOrigin.Begin);
                        reader.BaseStream.Position = 0;

                        faileEread = false;
                    }
                    catch
                    {
                        faileEread = true;
                    }
                }

                //reader = artFile.OpenText();
                //reader = artFile.OpenRead(); 
                //while (!reader.EndOfStream) 
                //{
                //    reader.ReadLine();
                //    lineCount++;
                //}
                //reader.DiscardBufferedData();
                //reader.BaseStream.Seek(0, SeekOrigin.Begin);
                //reader.BaseStream.Position = 0;
            }
            else
            {
                if (!File.Exists(filename))//IF NO VALID EXISTS - TRY WITH TEST
                    filename = filename.Replace("Valid", "testing");
                reader = new StreamReader(File.OpenRead(filename));
                lineCount = File.ReadAllLines(filename).Length;
            }


            //GET THE FIRST LINE 
            string line = reader.ReadLine();
            string[] values = line.Split(seperator, StringSplitOptions.RemoveEmptyEntries);

            //IF NO VALUES ALERT
            if (values.Count() < 1)
                return null;

            double[][] dt = new double[lineCount][];
            dt[0] = new double[values.Count()];
            for (int j = 0; j < values.Count(); j++)
                dt[0][j] = double.Parse(values[j]);

            //SET VALUES TO TABLE
            int counter = 1;
            while (!reader.EndOfStream)
            {
                line = reader.ReadLine();
                values = line.Split(seperator, StringSplitOptions.RemoveEmptyEntries);
                dt[counter] = new double[values.Count()];
                for (int j = 0; j < values.Count(); j++)
                { 
                    double tmp;
                    if (double.TryParse(values[j], out tmp))
                        dt[counter][j] = tmp;
                    else
                        dt[counter][j] = -1;
                } 
                counter++;
            }

            reader.Close();

            return dt;
        }*/
    }
}

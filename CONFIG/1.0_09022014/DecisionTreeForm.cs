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
    public partial class DecisionTreeForm : Form
    {
        public DecisionTreeForm(DataTable _trainingData, DataTable _trainingLabel)
        {
            InitializeComponent();
            trainingData  = _trainingData;
            trainingLabel = _trainingLabel;
            dataDim = trainingData.Columns.Count;
            BoundingBox = new double[2, dataDim];//[min/max, dim]
            segments = new double[dataDim];//[min/max, dim]
            segments_step = new double[dataDim];//[min/max, dim]
            labelDim = trainingLabel.Columns.Count;
        }

        #region parameters


        DataTable trainingData;
        DataTable trainingLabel;
        
        public static double[,] BoundingBox;//the domain of the training data
        public static double[] segments;//LENGTH OF EACH SIDE OF THE BOX
        public static double[] segments_step;//segments/resolution
        public static int dataDim;
        public static int labelDim;
        public static double resolution;
        public static int approxOrder;
        public static string method;
        public static double ApproximationThreshold;
        double domain_extantion;

        #endregion

        private void setsegments(double[] segments, double[,] box)
        {
            for (int i = 0; i < box.GetLength(1); i++)
            {
                segments[i] = box[1, i] - box[0, i];
                segments_step[i] = segments[i] / resolution;
                if (box[1, i] <= box[0, i])
                    MessageBox.Show("segment size can't be negitive or zero");
            }
        }

        //GET BOUNDING BOX
        private void getBoundingBox(DataTable dt, double[,] box)
        {
            int dim = dt.Columns.Count;// box.GetLength(1)

            box[0, 0] = double.MaxValue;//set min with max
            box[1, 0] = double.MinValue; //set max with min

            foreach (DataRow dr in dt.Rows)
            {
                for (int i = 0; i < dim; i++)
                {
                    double value = double.Parse(dr[i].ToString());
                    box[0, i] = Math.Min(box[0, i], value);
                    box[1, i] = Math.Max(box[1, i], value);
                }                    
            }   
        }

        //extendBox        
        private void extendBox(double[,] box, double extention_precent)
        {
            int dim = box.GetLength(1);

            for (int i = 0; i < dim; i++)
            {
                double length = box[1, i] - box[0, i];
                box[0, i] = box[0, i] - length * extention_precent;
                box[1, i] = box[1, i] + length * extention_precent;
            }
        }
        
        //INIT BOUNDING SURFACES FROM BOUNDING BOX
        private void InitBoundingSurfacesFromBoundingBox(double[,] Box, List<Surface> Bounding_surface_arr)
        {
            int dim = Box.GetLength(1);
            double[] center_pnt = new double[dim];//create a point of box dimantion
            double[][] normal_array = new double[2*dim][];//[INDEX,VALUES] - 2*DIM ARE FOR THE TWO SIDES OF EACH DIM (min - even, max - odd)

            //SET CENTER OF MASS IN EACH DIMANTION
            for (int i = 0; i < dim; i++)
            {
                if (Box[1, i] == Box[0, i])
                    MessageBox.Show("the cordinate" + i.ToString() +"is of size 0 - stop and cheack");

                center_pnt[i] = 0.5 * (Box[1, i] + Box[0, i]);

                //ALLOC TWO ARRAYS
                normal_array[2*i] = new double[dim];
                normal_array[2*i+1] = new double[dim];
                
                //FOR EACH DIM SET 2 NORMALS IN THE EXTERNAL DIRECTION
                for (int j = 0; j < dim; j++)
                {
                    if(i==j)
                    {
                        normal_array[2*i][j]    = -1;
                        normal_array[2*i +1][j] = 1;                    
                    }
                    else
                    {
                        normal_array[2*i][j]    = 0;
                        normal_array[2*i +1][j] = 0;                           
                    }
                }
            }

            //SET SURFACES
            for (int i = 0; i < dim; i++)
            {
                //SET LOWER SURFACE 
                double[] point_low = new double[center_pnt.Count()];
                center_pnt.CopyTo(point_low,0);

                point_low[i] = Box[0, i];
                double[] normal_low = new double[normal_array[2 * i].Count()];
                normal_array[2 * i].CopyTo(normal_low, 0);
                Surface surface_low = new Surface(point_low, normal_low);
                Bounding_surface_arr.Add(surface_low);

                //SET higher SURFACE 
                double[] point_high = new double[center_pnt.Count()];
                center_pnt.CopyTo(point_high, 0);
                point_high[i] = Box[1, i];
                double[] normal_high = new double[normal_array[2 * i + 1].Count()];
                normal_array[2 * i + 1].CopyTo(normal_high, 0);
                Surface surface_high = new Surface(point_high, normal_high);
                Bounding_surface_arr.Add(surface_high);
            }
        }
        
        private void DecomposeData(DataTable inputData, DataTable labels, double[,] boundingBox)
        {
            getBoundingBox(inputData, boundingBox);
            extendBox(boundingBox, domain_extantion);

            //SET SURFACEARR
            List<Surface> SurfaceArr = new List<Surface>();
            InitBoundingSurfacesFromBoundingBox(boundingBox, SurfaceArr);
            setsegments(segments, boundingBox);

            //SET ROOT WAVELETE
            GeoWave gwRoot = new GeoWave();
            gwRoot.SurfaceArr = Surface.CopyList(SurfaceArr);

            Form1.setMatrix(gwRoot.boubdingBox, boundingBox);

            //SET REGION POINTS IDS
            for (int i = 0; i < inputData.Rows.Count; i++)
                gwRoot.pointsIdArray.Add(i);

            //BSP
            BSP bsp = new BSP(inputData, labels, SurfaceArr);
            bsp.start(gwRoot);
        }

        public void Run()
        {
            //SET INPUT CHARACTERISTICS
            method = cbMethod.Text;
            if (cbApprox.Text == "Constants")
                approxOrder = 0;
            else if (cbApprox.Text == "Linear")
                approxOrder = 1;
            else
                MessageBox.Show("approximation order " + cbApprox.Text + " is not supported");

            resolution = Int32.Parse(tbResolution.Text);
            domain_extantion = double.Parse(tbDomainExt.Text);
            ApproximationThreshold = double.Parse(tbApproxThreshol.Text);

            //OPTIONAL TILING
            DecomposeData(trainingData, trainingLabel, BoundingBox);        
        }

        private void btnRun_Click(object sender, EventArgs e)
        {
            Run();
        }

        private void lblResolution_MouseHover(object sender, EventArgs e)
        {
            System.Windows.Forms.ToolTip ToolTip1 = new System.Windows.Forms.ToolTip();
            ToolTip1.SetToolTip(this.lblResolution, this.lblResolution.AccessibleDescription);
        }

        private void lblDomainExtantion_MouseHover(object sender, EventArgs e)
        {
            System.Windows.Forms.ToolTip ToolTip1 = new System.Windows.Forms.ToolTip();
            ToolTip1.SetToolTip(this.lblDomainExtantion, this.lblDomainExtantion.AccessibleDescription);
        }

    }
}


////EXTEND BOUNDING BOX
//private void getextendBoundingBox(double[,] in_box, double[,] extended_box, double extention_precent)
//{
//    int dim = in_box.GetLength(1);

//    for (int i = 0; i < dim; i++)
//    {
//        double length = in_box[1, i] - in_box[0, i];
//        extended_box[0, i] = in_box[0, i] - length * extention_precent;
//        extended_box[1, i] = in_box[1, i] + length * extention_precent;
//    }
//}
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Accord.Math.Optimization;
using System.Data;
using Accord.Math;
using Accord.Math.Decompositions;
using System.Windows.Forms;
//using Math;


namespace DataScienceAnalysis
{
    public class Surface 
    {
        public Surface(double[] Point, double[] Normal)
        {
            normal = new double[Normal.Count()];
            pnt = new double[Normal.Count()];
            Point.CopyTo(pnt,0);
            Normal.CopyTo(normal, 0); 
        }

        public Surface()
        {
            normal = new double[DecisionTreeForm.dataDim];
            pnt = new double[DecisionTreeForm.dataDim];
        }

        public Surface(Surface surface)
        {
            normal = new double[surface.normal.Count()];
            pnt = new double[surface.pnt.Count()];
            surface.pnt.CopyTo(pnt, 0);
            surface.normal.CopyTo(normal, 0); 
        }

        public double[] normal;
        public double[] pnt;
        public bool IsPntBelow(double[] point)
        {
            double[] tmp_pnt = new double[point.Count()];
            pointnD.minus(tmp_pnt, point, pnt);
            if (pointnD.dot(normal, tmp_pnt) > 0) //NO NEED TO NORMALIZE - WE NEED THE SIGN
                return false;
            return true;
        }

        public static List<Surface> CopyList(List<Surface> list)
        {
            List<Surface> newList = new List<Surface>(list.Count);
            foreach (Surface item in list)
                newList.Add(item.DeepClone());
            return newList;
        }         
        
        public  Surface DeepClone()
        {
            Surface surf = new Surface(pnt, normal);
            return surf;
        }
    }

    public class pointnD
    {
        public pointnD(double[] pnt)
        {
            cordinates = new double[pnt.Count()];
            pnt.CopyTo(cordinates, 0);
        }

        public pointnD(int dim)
        {
            cordinates = new double[dim];
        }

        public pointnD(){}//defualt constructor

        //public static void setArray(double[] dst_array, double[] src_array)
        //{
        //    for (int i = 0; i < src_array.Count(); i++)
        //        dst_array[i] = src_array[i];
        //}
        
        //public static void scalarMult(double[] dst_array, double[] src_array, double scalar)
        //{
        //    for (int i = 0; i < src_array.Count(); i++)
        //        dst_array[i] = src_array[i] * scalar;
        //}
        
        public double[] cordinates;
        double l2_norm;
        double l2_norm_calc()
        {
            double sum2=0;
            for (int i=0 ; i< cordinates.Count(); i++)
            {
                sum2 += cordinates[i]*cordinates[i];
            }
            return Math.Sqrt(sum2);
        }
        public static double l2_norm_calc(double[] pnt)
        {
            double sum2 = 0;
            for (int i = 0; i < pnt.Count(); i++)
            {
                sum2 += pnt[i] * pnt[i];
            }
            return Math.Sqrt(sum2);
        }
        void normalize()
        {
            double norm = l2_norm_calc();
            for (int i = 0; i < cordinates.Count(); i++)
            {
                cordinates[i] /= norm;
            }
        }

        public static void normalize(double[] pnt)
        {
            double norm = l2_norm_calc(pnt);
            for (int i = 0; i < pnt.Count(); i++)
            {
                pnt[i] /= norm;
            }
        }

        public void set(pointnD pnt)
        {
            l2_norm = pnt.l2_norm;
            cordinates = new double[pnt.cordinates.Count()];
            for (int i = 0; i < pnt.cordinates.Count(); i++)
            {
                cordinates[i] = pnt.cordinates[i];
            }
        }
        public static double dot(pointnD pnt1, pointnD pnt2)
        {
            double norm1 = pnt1.l2_norm_calc();
            double norm2 = pnt2.l2_norm_calc();
            double sum2 = 0;
            for (int i = 0; i < pnt1.cordinates.Count(); i++)
            {
                sum2 += pnt1.cordinates[i] * pnt2.cordinates[i] / (norm1 * norm2);
            }
            return sum2; 
        }

        public static double dot(double[] pnt1, double[] pnt2)
        {
            double norm1 = l2_norm_calc(pnt1);
            double norm2 = l2_norm_calc(pnt2);
            double sum2 = 0;
            for (int i = 0; i < pnt1.Count(); i++)
            {
                sum2 += pnt1[i] * pnt2[i] / (norm1 * norm2);
            }
            return sum2;
        }

        public pointnD minus(pointnD pnt)
        {
            pointnD out_pnt = new pointnD();

            for (int i = 0; i < pnt.cordinates.Count(); i++)
            {
                out_pnt.cordinates[i] = cordinates[i] - pnt.cordinates[i];
            }
            return out_pnt;
        }

        public static void minus(double[] result, double[] left, double[] right)
        {
            for (int i = 0; i < result.Count(); i++)
            {
                result[i] = left[i] - right[i];
            }
        }

        //public static void plus(double[] result, double[] left, double[] right)
        //{
        //    for (int i = 0; i < result.Count(); i++)
        //    {
        //        result[i] = left[i] + right[i];
        //    }
        //}
    }

    public class GeoWave 
    {
        public GeoWave()
        { 
            parentID    = -1;
            child0      = -1;
            child1      = -1;
            level       = -1;
            norm        = -1;
            approx_solution = new double[DecisionTreeForm.dataDim, DecisionTreeForm.labelDim];
            boubdingBox = new double[2 , DecisionTreeForm.dataDim];
        }
        
        public GeoWave(double[,] BOX)
        {
            parentID = -1;
            child0 = -1;
            child1 = -1;
            level = -1;
            norm = -1;
            approx_solution = new double[DecisionTreeForm.dataDim, DecisionTreeForm.labelDim];
            boubdingBox = new double[BOX.GetLength(0), BOX.GetLength(1)];
            Form1.setMatrix(boubdingBox, BOX);
        }

        public int parentID, child0, child1, level;
        public List<Surface> SurfaceArr = new List<Surface>();//array of all surface id that bound the GW
        //double[] approxPoly;//column of size n which can multiply n dimentional point to approximate its label/value 
        public double[,] approx_solution;
        public double proximity;
        public double norm;
        public double[,] boubdingBox;
        public List<int> pointsIdArray = new List<int>();//points in regeion (row index of static input data)

        private double SumInnerProductsDataRowWithSolution_(int inputDataRow_index, double[,] ApproximationMatrix)
        {
            if (ApproximationMatrix.GetLength(0) != BSP.inputData.Columns.Count)
            {
                MessageBox.Show("can't use dot if arrays are not in the same size");
                return -999;
            }
            double sum=0;

            for (int j = 0; j < ApproximationMatrix.GetLength(1); j++)
            {
                for (int i = 0; i < ApproximationMatrix.GetLength(0); i++)
                {
                    sum += ApproximationMatrix[i, j] * double.Parse(BSP.inputData.Rows[inputDataRow_index][i].ToString());
                }            
            }
            
            return sum;
        }

        public void computeNorm(GeoWave parent) 
        {
            norm = 0;
            //GO OVER ALL POINTS IN THE REGION
            double[,] diff_matrix = new double[approx_solution.GetLength(0), approx_solution.GetLength(1)];
            for (int i = 0; i < approx_solution.GetLength(0); i++)
            { 
                for (int j = 0; j < approx_solution.GetLength(1); j++)
                {
                    diff_matrix[i,j] = approx_solution[i,j] - parent.approx_solution[i,j];
                }
            }

           for (int i = 0; i < pointsIdArray.Count(); i++)
           {
               norm += Math.Pow(SumInnerProductsDataRowWithSolution_(pointsIdArray[i], diff_matrix), 2);
           }
            //NO NEED TO SQUARE
        }
        public void computeNorm() 
        {
            norm = 0;
            //GO OVER ALL POINTS IN THE REGION
            for (int i = 0; i < pointsIdArray.Count(); i++)
            {
                for (int j = 0; j < approx_solution.GetLength(1); j++)
                {
                    norm += Math.Pow(SumInnerProductsDataRowWithSolution_(pointsIdArray[i], approx_solution), 2);
                }
            }
            //NO NEED TO SQUARE
        }


        //public void addContribution(floatmat  &approxedIamge);
        //publicvoid addContribution(floatmat  &approxedIamge, GeoWave inParent);
	}

    public class SurfaceFactory
    {
        public SurfaceFactory(List<Surface> _SurfaceArr, double[,] boundingBox) 
        { 
            counter = 0; 
            dim_counter = 0;
            SurfaceArr = _SurfaceArr;
            if (DecisionTreeForm.method == "Waveletes")
            {
                resolution_max_steps = new int[boundingBox.GetLength(1)];
                set_resolution_max_steps(resolution_max_steps, boundingBox);
                bounding_box = boundingBox;
            }
        }

        //FOR EACH DIM SET HOW MANY STEPS ARE NEEDED TO COVER IT , USING THE RESOLUTION
        void set_resolution_max_steps(int[] res_max_steps, double[,] boundingBox)
        { 
            for(int i=0; i < res_max_steps.Count(); i++)
            {
                //double[] pnt1 = SurfaceArr[2*i].pnt;
                //double[] pnt2 = SurfaceArr[2*i+1].pnt;
                //double[] diff_pnt = new double[pnt2.Count()];
                //pointnD.minus(diff_pnt, pnt1, pnt2);
                //double length = pointnD.l2_norm_calc(diff_pnt);

                //res_max_steps[i] = int.Parse((length / DecisionTreeForm.segments_step[i]).ToString());
                double length = boundingBox[1, i] - boundingBox[0, i];
                if (length <= 0)
                    MessageBox.Show("the box has negitive size");
                res_max_steps[i] = int.Parse((length / DecisionTreeForm.segments_step[i]).ToString());
            }
        }

        static int counter;
        static List<Surface> SurfaceArr;
        static int dim_counter;
        static int resolution_counter;
        static int[] resolution_max_steps;
        double[,] bounding_box;

        public bool getNextSurface(Surface surface, ref int coordinate_index)
        {
            if (DecisionTreeForm.method == "Waveletes")
            {
                for (; dim_counter < bounding_box.GetLength(1); dim_counter++)
                {
                    for(; resolution_counter < resolution_max_steps[dim_counter] ; )
                    {
                        double[] normal = SurfaceArr[2 * dim_counter].normal;
                        double[] pnt = SurfaceArr[2 * dim_counter].pnt;
                        double[] Shiftpnt = new double[pnt.Count()];

                        //BOUND PNT TO BOX (if not inside - project it)
                        for (int i = 0; i < bounding_box.GetLength(1); i++)
                        { 
                            if(pnt[i] < bounding_box[0,i])
                                pnt[i] = bounding_box[0,i];
                            else if (pnt[i] > bounding_box[1, i])
                                pnt[i] = bounding_box[1, i];
                        }

                            normal.CopyTo(surface.normal, 0);
                        Shiftpnt = normal.Multiply(-1 * resolution_counter * DecisionTreeForm.segments_step[dim_counter]);
                        //pointnD.scalarMult(Shiftpnt, normal, -1 * resolution_counter * DecisionTreeForm.segments_step[dim_counter]);
                        surface.pnt = Shiftpnt.Add(pnt);
                        //pointnD.plus(surface.pnt, Shiftpnt, pnt);
                        
                        resolution_counter++;
                        coordinate_index = dim_counter;
                        return true;
                    }
                    resolution_counter = 0;
                }
                return false;//END OF SUGGESTED SURFACES
            }
            return false;
        }
    }

    public class BSP
    {
        public BSP(DataTable _inputData, DataTable _Labels, List<Surface> _SurfaceArr)
        {
            inputData   = _inputData;//get the data by reference
            Labels      = _Labels;
            dim         = inputData.Columns.Count;
            SurfaceArr  = _SurfaceArr;
        }
        
        public static DataTable inputData;
        public static DataTable Labels;
        public static List<Surface> SurfaceArr;
        public static int dim;
        
        public static List<GeoWave> GeoWaveArr = new List<GeoWave>();

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

        //private void SetMatrix(double[,] dst, double[,] src)
        //{
        //    //SANITY CHECK
        //    if (src.GetLength(0) < 1 || src.GetLength(1) < 1)
        //        MessageBox.Show("data table " + dt.ToString() + " is empty ...");

        //    //SET VALUES
        //    for (int i = 0; i < src.GetLength(0); i++)
        //    {
        //        for (int j = 0; j < src.GetLength(1); j++)
        //        {
        //            dst[i, j] = src[i, j];
        //        }
        //    }
        //}

        private double binary_labeled_return(double val)
        {
            if (val > 0) 
                return 1;
            return -1;            
        }

        private double getErrorSize(double[,] input, double [,] solution, double[,] labels)
        {
            double[,] mult_result = input.Multiply(solution);
            double error_square_sum = 0;
            for (int j = 0; j < mult_result.GetLength(1); j++)
            {
                for (int i = 0; i < mult_result.GetLength(0); i++)
                {
                    error_square_sum += Math.Pow(binary_labeled_return(mult_result[i, j]) - labels[i, j], 2);//consider changing the loos function
                }
            }
            return Math.Sqrt(error_square_sum);
        }
        
        private double find_least_squares_proximity(DataTable _inputData, DataTable _Labels)
        {

            //SingularValueDecomposition svd;
            double[,] matrix = new double[_inputData.Rows.Count,_inputData.Columns.Count];
            SetMatrixFromDataTable(matrix, _inputData);            
            //var svd = new SingularValueDecomposition(matrix);
            //double[,] B_t = B.Transpose();
            double[,] B = new double[_Labels.Rows.Count,_Labels.Columns.Count];
            SetMatrixFromDataTable(B, _Labels);
            //double[,] X = svd.Solve(B);
            double[,] X = matrix.Solve(B);
            double error_size = getErrorSize(matrix, X, B);
            return error_size;
        }

        private double find_least_squares_proximity(double[,] matrix, double[,] B)
        {
            double[,] X = matrix.Solve(B);
            double error_size = getErrorSize(matrix, X, B);
            return error_size;
        }

        private bool calc_least_squares_proximity(DataTable _inputData, DataTable _Labels, GeoWave gw)
        {

            //SingularValueDecomposition svd;
            double[,] matrix = new double[_inputData.Rows.Count, _inputData.Columns.Count];
            SetMatrixFromDataTable(matrix, _inputData);
            //var svd = new SingularValueDecomposition(matrix);
            //double[,] B_t = B.Transpose();
            double[,] B = new double[_Labels.Rows.Count, _Labels.Columns.Count];
            SetMatrixFromDataTable(B, _Labels);
            //double[,] X = svd.Solve(B);
            double[,] X = matrix.Solve(B);

            //X.CopyTo(gw.approx_solution, 0);
            gw.approx_solution = (double[,])X.Clone();

            //SetMatrix(gw.approx_solution,X);

            gw.proximity = getErrorSize(matrix, X, B);
            return (gw.proximity > DecisionTreeForm.ApproximationThreshold);
        }

        private double calc_least_squares_proximity(double[,] matrix, double[,] B, GeoWave gw)
        {
            //gw.approx_solution.Initialize();
            double[,] X = matrix.Solve(B);
            gw.approx_solution = (double[,])X.Clone();
            gw.proximity = getErrorSize(matrix, X, B);
            return (gw.proximity);
        }

        private bool isVolumTooSmall()
        {
            return true;//tbd
        }

        private double evaluateErrorWithPartition(Surface surface, List<int> UppointsIdArray, List<int> LowpointsIdArray, int BSP_Index)
        {
            //SET TOW MATRICES ACCORDING TO POINTS BISECTIONS
            
            //GO OVER ALL POINTS
            int n_points = GeoWaveArr[BSP_Index].pointsIdArray.Count;
            double[,] upperMatrix = new double[n_points,dim];
            double[,] lowerMatrix = new double[n_points,dim];
            double[,] upperB = new double[n_points,Labels.Columns.Count];
            double[,] lowerB = new double[n_points,Labels.Columns.Count];
            int low_row_counter = 0;
            int up_row_counter = 0;
            for (int i = 0; i < n_points; i++)
            { 
                //GET POINT BY INDEX 
                int index = GeoWaveArr[BSP_Index].pointsIdArray[i];
                double[] point = inputData.Rows[index].ToArray();
                double[] lbl = Labels.Rows[index].ToArray();
                
                if (surface.IsPntBelow(point))
                {
                    lowerMatrix.SetRow(low_row_counter, point);
                    lowerB.SetRow(low_row_counter, lbl);
                    LowpointsIdArray.Add(index);
                    low_row_counter++;
                }
                else
                {
                    upperMatrix.SetRow(up_row_counter, point);
                    upperB.SetRow(up_row_counter, lbl);
                    UppointsIdArray.Add(index);
                    up_row_counter++;                    
                }
            }
            
            double lower_error = find_least_squares_proximity(lowerMatrix, lowerB);
            double upper_error = find_least_squares_proximity(upperMatrix, upperB);


            return (lower_error + upper_error);
        }

        private double evaluateErrorWithPartition(Surface surface, GeoWave child0, GeoWave child1, int BSP_Index)
        {
            //SET TOW MATRICES ACCORDING TO POINTS BISECTIONS

            //CLEAR LIST BEFORE ITS NEW SETTING
            child0.pointsIdArray.Clear();
            child1.pointsIdArray.Clear();

            //GO OVER ALL POINTS
            int n_points = GeoWaveArr[BSP_Index].pointsIdArray.Count;
            double[,] upperMatrix = new double[n_points, dim];
            double[,] lowerMatrix = new double[n_points, dim];
            double[,] upperB = new double[n_points, Labels.Columns.Count];
            double[,] lowerB = new double[n_points, Labels.Columns.Count];
            int low_row_counter = 0;
            int up_row_counter = 0;
            for (int i = 0; i < n_points; i++)
            {
                //GET POINT BY INDEX 
                int index = GeoWaveArr[BSP_Index].pointsIdArray[i];

                double[] point = new double[inputData.Columns.Count];
                Form1.setdataRow(inputData.Rows[index], point);
                double[] lbl = new double[Labels.Columns.Count];
                Form1.setdataRow(Labels.Rows[index], lbl);

                if (surface.IsPntBelow(point))
                {
                    lowerMatrix.SetRow(low_row_counter, point);
                    lowerB.SetRow(low_row_counter, lbl);
                    child0.pointsIdArray.Add(index);
                    low_row_counter++;
                }
                else
                {
                    upperMatrix.SetRow(up_row_counter, point);
                    upperB.SetRow(up_row_counter, lbl);
                    child1.pointsIdArray.Add(index);
                    up_row_counter++;
                }
            }

            double lower_error = 0;
            double upper_error = 0;

            if (low_row_counter > 0)
            {
                double[,] lower_matrix = new double[low_row_counter, dim];
                Form1.setMatrix(lower_matrix, lowerMatrix);
                double[,] lower_B = new double[low_row_counter, Labels.Columns.Count];
                Form1.setMatrix(lower_B, lowerB);
                lower_error = calc_least_squares_proximity(lower_matrix, lower_B, child0);
            }

            if (up_row_counter > 0)
            {
                double[,] upper_matrix = new double[up_row_counter, dim];
                Form1.setMatrix(upper_matrix, upperMatrix);
                double[,] upper_B = new double[up_row_counter, Labels.Columns.Count];
                Form1.setMatrix(upper_B, upperB);
                upper_error = calc_least_squares_proximity(upper_matrix, upper_B, child1);
            }

            return (lower_error + upper_error);
        }    

        private void recursiveBSP(int BSP_Index)
        {
            //CALC APPROX_SOLUTION FOR GEO WAVE
            bool ContinueBSP = calc_least_squares_proximity(inputData, Labels, GeoWaveArr[BSP_Index]);
            long debog;
            if (BSP_Index == 10)
                 debog = 15;
            if (ContinueBSP)
            {
                //FIND BEST SURFACE (PARTITION) - //SUGGEST SURFACE
                SurfaceFactory surface_fact = new SurfaceFactory(GeoWaveArr[BSP_Index].SurfaceArr, GeoWaveArr[BSP_Index].boubdingBox);
                Surface surface = new Surface();
                Surface best_surface = new Surface();

                GeoWave child0 = new GeoWave(GeoWaveArr[BSP_Index].boubdingBox);
                GeoWave child1 = new GeoWave(GeoWaveArr[BSP_Index].boubdingBox);
                GeoWave best_child0 = new GeoWave();
                GeoWave best_child1 = new GeoWave();

                double error_size = double.MaxValue; ;
                int coordinate_index = -1;
                int best_coordinate_index = -1;
                while (surface_fact.getNextSurface(surface, ref coordinate_index))
                {
                    //find error size
                    double error = evaluateErrorWithPartition(surface, child0, child1, BSP_Index);

                    //SET BETTER SURFACE - TO REMAIN WITH THE BEST 
                    if (error < error_size)
                    {
                        error_size = error;
                        surface.pnt.CopyTo(best_surface.pnt,0);
                        surface.normal.CopyTo(best_surface.normal, 0);
                        Form1.setMatrix(best_child0.approx_solution, child0.approx_solution);
                        Form1.setMatrix(best_child1.approx_solution, child1.approx_solution);
                        best_child0.proximity = child0.proximity;
                        best_child1.proximity = child1.proximity;
                        best_child0.pointsIdArray.Clear();
                        best_child1.pointsIdArray.Clear();
                        best_child0.pointsIdArray = child0.pointsIdArray.ToList();
                        best_child1.pointsIdArray = child1.pointsIdArray.ToList();
                        best_coordinate_index = coordinate_index;
                    }
                }

                if (error_size < DecisionTreeForm.ApproximationThreshold)
                    return;

                //ADD SURFACE TO SURFACE LIST 
                //int surfaceID = SurfaceArr.Count;
                //SurfaceArr.Add(best_surface);
                
                //AT THIS POINT INEED TO IMPROVE PERFORMANCE SIGNIFICANTLY BY SWITHCING SURFACES IN WAVELETS OR 
                //PASSING SURFACES BY REFERENCE (SAVING MEMORY)
                
                //because I might like to ermove surfaces in future code - copy surfaces by value 
                child1.SurfaceArr = Surface.CopyList(GeoWaveArr[BSP_Index].SurfaceArr);
                child1.SurfaceArr.Add(new Surface(best_surface));

                //switch normal direction (for child0)
                best_surface.normal = best_surface.normal.Multiply(-1);
                child0.SurfaceArr = Surface.CopyList(GeoWaveArr[BSP_Index].SurfaceArr);
                child0.SurfaceArr.Add(new Surface(best_surface));

                //VERIFY THAT THE NEW SURFACE IS NOT TRIVIAL
                if (child0.boubdingBox[1, best_coordinate_index] == best_surface.pnt[best_coordinate_index] ||
                    child1.boubdingBox[0, best_coordinate_index] == best_surface.pnt[best_coordinate_index])
                    return;
                
                //ADJUST BOUNDING BOX
                child0.boubdingBox[1, best_coordinate_index] = best_surface.pnt[best_coordinate_index];
                child1.boubdingBox[0, best_coordinate_index] = best_surface.pnt[best_coordinate_index];

                child0.pointsIdArray.Clear();
                child1.pointsIdArray.Clear();
                child0.pointsIdArray = best_child0.pointsIdArray.ToList();
                child1.pointsIdArray = best_child1.pointsIdArray.ToList();
                Form1.setMatrix(child0.approx_solution, best_child0.approx_solution);
                Form1.setMatrix(child1.approx_solution, best_child1.approx_solution);
                child0.proximity = best_child0.proximity;
                child1.proximity = best_child1.proximity;
                
                //SET TWO CHILDS
                child0.parentID = child1.parentID = BSP_Index;
                child0.child0 = child1.child0 = -1;
                child0.child1 = child1.child1 = -1;
                child0.level = child1.level = GeoWaveArr[BSP_Index].level +1;
                child0.computeNorm(GeoWaveArr[BSP_Index]);
                child1.computeNorm(GeoWaveArr[BSP_Index]);
                //set ids of surfs... - tbd !!!!
                GeoWaveArr.Add(child0);
                GeoWaveArr.Add(child1);
                GeoWaveArr[BSP_Index].child0 = GeoWaveArr.Count - 2; 
                GeoWaveArr[BSP_Index].child1 = GeoWaveArr.Count - 1; 

                //RECURSION STEP !!!
                recursiveBSP(GeoWaveArr[BSP_Index].child0);
                recursiveBSP(GeoWaveArr[BSP_Index].child1);
            }
        }
        
        public void start(GeoWave gwRoot)
        {
            calc_least_squares_proximity(inputData, Labels, gwRoot);
            gwRoot.computeNorm();
            gwRoot.level = 0;
            GeoWaveArr.Add(gwRoot);
            recursiveBSP(0);//0 is the root index

            //GeoWave gw_child1 = new GeoWave();
            //GeoWave gw_child2 = new GeoWave();
            //find two childs ()
            
            
            //lst.Minimize(,)
            //Minimize()
        }
    }
}

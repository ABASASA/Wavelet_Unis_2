using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
//using Amazon.S3.Model;

namespace DataScienceAnalysis
{
    class methodConfig
    {
        public methodConfig(bool setDefualt)
        {
            if (setDefualt)
            {
                //set default values
                dim.Add(-1);
                approxThresh.Add(0.001);
                partitionErrType.Add(2);
                approxOrder.Add(0);
                boostNum.Add(5);
                boostlamda_0.Add(1);
                boostProoning_0.Add(5);
                boostNormTarget.Add(2);
                boostNormsecond.Add(0);
                boostTau.Add(1);
                rfNum.Add(5);
                rfBaggingPercent.Add(0.65);
                NskipsinKfunc.Add(0.1);
                minWaveSize.Add(9);
                hopping_size.Add(1);
                test_error_size.Add(1);

            }
        }

        public List<int> dim = new List<int>();//-1 is no setting
        public List<double> approxThresh = new List<double>();//defines where to stop the partitions (when error is smaller then this)
        public List<int> partitionErrType = new List<int>();//defines how to measure the partition error (L1, L2, L0, lINF, lSINGULARITY)
        public List<int> approxOrder = new List<int>();//Polinoma=ial order - 0 const, ....
        public List<int> boostNum = new List<int>();//number of boosts - -1 no boosting
        // ||F-f||boostNormTarget + boostlamda_0 * (||F-f||boostNormsecond)pow(tau)
        public List<double> boostlamda_0 = new List<double>();//boosting lambda 0
        public List<double> boostProoning_0 = new List<double>();//boosting with prooning  
        public List<int> boostNormTarget = new List<int>();//boostNormTarget
        public List<int> boostNormsecond = new List<int>();//boostNormsecond
        public List<int> boostTau = new List<int>();//boostNormsecond
        public List<int> rfNum = new List<int>();///number of RF trees
        public List<double> rfBaggingPercent = new List<double>();//boosting lambda 0
        public List<double> NskipsinKfunc = new List<double>();///number of skipps in k functional (trick to gain performance)
        public List<int> minWaveSize = new List<int>();
        public List<int> hopping_size = new List<int>();
        public List<int> test_error_size = new List<int>();
       

        public List<recordConfig> recArr = new List<recordConfig>();
        public recordConfig.LearnigType learningType { set; get; }

        public void generateRecordConfigArr()
        {
            for (int i0 = 0; i0 < dim.Count; i0++)
            for (int i1 = 0; i1 < approxThresh.Count; i1++)
            for (int i2 = 0; i2 < partitionErrType.Count; i2++)
            for (int i3 = 0; i3 < approxOrder.Count; i3++)
            for (int i4 = 0; i4 < boostNum.Count; i4++)
            for (int i5 = 0; i5 < boostlamda_0.Count; i5++)
            for (int i6 = 0; i6 < boostNormTarget.Count; i6++)
            for (int i7 = 0; i7 < boostNormsecond.Count; i7++)
            for (int i8 = 0; i8 < boostTau.Count; i8++)
            for (int i9 = 0; i9 < rfNum.Count; i9++)
            for (int i10 = 0; i10 < rfBaggingPercent.Count; i10++)
            for (int i11 = 0; i11 < NskipsinKfunc.Count; i11++)
            for (int i12 = 0; i12 < minWaveSize.Count; i12++)
            for (int i13 = 0; i13 < hopping_size.Count; i13++)
            for (int i14 = 0; i14 < test_error_size.Count; i14++)    
            for (int i15 = 0; i15 < boostProoning_0.Count; i15++)  
            {
                recordConfig rc = new recordConfig(dim[i0],
                                                    approxThresh[i1],
                                                    partitionErrType[i2],
                                                    approxOrder[i3],
                                                    boostNum[i4],
                                                    boostlamda_0[i5],
                                                    boostNormTarget[i6],
                                                    boostNormsecond[i7],
                                                    boostTau[i8],
                                                    rfNum[i9],
                                                    rfBaggingPercent[i10],
                                                    NskipsinKfunc[i11],
                                                    minWaveSize[i12],
                                                    hopping_size[i13],
                                                    test_error_size[i14],
                                                    boostProoning_0[i15], 0);
                recArr.Add(rc);                                                    
            }
        }
    }

    public class recordConfig
    {
        //CONSTRUCTOR
        public recordConfig(
                             int _dim,
                             double _approxThresh,
                             int _partitionErrType,
                             int _approxOrder,
                             int _boostNum,
                             double _boostlamda_0,
                             int _boostNormTarget,
                             int _boostNormsecond,
                             int _boostTau,
                             int _rfNum,
                             double _rfBaggingPercent,
                             double _NskipsinKfunc,
                             int _minWaveSize,
                             int _hopping_size,
                             double _test_error_size,
                             double _boostProoning_0,
                             int _labelDim)
        {
            dim = _dim;
            approxThresh = _approxThresh;
            partitionErrType = _partitionErrType;
            approxOrder = _approxOrder;
            boostNum = _boostNum;
            boostlamda_0 = _boostlamda_0;
            boostProoning_0 = _boostProoning_0;
            boostNormTarget = _boostNormTarget;
            boostNormsecond = _boostNormsecond;
            boostTau = _boostTau;
            rfNum = _rfNum;
            rfBaggingPercent = _rfBaggingPercent;
            NskipsinKfunc = _NskipsinKfunc;
            minWaveSize = _minWaveSize;
            hopping_size = _hopping_size;
            test_error_size = _test_error_size;
            labelDim = _labelDim;
            //for categorical variables only
            indOfCategorical = new List<int>();
            hasCategorical = false;
        }

        public enum LearnigType { Regression, BinaryClassification, L1, ManyClassification }
        public LearnigType learningType { set; get; }


        public List<int> indOfCategorical { set; get; }

       

        public bool hasCategorical
        {
            set {  }
            get { return (indOfCategorical.Count != 0); }
        }
        //for binary classification
        public double binaryMinClass { set; get; }
        public double binaryMaxClass { set; get; }
        public double midClassValue { set; get; }

        public int labelDim;
        public int dim;
        public double approxThresh;
        public int partitionErrType;
        public int approxOrder;
        public int boostNum;
        public double boostlamda_0;
        public double boostProoning_0;
        public int boostNormTarget;
        public int boostNormsecond;
        public int boostTau;
        public int rfNum;
        public double rfBaggingPercent;
        public double NskipsinKfunc;
        public int minWaveSize;
        public int hopping_size;
        public double test_error_size;
        public int NwaveletsBoosting;
        public double learningRate;
        public int boostNumLearningRate;
        public double percent_training_db;
        public int[] waveletsTestRange = new int[2];
        public int[] pruningTestRange = new int[2];
        public int[] RFwaveletsTestRange = new int[2];
        public int[] RFpruningTestRange = new int[2];
        public int BoundLevel;
        public int NDimsinRF;
        public int split_type;//0 regular l2 partition, 1 rand split
        public int NormLPType;
        public int CrossValidFold;
        public int boundDepthTree;

        public int getEstimationArrSize(int geoWaveArrSize)
        {
            return waveletsTestRange[1] == 0 ?
                Convert.ToInt32(test_error_size * geoWaveArrSize / hopping_size)
                : Convert.ToInt32((1 + waveletsTestRange[1] - waveletsTestRange[0]) / hopping_size);
        }
        public string getFullName()
        {
            string name =

            "dim_" + dim.ToString()
            + "_appThsh_" + approxThresh.ToString()
            + "_partErrType_" + partitionErrType.ToString()
            + "_appOrdr_" + approxOrder.ToString()
            + "_Bnum_" + boostNum.ToString()
            + "_Blamda_" + boostlamda_0.ToString()
            + "_Bproonn_" + boostProoning_0.ToString()
            + "_Bnrm0_" + boostNormTarget.ToString()
            + "_Bnrm1_" + boostNormsecond.ToString()
            + "_BnrmTau_" + boostTau.ToString()
            + "_rfNum_" + rfNum.ToString()
            + "_rfBagPercnt_" + rfBaggingPercent.ToString()
            + "_NskipsinKfunc_" + NskipsinKfunc.ToString()
            + "_minWaveSize_" + minWaveSize.ToString()
            + "_hopping_size_" + hopping_size.ToString()
            + "_test_error_size_" + test_error_size.ToString()
            + "_NwaveletsBoosting_" + NwaveletsBoosting.ToString()
            + "_learningRate_" + learningRate.ToString()
            + "_boostNumLearningRate_" + boostNumLearningRate.ToString()
            + "_percent_training_db_" + percent_training_db.ToString()
            + "_waveletsTestRange_" + waveletsTestRange[0].ToString() + "To" + waveletsTestRange[1].ToString()
            + "_pruningTestRange_" + pruningTestRange[0].ToString() + "To" + pruningTestRange[1].ToString()
            + "_waveletsTestRange_" + RFwaveletsTestRange[0].ToString() + "To" + RFwaveletsTestRange[1].ToString()
            + "_pruningTestRange_" + RFpruningTestRange[0].ToString() + "To" + RFpruningTestRange[1].ToString()
            + "_BoundLevel_" + BoundLevel.ToString()
            + "_NDimsinRF_" + NDimsinRF.ToString()
            + "_NDimsinRF_" + split_type.ToString()
            + "_NormLPType_" + NormLPType.ToString()
            + "_boundDepthTree_" + boundDepthTree.ToString()
            + "_CrossValidFold_" + CrossValidFold.ToString();

            
            return name;
        }

        public string getShortName()
        {
            string name =

            dim.ToString()
            + "_" + approxThresh.ToString()
            + "_" + partitionErrType.ToString()
            + "_" + approxOrder.ToString()
            + "_" + boostNum.ToString()
            + "_" + boostlamda_0.ToString()
            + "_" + boostProoning_0.ToString()
            + "_" + boostNormTarget.ToString()
            + "_" + boostNormsecond.ToString()
            + "_" + boostTau.ToString()
            + "_" + rfNum.ToString()
            + "_" + rfBaggingPercent.ToString()
            + "_" + NskipsinKfunc.ToString()
            + "_" + minWaveSize.ToString()
            + "_" + hopping_size.ToString()
            + "_" + test_error_size.ToString()
            + "_" + NwaveletsBoosting.ToString()
            + "_" + learningRate.ToString()
            + "_" + boostNumLearningRate.ToString()
            + "_" + percent_training_db.ToString()
            + "_" + waveletsTestRange[0].ToString() + "To" + waveletsTestRange[1].ToString()
            + "_" + pruningTestRange[0].ToString() + "To" + pruningTestRange[1].ToString()
            + "_" + RFwaveletsTestRange[0].ToString() + "To" + RFwaveletsTestRange[1].ToString()
            + "_" + RFpruningTestRange[0].ToString() + "To" + RFpruningTestRange[1].ToString()
            + "_" + BoundLevel.ToString()
            + "_" + NDimsinRF.ToString()
            + "_" + split_type.ToString()
            + "_" + boundDepthTree.ToString()
            + "_" + CrossValidFold.ToString();

            return name;
        }
    }
}

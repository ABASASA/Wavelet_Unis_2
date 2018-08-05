using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Accord.Math;
using Accord.Statistics.Analysis;

//using Amazon.S3;
//using Amazon.S3.IO;

namespace DataScienceAnalysis
{
    static class Helpers
    {
        public static void applyFor(int begin, int size, Action<int> body)
        {
             if (Form1.rumPrallel) Parallel.For(begin, size, body); 
             else regularDelegateFor(begin, size, body);
        }
        private static void regularDelegateFor(int begin, int size, Action<int> body)
        {
            for (int i = begin; i < size; i++)
            {
                body.Invoke(i);
            }
        }

        public static int indexOfMin(double[] self)
        {
            if (self == null)
            {
                throw new ArgumentNullException("self");
            }

            if (self.Length == 0)
            {
                return -1;
                //throw new ArgumentException("List is empty.", "self");
            }

            double min = self[0];
            int minIndex = 0;

            for (int i = 1; i < self.Length; ++i)
            {
                if (self[i] > min) continue;
                min = self[i];
                minIndex = i;
            }

            return minIndex;
        }
        public static int indexOfMin( double[][] self)
        {
            if (self == null)
            {
                throw new ArgumentNullException("self");
            }

            if (self.Length == 0)
            {
                throw new ArgumentException("List is empty.", "self");
            }

            double min = self[0][0];
            int minIndex = 0;

            for (int i = 1; i < self.Length; ++i)
            {
                if (self[i][0] > min) continue;
                min = self[i][0];
                minIndex = i;
            }

            return minIndex;
        }

    /*    public static AmazonS3Client configAmazonS3ClientS3Client(int timeOutHours=3)
        {
            AmazonS3Config confisS3 = new AmazonS3Config { ProxyHost = null };
            TimeSpan timeOUT = new TimeSpan(timeOutHours, 0, 0);
            confisS3.ReadWriteTimeout = timeOUT;
            confisS3.Timeout = timeOUT;
            return new AmazonS3Client(confisS3);
        }*/

        public static void createMainDirectoryOrResultPath(string path,string bucketName)
        {

            //if (!Form1.UseS3)
           // {
                if (!Directory.Exists(path))
                    Directory.CreateDirectory(path);
           // }
/*            else
            {
                S3DirectoryInfo s3results_path = new S3DirectoryInfo(client, bucketName, path);
                if (!s3results_path.Exists)
                    s3results_path.Create();
            }*/
        }

        public static void createOutputDirectories (List<recordConfig> configArr,
                                                userConfig uConfig, string bucketName, string resultsPath)
        {
            string MainFolderName = Form1.MainFolderName;
            foreach (recordConfig t in configArr)
            {
                if (!Form1.UseS3 && !Directory.Exists(MainFolderName + "\\" + t.getShortName()))
                {
                    Directory.CreateDirectory(MainFolderName + "\\" + t.getShortName());
                    StreamWriter sw = new StreamWriter(MainFolderName + "\\" + t.getShortName() + "\\record_properties.txt", false);
                    sw.WriteLine(t.getFullName());
                    sw.Close();
                    uConfig.printConfig(MainFolderName + "\\config.txt");
                }
               // if (!Form1.UseS3) continue;
              /*  S3DirectoryInfo s3results_path_with_folders =
                    new S3DirectoryInfo(client, bucketName, resultsPath + "\\" + t.getShortName());
                if (!s3results_path_with_folders.Exists)
                {
                    s3results_path_with_folders.Create();
                    S3FileInfo outFile = s3results_path_with_folders.GetFile("record_properties.txt");
                    StreamWriter sw = new StreamWriter(outFile.OpenWrite());
                    sw.WriteLine(t.getFullName());
                    sw.Close();

                    S3FileInfo configFile = s3results_path_with_folders.GetFile("config.txt");
                    uConfig.printConfig("", configFile);
                }*/
            }
        }

        public static double[][] copyAndRemoveCategoricalColumns(double[][] originalData, recordConfig rc)
        {
            if (!rc.hasCategorical) return originalData; //don't have categorical variables

            double[][] withRemovedColumns = new double[originalData.Length][];
            for (int i = 0; i < originalData.Length; i++)
            {
                double[] row = originalData[i];
                row = rc.indOfCategorical.Aggregate(row, (current, indOfColumnToRemove) => current.Where((val, ind) => ind != indOfColumnToRemove).ToArray());
                withRemovedColumns[i]=(double[])row.Clone();
            }
            return withRemovedColumns;
        }


    }


}

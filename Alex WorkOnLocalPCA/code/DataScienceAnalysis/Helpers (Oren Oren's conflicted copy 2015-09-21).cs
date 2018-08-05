using System;
using System.Threading.Tasks;

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

      
    }


}

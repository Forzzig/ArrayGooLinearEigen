#ifndef __OUTPUT_HELPER_H__
#define __OUTPUT_HELPER_H__

#include<fstream>
#include<string>
#include<direct.h>
#include<iostream>
#include<TimeControl.h>
#include<ctime>
#include<Eigen/Sparse>
#include<iomanip>

using namespace std;
using namespace Eigen;

int fstream_prepare(ofstream& txt, ofstream& csv, SparseMatrix<double>& A, string matrix_name, string method, string suff);

#endif

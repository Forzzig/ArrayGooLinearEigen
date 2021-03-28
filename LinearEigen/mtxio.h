#ifndef __MTXIO_H__

#define __MTXIO_H__
#include "Eigen/Sparse"
#include "iostream"
#include <vector>
#include "fstream"
#include "algorithm"
#include <string>

using namespace std;
using namespace Eigen;
class mtxio {
public:
	static SparseMatrix<double, RowMajor, __int64>& getSparseMatrix(string filename, string filepath);
	static SparseMatrix<double, RowMajor, __int64>& getSparseMatrix(string filename);
};

#endif
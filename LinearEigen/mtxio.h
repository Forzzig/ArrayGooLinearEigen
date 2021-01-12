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
	static SparseMatrix<double>& getSparseMatrix(string filename, string filepath);
	static SparseMatrix<double>& getSparseMatrix(string filename);
};

#endif
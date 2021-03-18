#ifndef __LOBPCG_II_H__
#define __LOBPCG_II_H__
#include<LinearEigenSolver.h>

using namespace Eigen;

class LOBPCG_II : public LinearEigenSolver {
public:
	int cgstep;
	double* storage = NULL;
	Map<MatrixXd, Unaligned, OuterStride<>> X, P, W;
	MatrixXd LAM;
	LOBPCG_II(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep);
	void compute();
};
#endif

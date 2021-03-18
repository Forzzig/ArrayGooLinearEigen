#ifndef __LOBPCG_II_BATCH_H__
#define __LOBPCG_II_BATCH_H__
#include<LinearEigenSolver.h>

using namespace Eigen;

class LOBPCG_II_Batch : public LinearEigenSolver {
public:
	int cgstep, batch;
	double* storage = NULL;
	Map<MatrixXd, Unaligned, OuterStride<>> X, P, W;
	MatrixXd LAM;
	LOBPCG_II_Batch(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int batch);
	void compute();
};
#endif

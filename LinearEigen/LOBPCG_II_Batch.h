#ifndef __LOBPCG_II_BATCH_H__
#define __LOBPCG_II_BATCH_H__
#include<LinearEigenSolver.h>

using namespace Eigen;

class LOBPCG_II_Batch : public LinearEigenSolver {
public:
	int cgstep, batch;
	double* storage = NULL;
	Map<MatrixXd, Unaligned, OuterStride<>> X, P, W;
	MatrixXd Lam;

	ConjugateGradient<SparseMatrix<double, RowMajor, __int64>, Lower | Upper> linearsolver;

	LOBPCG_II_Batch(SparseMatrix<double, RowMajor, __int64>& A, SparseMatrix<double, RowMajor, __int64>& B, int nev, int cgstep, int batch);
	~LOBPCG_II_Batch();
	void compute();
};
#endif

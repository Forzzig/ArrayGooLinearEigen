#ifndef __LOBPCG_I_BATCH_H__
#define __LOBPCG_I_BATCH_H__
#include<LinearEigenSolver.h>

using namespace Eigen;

class LOBPCG_I_Batch : public LinearEigenSolver {
public:
	int cgstep, batch;
	double* storage = NULL;
	Map<MatrixXd> X, P, W;
	MatrixXd H;

	ConjugateGradient<SparseMatrix<double, RowMajor>, Lower | Upper> linearsolver;

	LOBPCG_I_Batch(SparseMatrix<double, RowMajor>& A, SparseMatrix<double, RowMajor>& B, int nev, int cgstep, int batch);
	~LOBPCG_I_Batch();
	void compute();
};
#endif

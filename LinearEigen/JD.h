#ifndef __JD_H__
#define __JD_H__
#include<Eigen\Sparse>
#include<Eigen\Dense>
#include<EigenResult.h>
#include<iostream>
#include<LinearEigenSolver.h>

using namespace Eigen;
class JD : public LinearEigenSolver {
public:
	int restart;
	MatrixXd V, W, H;
	int cgstep;
	int batch;
	JD(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int restart, int batch);
	void specialCG(SparseMatrix<double>& A, SparseMatrix<double>& B, MatrixXd& b, Map<MatrixXd>& U, Map<MatrixXd>& X, MatrixXd& lam);
	void compute();
};
#endif

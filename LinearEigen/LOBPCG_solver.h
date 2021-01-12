#ifndef __LOBPCG_SOLVER_H__
#define __LOBPCG_SOLVER_H__
#include<Eigen\Sparse>
#include<Eigen\Dense>
#include<EigenResult.h>
#include<iostream>
#include<LinearEigenSolver.h>

using namespace Eigen;
class LOBPCG_solver : public LinearEigenSolver {
public:
	MatrixXd X, P, W, V, LAM;
	LOBPCG_solver(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep);
	void compute();
};
#endif
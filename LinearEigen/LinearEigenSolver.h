#ifndef __LINEAR__EIGEN__SOLVER__
#define __LINEAR__EIGEN__SOLVER__

#include<Eigen/Dense>
#include<Eigen/Sparse>
#include<vector>
#include<cstdlib>

using namespace std;
using namespace Eigen;
class LinearEigenSolver {
public:
	static double ORTH_TOL;
	static double EIGTOL;
	static int CHECKNUM;
	int nIter;
	SparseMatrix<double>& A;
	SparseMatrix<double>& B;
	int nev;
	vector<double> eigenvalues;
	MatrixXd eigenvectors;
	//GeneralizedSelfAdjointEigenSolver<MatrixXd> eigensolver;
	SelfAdjointEigenSolver<MatrixXd> eigensolver;
	ConjugateGradient<SparseMatrix<double>, Lower | Upper> linearsolver;
	void projection_RR(MatrixXd& V, SparseMatrix<double>& A, MatrixXd& eigenvalues, MatrixXd& eigenvectors);
	void projection_RR(Map<MatrixXd>& V, SparseMatrix<double>& A, MatrixXd& eigenvalues, MatrixXd& eigenvectors);
	void RR(MatrixXd& A, MatrixXd& eigenvalues, MatrixXd& eigenvectors);
	void RR(Block<MatrixXd>& A, MatrixXd& eigenvalues, MatrixXd& eigenvectors);
	int normalize(Block<MatrixXd>& v, SparseMatrix<double>& B);
	int normalize(Block<Map<MatrixXd>>& v, SparseMatrix<double>& B);
	void orthogonalization(MatrixXd& V, SparseMatrix<double>& B);
	void orthogonalization(MatrixXd& V1, MatrixXd& V2, SparseMatrix<double>& B);
	int orthogonalization(Map<MatrixXd>& V, SparseMatrix<double>& B);
	int orthogonalization(Map<MatrixXd>& V1, Map<MatrixXd>& V2, SparseMatrix<double>& B);
	int LinearEigenSolver::conv_select(MatrixXd& eval, MatrixXd& evec, double shift, MatrixXd& valout, MatrixXd& vecout);
	int LinearEigenSolver::conv_check(Map<MatrixXd>& eval, Map<MatrixXd>& evec, double shift);
	LinearEigenSolver(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev);
	virtual void compute() = 0;
};
#endif
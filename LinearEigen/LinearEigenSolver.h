#ifndef __LINEAR__EIGEN__SOLVER__
#define __LINEAR__EIGEN__SOLVER__

#include<Eigen/Dense>
#include<Eigen/Sparse>
#include<vector>
#include<iostream>
#include<fstream>
#include<TimeControl.h>

using namespace std;
using namespace Eigen;
class LinearEigenSolver {
public:
	static double ORTH_TOL;
	static double EIGTOL;
	static int CHECKNUM;
	static fstream coutput;
	int nIter;
	time_t start_time, end_time;
	long long com_of_mul;
	SparseMatrix<double>& A;
	SparseMatrix<double>& B;
	int nev;
	vector<double> eigenvalues;
	MatrixXd eigenvectors;
	//GeneralizedSelfAdjointEigenSolver<MatrixXd> eigensolver;
	SelfAdjointEigenSolver<MatrixXd> eigensolver;
	ConjugateGradient<SparseMatrix<double>, Lower | Upper> linearsolver;
	template <typename Derived>
	void projection_RR(Derived& V, SparseMatrix<double>& A, MatrixXd& eigenvalues, MatrixXd& eigenvectors);
	
	template<typename Derived>
	void RR(Derived& H, MatrixXd& eigenvalues, MatrixXd& eigenvectors);
	
	template<typename Derived>
	int normalize(Derived& v, SparseMatrix<double>& B);
	
	template<typename Derived>
	int orthogonalization(Derived& V, SparseMatrix<double>& B);
	
	template<typename Derived1, typename Derived2>
	void orthogonalization(Derived1& V1, Derived2& V2, SparseMatrix<double>& B);

	template<typename Derived_val, typename Derived_vec, typename Out_val, typename Out_vec>
	int conv_select(Derived_val& eval, Derived_vec& evec, double shift, Out_val& valout, Out_vec& vecout);

	LinearEigenSolver(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev);
	virtual void compute() = 0;

	void finish() {
		end_time = time(NULL);
	}
};

template<typename Derived>
void LinearEigenSolver::projection_RR(Derived& V, SparseMatrix<double>& A, MatrixXd& eigenvalues, MatrixXd& eigenvectors) {
	MatrixXd tmpA = V.transpose() * A * V;
	com_of_mul += A.nonZeros() * V.cols() + V.cols() * A.rows() * V.cols();
	eigensolver.compute(tmpA);
	eigenvalues = eigensolver.eigenvalues();
	eigenvectors = eigensolver.eigenvectors();
	com_of_mul += 24 * V.cols() * V.cols() * V.cols();
}

template<typename Derived>
void LinearEigenSolver::RR(Derived& H, MatrixXd& eigenvalues, MatrixXd& eigenvectors) {
	eigensolver.compute(H);
	eigenvalues = eigensolver.eigenvalues();
	eigenvectors = eigensolver.eigenvectors();
	com_of_mul += 24 * H.cols() * H.cols() * H.cols();
}

template<typename Derived>
int LinearEigenSolver::normalize(Derived& v, SparseMatrix<double>& B) {
	double r = sqrt((v.transpose() * B * v)(0, 0));
	com_of_mul += B.nonZeros() + B.rows();
	if (r < LinearEigenSolver::ORTH_TOL) {
		return 1;
	}
	v /= r;
	return 0;
}

template<typename Derived>
int LinearEigenSolver::orthogonalization(Derived& V, SparseMatrix<double>& B) {
	vector<int> pos;
	MatrixXd tmpv;
	for (int i = 0; i < V.cols(); ++i) {
		int flag = normalize(V.col(i), B);
		if (!flag) {
			tmpv = (B * V.col(i)).transpose();
			com_of_mul += B.nonZeros();
			for (int j = i + 1; j < V.cols(); ++j) {
				double tmp = (tmpv * V.col(j))(0, 0);
				V.col(j) -= tmp * V.col(i);
			}
			com_of_mul += 2 * V.rows() * (V.cols() - 1 - i);
			pos.push_back(i);
		}
	}
	for (int i = 0; i < pos.size(); ++i)
		if (i != pos[i])
			V.col(i) = V.col(pos[i]);
	return V.cols() - pos.size();
}

template<typename Derived1, typename Derived2>
void LinearEigenSolver::orthogonalization(Derived1& V1, Derived2& V2, SparseMatrix<double>& B) {
	MatrixXd tmpv;
	for (int i = 0; i < V2.cols(); ++i) {
		tmpv = (B * V2.col(i)).transpose();
		for (int j = 0; j < V1.cols(); ++j) {
			double tmp = (tmpv * V1.col(j))(0, 0);
			V1.col(j) -= tmp * V2.col(i);
		}
		com_of_mul += B.nonZeros() + 2 * V2.rows() * V1.cols();
	}
}

template<typename Derived_val, typename Derived_vec, typename Out_val, typename Out_vec>
int LinearEigenSolver::conv_select(Derived_val& eval, Derived_vec& evec, double shift, Out_val& valout, Out_vec& vecout) {
	MatrixXd tmp, tmpA, tmpB, eig(evec.rows(), evec.cols());
	int flag = LinearEigenSolver::CHECKNUM;
	int prev = eigenvectors.cols();
	int goon = 0;
	int cnv = 0;
	for (int i = 0; i < evec.cols(); ++i) {
		double err = 1;
		if (flag > 0) {
			tmpA = A * evec.col(i);
			tmpB = B * evec.col(i);
			tmp = tmpB * (eval(i, 0) - shift) - tmpA;
			err = tmp.col(0).norm() / tmpA.norm();

			com_of_mul += A.nonZeros() + B.nonZeros() + 3 * A.rows();

			if (err < EIGTOL) {
				cout << "检查第" << i + 1 << "个特征值：" << eval(i, 0) - shift << "，相对误差：" << err << endl;
				cout << "达到收敛条件！" << endl;
				eigenvalues.push_back(eval(i, 0) - shift);
				eig.col(cnv) = evec.col(i);
				++cnv;
				if (prev + cnv >= nev)
					break;
				continue;
			}
		}
		cout << "检查第" << i + 1 << "个特征值：" << eval(i, 0) - shift << "，相对误差：" << err << endl;
		//vecout与evec可能相同
		if (&vecout(0, goon) != &evec(0, i))
			vecout.col(goon) = evec.col(i);
		valout(goon, 0) = eval(i, 0) - shift;
		++goon;
		--flag;
		if (goon >= vecout.cols())
			break;
	}
	eigenvectors.conservativeResize(evec.rows(), prev + cnv);
	eigenvectors.rightCols(cnv) = eig.leftCols(cnv);
	return eigenvalues.size();
}

#endif
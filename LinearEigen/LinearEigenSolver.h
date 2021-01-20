#ifndef __LINEAR__EIGEN__SOLVER__
#define __LINEAR__EIGEN__SOLVER__

#include<Eigen/Dense>
#include<Eigen/Sparse>
#include<vector>
#include<iostream>
#include<iomanip>
#include<fstream>

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
	template <typename Derived>
	void projection_RR(Derived& V, SparseMatrix<double>& A, MatrixXd& eigenvalues, MatrixXd& eigenvectors);
	
	template<typename Derived>
	void RR(Derived& A, MatrixXd& eigenvalues, MatrixXd& eigenvectors);
	
	template<typename Derived>
	int normalize(Derived& v, SparseMatrix<double>& B);
	
	template<typename Derived>
	int orthogonalization(Derived& V, SparseMatrix<double>& B);
	
	template<typename Derived1, typename Derived2>
	int orthogonalization(Derived1& V1, Derived2& V2, SparseMatrix<double>& B);

	template<typename Derived_val, typename Derived_vec, typename Out_val, typename Out_vec>
	int conv_select(Derived_val& eval, Derived_vec& evec, double shift, Out_val& valout, Out_vec& vecout);
	
	template<typename Derived_val, typename Derived_vec>
	int conv_check(Derived_val& eval, Derived_vec& evec, double shift);
	LinearEigenSolver(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev);
	virtual void compute() = 0;
};

template<typename Derived>
void LinearEigenSolver::projection_RR(Derived& V, SparseMatrix<double>& A, MatrixXd& eigenvalues, MatrixXd& eigenvectors) {
	MatrixXd tmpA = V.transpose() * A * V;
	eigensolver.compute(tmpA);
	eigenvalues = eigensolver.eigenvalues();
	eigenvectors = eigensolver.eigenvectors();
}

template<typename Derived>
void LinearEigenSolver::RR(Derived& A, MatrixXd& eigenvalues, MatrixXd& eigenvectors) {
	eigensolver.compute(A);
	eigenvalues = eigensolver.eigenvalues();
	eigenvectors = eigensolver.eigenvectors();
}

template<typename Derived>
int LinearEigenSolver::normalize(Derived& v, SparseMatrix<double>& B) {
	MatrixXd tmp = v.transpose() * B * v;
	double r = sqrt(tmp(0, 0));
	if (r < LinearEigenSolver::ORTH_TOL) {
		return 1;
	}
	v /= r;
	return 0;
}

template<typename Derived>
int LinearEigenSolver::orthogonalization(Derived& V, SparseMatrix<double>& B) {
	MatrixXd tmpvi;
	MatrixXd tmp;
	int dep = 0;
	for (int i = 0; i < V.cols() - dep; ++i) {
		int flag = normalize(V.block(0, i, V.rows(), 1), B);
		if (flag) {
			V.col(i) = V.col(V.cols() - 1 - dep);
			++dep;
			--i;
			continue;
		}
		tmpvi = V.col(i).transpose();
		for (int j = i + 1; j < V.cols() - dep; ++j) {
			tmp = tmpvi * B * V.col(j);
			V.col(j) -= tmp(0, 0) * V.col(i);
		}
	}
	return dep;
}

template<typename Derived1, typename Derived2>
int LinearEigenSolver::orthogonalization(Derived1& V1, Derived2& V2, SparseMatrix<double>& B) {
	MatrixXd tmpvi;
	MatrixXd tmp;
	int dep = 0;
	for (int i = 0; i < V2.cols(); ++i) {
		tmpvi = V2.col(i).transpose();
		for (int j = 0; j < V1.cols(); ++j) {
			tmp = tmpvi * B * V1.col(j);
			V1.col(j) -= tmp(0, 0) * V2.col(i);
		}
		//cout << V1.col(0) << endl;
	}
	for (int j = 0; j < V1.cols() - dep; ++j) {
		int flag = normalize(V1.block(0, j, V1.rows(), 1), B);
		if (flag) {
			V1.col(j) = V1.col(V1.cols() - 1 - dep);
			++dep;
			--j;
		}
	}
	return dep;
}

template<typename Derived_val, typename Derived_vec, typename Out_val, typename Out_vec>
int LinearEigenSolver::conv_select(Derived_val& eval, Derived_vec& evec, double shift, Out_val& valout, Out_vec& vecout) {
	MatrixXd tmp1, tmp2;
	int flag = LinearEigenSolver::CHECKNUM;
	vector<int> hitpos;
	vector<int> goonpos;
	vector<double> ans;
	for (int i = 0; i < evec.cols(); ++i) {
		tmp1 = A * evec.col(i);
		tmp2 = B * evec.col(i) * (eval(i, 0) - shift) - tmp1;
		double tmp = tmp2.col(0).norm() / tmp1.col(0).norm();
		ans.push_back(tmp);
	}
	int prev = eigenvectors.cols();
	for (int i = 0; i < evec.cols(); ++i) {
		if ((ans[i] >= EIGTOL) || (flag <= 0) || (prev + hitpos.size() >= nev)) {
			cout << "检查第" << i + 1 << "个特征向量相对误差：" << ans[i] << endl;
			cout << "检查第" << i + 1 << "个特征值：" << eval(i, 0) - shift << endl;
			goonpos.push_back(i);
			if (goonpos.size() == vecout.cols())
				break;
			--flag;
		}
		else {
			cout << "第" << i + 1 << "个特征向量相对误差：" << ans[i] << endl;
			cout << "第" << i + 1 << "个特征值：" << eval(i, 0) - shift << endl;
			cout << "收敛！" << endl;
			hitpos.push_back(i);
		}
	}
	eigenvectors.conservativeResize(evec.rows(), min(nev, (int)(prev + hitpos.size())));
	for (int i = 0; i < hitpos.size(); ++i) {
		if (eigenvalues.size() == nev)
			break;
		eigenvalues.push_back(eval(hitpos[i], 0) - shift);
		eigenvectors.col(prev + i) = evec.col(hitpos[i]);
	}
	for (int i = 0; i < goonpos.size(); ++i) {
		vecout.col(i) = evec.col(goonpos[i]);
		valout(i, 0) = eval(goonpos[i], 0) - shift;
	}
	return eigenvalues.size();
}

template<typename Derived_val, typename Derived_vec>
int LinearEigenSolver::conv_check(Derived_val& eval, Derived_vec& evec, double shift) {
	MatrixXd tmp1, tmp2;
	int flag = LinearEigenSolver::CHECKNUM;
	vector<int> hitpos;
	vector<int> goonpos;
	vector<double> ans;
	for (int i = 0; i < evec.cols(); ++i) {
		tmp1 = A * evec.col(i);
		tmp2 = B * evec.col(i) * (eval(i, 0) - shift) - tmp1;
		double tmp = tmp2.col(0).norm() / tmp1.col(0).norm();
		ans.push_back(tmp);
	}
	int prev = eigenvectors.cols();
	for (int i = 0; i < evec.cols(); ++i) {
		if ((ans[i] >= EIGTOL) || (prev + hitpos.size() >= nev)) {
			cout << "检查第" << i + 1 << "个特征向量相对误差：" << ans[i] << endl;
			cout << "检查第" << i + 1 << "个特征值：" << eval(i, 0) - shift << endl;
			goonpos.push_back(i);
			--flag;
			if (flag <= 0)
				break;
		}
		else {
			cout << "第" << i + 1 << "个特征向量相对误差：" << ans[i] << endl;
			cout << "第" << i + 1 << "个特征值：" << eval(i, 0) - shift << endl;
			cout << "收敛！" << endl;
			hitpos.push_back(i);
		}
	}
	return hitpos.size();
}

#endif
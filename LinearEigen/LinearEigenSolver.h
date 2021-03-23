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
	static ofstream coutput;
	int nIter;
	time_t start_time, end_time;
	long long com_of_mul;
	SparseMatrix<double, RowMajor>& A;
	SparseMatrix<double, RowMajor>& B;
	int nev;
	vector<double> eigenvalues;
	MatrixXd eigenvectors;
	
	//GeneralizedSelfAdjointEigenSolver<MatrixXd> eigensolver;
	SelfAdjointEigenSolver<MatrixXd> eigensolver;
	
	template <typename Derived, typename Derived_val, typename Derived_vec>
	void projection_RR(Derived& V, SparseMatrix<double, RowMajor>& A, Derived_val& eigenvalues, Derived_vec& eigenvectors) {
		MatrixXd tmpA;
		tmpA.noalias() = A * V;
		tmpA.applyOnTheLeft(V.transpose());
		com_of_mul += A.nonZeros() * V.cols() + V.cols() * A.rows() * V.cols();
		eigensolver.compute(tmpA);
		eigenvalues = eigensolver.eigenvalues();
		eigenvectors = eigensolver.eigenvectors();
		com_of_mul += 24 * V.cols() * V.cols() * V.cols();
	}
	
	template<typename Derived, typename Derived_val, typename Derived_vec>
	void RR(Derived& H, Derived_val& eigenvalues, Derived_vec& eigenvectors) {
		eigensolver.compute(H);
		eigenvalues = eigensolver.eigenvalues();
		eigenvectors = eigensolver.eigenvectors();
		com_of_mul += 24 * H.cols() * H.cols() * H.cols();
	}
	
	template<typename Derived>
	int normalize(Derived& v, SparseMatrix<double, RowMajor>& B) {
		VectorXd tmp;
		tmp.noalias() = B * v;
		double r = sqrt(tmp.dot(v));
		com_of_mul += B.nonZeros() + B.rows();
		if (r < LinearEigenSolver::ORTH_TOL) {
			return 1;
		}
		v /= r;
		return 0;
	}
	
	template<typename Derived>
	int orthogonalization(Derived& V, SparseMatrix<double, RowMajor>& B) {
		vector<int> pos;
		VectorXd tmpv;
		for (int i = 0; i < V.cols(); ++i) {
			int flag = normalize(V.col(i), B);
			if (!flag) {
				tmpv.noalias() = B * V.col(i);
				com_of_mul += B.nonZeros();
					
#pragma omp parallel for
				for (int j = i + 1; j < V.cols(); ++j) {
					double tmp = V.col(j).dot(tmpv);
					V.col(j) -= tmp * V.col(i);
				}
				com_of_mul += 2 * V.rows() * (V.cols() - 1 - i);
				pos.push_back(i);
			}
		}
		for (int i = 0; i < pos.size(); ++i)
			if (i != pos[i])
				memcpy(&V(0, i), &V(0, pos[i]), V.rows() * sizeof(double));
				//V.col(i) = V.col(pos[i]);
		return V.cols() - pos.size();
	}
	
	template<typename Derived1, typename Derived2>
	void orthogonalization(Derived1& V1, Derived2& V2, SparseMatrix<double, RowMajor>& B) {
		VectorXd tmpv;
		for (int i = 0; i < V2.cols(); ++i) {
			tmpv.noalias() = B * V2.col(i);
#pragma omp parallel for
			for (int j = 0; j < V1.cols(); ++j) {
				double tmp = V1.col(j).dot(tmpv);
				V1.col(j) -= tmp * V2.col(i);
			}
			com_of_mul += B.nonZeros() + 2 * V2.rows() * V1.cols();
		}
	}

	template<typename Derived_val, typename Derived_vec, typename Out_val, typename Out_vec>
	int conv_select(Derived_val& eval, Derived_vec& evec, double shift, Out_val& valout, Out_vec& vecout) {
		VectorXd tmp, tmpA, tmpB;
		int flag = LinearEigenSolver::CHECKNUM;
		int prev = eigenvectors.cols();
		int goon = 0;
		for (int i = 0; i < evec.cols(); ++i) {
			double err = 1;
			if (flag > 0) {
				tmpA.noalias() = A * evec.col(i);
				tmpB.noalias() = B * evec.col(i);
				tmp.noalias() = tmpB * (eval(i, 0) - shift) - tmpA;
				err = tmp.col(0).norm() / tmpA.norm();

				com_of_mul += A.nonZeros() + B.nonZeros() + 3 * A.rows();

				if (err < LinearEigenSolver::EIGTOL) {
					cout << "检查第" << i + 1 << "个特征值：" << eval(i, 0) - shift << "，相对误差：" << err << endl;
					cout << "达到收敛条件！" << endl;
					eigenvalues.push_back(eval(i, 0) - shift);
					memcpy(&eigenvectors(0, eigenvalues.size() - 1), &evec(0, i), A.rows() * sizeof(double));
					//eigenvectors.col(eigenvalues.size() - 1) = evec.col(i);
					
					if (eigenvalues.size() >= nev)
						break;
					continue;
				}
			}
			cout << "检查第" << i + 1 << "个特征值：" << eval(i, 0) - shift << "，相对误差：" << err << endl;
			//vecout与evec可能相同
			if (&vecout(0, goon) != &evec(0, i))
				memcpy(&vecout(0, goon), &evec(0, i), A.rows() * sizeof(double));
				//vecout.col(goon) = evec.col(i);
			valout(goon, 0) = eval(i, 0) - shift;
			++goon;
			--flag;
			if (goon >= vecout.cols())
				break;
		}
		return eigenvalues.size();
	}

	LinearEigenSolver(SparseMatrix<double, RowMajor>& A, SparseMatrix<double, RowMajor>& B, int nev);
	virtual void compute() = 0;

	void finish() {
		end_time = time(NULL);
	}
};
#endif
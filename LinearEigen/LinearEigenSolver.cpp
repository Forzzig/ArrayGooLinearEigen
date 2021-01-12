#include<LinearEigenSolver.h>
#include<iostream>

double LinearEigenSolver::ORTH_TOL = 1e-10;
double LinearEigenSolver::EIGTOL = 1e-3;
int LinearEigenSolver::CHECKNUM = 3;

void LinearEigenSolver::projection_RR(MatrixXd& V, SparseMatrix<double>& A, MatrixXd& eigenvalues, MatrixXd& eigenvectors) {
	MatrixXd tmpA = V.transpose() * A * V;
	eigensolver.compute(tmpA);
	eigenvalues = eigensolver.eigenvalues();
	eigenvectors = eigensolver.eigenvectors();
}

void LinearEigenSolver::projection_RR(Map<MatrixXd>& V, SparseMatrix<double>& A, MatrixXd& eigenvalues, MatrixXd& eigenvectors) {
	MatrixXd tmpA = V.transpose() * A * V;
	eigensolver.compute(tmpA);
	eigenvalues = eigensolver.eigenvalues();
	eigenvectors = eigensolver.eigenvectors();
}

void LinearEigenSolver::RR(MatrixXd& A, MatrixXd& eigenvalues, MatrixXd& eigenvectors) {
	eigensolver.compute(A);
	eigenvalues = eigensolver.eigenvalues();
	eigenvectors = eigensolver.eigenvectors();
}

void LinearEigenSolver::RR(Block<MatrixXd>& A, MatrixXd& eigenvalues, MatrixXd& eigenvectors) {
	eigensolver.compute(A);
	//cout << A << endl;
	//system("pause");
	eigenvalues = eigensolver.eigenvalues();
	eigenvectors = eigensolver.eigenvectors();
}

int LinearEigenSolver::normalize(Block<MatrixXd>& v, SparseMatrix<double>& B) {
	MatrixXd tmp = v.transpose() * B * v;
	double r = sqrt(tmp(0, 0));
	if (r < LinearEigenSolver::ORTH_TOL) {
		return 1;
	}
	v /= r;
	return 0;
}

int LinearEigenSolver::normalize(Block<Map<MatrixXd>>& v, SparseMatrix<double>& B) {
	MatrixXd tmp = v.transpose() * B * v;
	double r = sqrt(tmp(0, 0));
	if (r < LinearEigenSolver::ORTH_TOL) {
		return 1;
	}
	v /= r;
	return 0;
}

void LinearEigenSolver::orthogonalization(MatrixXd& V, SparseMatrix<double>& B) {
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
	V.conservativeResize(V.rows(), V.cols() - dep);
}
void LinearEigenSolver::orthogonalization(MatrixXd& V1, MatrixXd& V2, SparseMatrix<double>& B) {
	MatrixXd tmpvi;
	MatrixXd tmp;
	int dep = 0;
	for (int i = 0; i < V2.cols(); ++i) {
		tmpvi = V2.col(i).transpose();
		for (int j = 0; j < V1.cols(); ++j) {
			tmp = tmpvi * B * V1.col(j);
			V1.col(j) -= tmp(0, 0) * V2.col(i);
		}
	}
	for (int j = 0; j < V1.cols() - dep; ++j) {
		int flag = normalize(V1.block(0, j, V1.rows(), 1), B);
		if (flag) {
			V1.col(j) = V1.col(V1.cols() - 1 - dep);
			++dep;
			--j;
		}
	}
	V1.conservativeResize(V1.rows(), V1.cols() - dep);
}

int LinearEigenSolver::orthogonalization(Map<MatrixXd>& V, SparseMatrix<double>& B) {
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
int LinearEigenSolver::orthogonalization(Map<MatrixXd>& V1, Map<MatrixXd>& V2, SparseMatrix<double>& B) {
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
		//cout << V1.col(0) << endl;
		//if (V2.cols() > 0)
		//	cout << V1.col(0).transpose() * B * V2.col(0) << endl;
		int flag = normalize(V1.block(0, j, V1.rows(), 1), B);
		//cout << V1.col(0) << endl;
		//if (V2.cols() > 0)
		//	cout << V1.col(0).transpose() * B * V2.col(0) << endl;
		if (flag) {
			V1.col(j) = V1.col(V1.cols() - 1 - dep);
			++dep;
			--j;
		}
	}
	return dep;
}
int LinearEigenSolver::conv_select(MatrixXd& eval, MatrixXd& evec, double shift, MatrixXd& valout, MatrixXd& vecout) {
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
int LinearEigenSolver::conv_check(Map<MatrixXd>& eval, Map<MatrixXd>& evec, double shift) {
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

LinearEigenSolver::LinearEigenSolver(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev) : A(A), B(B), nev(nev), nIter(0) {
	eigenvectors.resize(A.rows(), 0);
}
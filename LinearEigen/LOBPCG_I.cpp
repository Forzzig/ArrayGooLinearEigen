#include<LOBPCG_I.h>
#include<iostream>

using namespace std;
LOBPCG_I::LOBPCG_I(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep)
	: LinearEigenSolver(A, B, nev),
		storage(new double[A.rows() * nev * 3]),
		X(storage + A.rows() * nev, A.rows(), nev),
		P(storage + A.rows() * nev * 2, A.rows(), 0),
		W(storage, A.rows(), nev) {
	X = MatrixXd::Random(A.rows(), nev);
	orthogonalization(X, B);
	cout << "X初始化" << endl;
	linearsolver.compute(A);
	linearsolver.setMaxIterations(cgstep);
	cout << "CG求解器准备完成..." << endl;
	cout << "初始化完成" << endl;
}

void LOBPCG_I::compute() {
	int cnv = 0;
	MatrixXd eval, evec, tmp, tmpA, mu;
	Map<MatrixXd> V(storage, A.rows(), X.cols() + W.cols()), vl(storage, 0, 0);
	while (true) {
		++nIter;
		cout << "迭代步：" << nIter << endl;
		tmpA = A * X;
		tmp = B * X;
		for (int i = 0; i < nev; ++i) {
			mu = X.col(i).transpose() * A * X.col(i);
			tmp.col(i) *= mu(0, 0);
		}
		W = linearsolver.solve(tmpA - tmp);
		tmp = X;
		cout << W.cols() << " " << X.cols() << " " << P.cols()<< endl;
		new (&V) Map<MatrixXd>(storage, A.rows(), W.cols() + X.cols() + P.cols());
		int dep = orthogonalization(V, B);
		new (&V) Map<MatrixXd>(storage, A.rows(), W.cols() + X.cols() + P.cols() - dep);
		cout << V.cols() << endl;
		
		projection_RR(V, A, eval, evec);

		system("cls");
		cout << V.rows() << " " << V.cols() << " " << evec.rows() << " " << evec.cols() << endl;
		X = V * evec.block(0, 0, V.cols(), nev);
		new (&P) Map<MatrixXd>(storage + A.rows() * (W.cols() + X.cols()), A.rows(), nev);
		P = tmp;
		new (&vl) Map<MatrixXd>(&(eval(0, 0)), nev, 1);
		
		cnv = conv_check(vl, X, 0);
		cout << "已收敛特征向量个数：" << cnv << endl;

		if (cnv >= nev) {
			eigenvectors = X;
			for (int i = 0; i < nev; ++i)
				eigenvalues.push_back(eval(i, 0));
			break;
		}
	}
}
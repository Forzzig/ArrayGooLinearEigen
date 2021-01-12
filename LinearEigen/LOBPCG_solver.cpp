#include<LOBPCG_solver.h>

LOBPCG_solver::LOBPCG_solver(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep) : LinearEigenSolver(A, B, nev) {
	X = MatrixXd::Random(A.rows(), nev);
	orthogonalization(X, B);
	linearsolver.compute(A);
	linearsolver.setMaxIterations(cgstep);
	P.resize(A.rows(), 0);
	eigenvectors.resize(A.rows(), 0);
	LAM.resize(nev, 1);
	cout << "初始化完成" << endl;
}
void LOBPCG_solver::compute(){

	int cnv = 0;
	MatrixXd eval, evec, Xnew, tmp;
	MatrixXd tmpA, rho;
	while (true) {
		++nIter;
		cout << "迭代步：" << nIter << endl;
		//cout << "移频：" << shift << endl;
		tmpA = A * X;
		tmp = B * X;
		for (int i = 0; i < nev; ++i) {
			rho = X.col(i).transpose() * A * X.col(i);
			tmp.col(i) *= rho(0, 0);
		}
		W = linearsolver.solve(tmpA - tmp);
		cout << W << endl;
		system("pause");
		//W = linearsolver.solve(A * X);

		if (nIter > 1) {
			V.resize(A.rows(), nev + W.cols() + P.cols());
		}
		else {
			V.resize(A.rows(), nev + W.cols());
		}
		V << W, X, P;
		cout << V.cols() << endl;
		orthogonalization(V, eigenvectors, B);
		orthogonalization(V, B);
		//cout << V.transpose() * B * V << endl;
		projection_RR(V, A, eval, evec);
		
		cout << V.rows() << " " << V.cols() << " " << evec.rows() << " " << evec.cols() << endl;
		Xnew = V * evec;
		P = X;
		X.resize(A.rows(), nev);
		system("cls");
		cnv = conv_select(eval, Xnew, 0, LAM, X);
		cout << "已收敛特征向量个数：" << cnv << endl;
		if (cnv >= nev)
			break;
	}
}
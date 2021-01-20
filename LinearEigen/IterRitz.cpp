#include<IterRitz.h>

IterRitz::IterRitz(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int q, int r) : LinearEigenSolver(A, B, nev){
	this->q = q;
	this->r = r;
	X = MatrixXd::Random(A.rows(), q);
	orthogonalization(X, B);
	MatrixXd evec;
	projection_RR(X, A, LAM, evec);
	X = X * evec;
	linearsolver.compute(A);
	linearsolver.setMaxIterations(cgstep);
	cout << "CG求解器准备完成..." << endl;
	nIter = 0;
	cout << "初始化完成" << endl;
}

void IterRitz::compute() {
	double shift = 0;
	int cnv = 0;
	MatrixXd eval, evec, Xnew, tmp, rls;
	SparseMatrix<double> tmpA;
	P.resize(A.rows(), 0);
	while (true) {
		++nIter;
		cout << "迭代步：" << nIter << endl;
		cout << "移频：" << shift << endl;
		X1.resize(A.rows(), q * r);
		for (int i = 0; i < r; ++i) {
			if (i == 0) {
				tmp = B * X;
				for (int j = 0; j < q; ++j) {
					tmp.col(j) *= LAM(j, 0);
				}
				X1.block(0, 0, A.rows(), q) = linearsolver.solveWithGuess(tmp, X);
				
				tmp = X1.block(0, 0, A.rows(), q);
				rls = tmp.transpose() * A * tmp;
				for (int j = 0; j < q; ++j)
					LAM(j, 0) = rls(j, j);
				orthogonalization(X1.block(0, 0, A.rows(), q), eigenvectors, B);
				orthogonalization(X1.block(0, 0, A.rows(), q), B);
			}
			else {
				/*tmp = B * X1.block(0, (i - 1) * q, A.rows(), q);
				for (int j = 0; j < q; ++j) {
					tmp.col(j) *= LAM(j, 0);
				}
				X1.block(0, i * q, A.rows(), q) = linearsolver.solveWithGuess(tmp, X1.block(0, (i - 1) * q, A.rows(), q));
				*/
				rls = A * tmp;
				for (int j = 0; j < q; ++j)
					rls.col(j) -= B * tmp.col(j) * LAM(j, 0);
				X1.block(0, i * q, A.rows(), q) = linearsolver.solve(rls); 
				tmp -= X1.block(0, i * q, A.rows(), q);
			}
			orthogonalization(X1.block(0, i * q, A.rows(), q), eigenvectors, B);
			orthogonalization(X1.block(0, i * q, A.rows(), q), X1.block(0, 0, A.rows(), i * q), B);
			orthogonalization(X1.block(0, i * q, A.rows(), q), B);
		}
		orthogonalization(P, X1, B);
		orthogonalization(P, B);
		V.resize(A.rows(), P.cols() + X1.cols());
		V << X1, P;
		//orthogonalization(V, eigenvectors, B);
		//orthogonalization(V, B);
		cout << V.cols() << endl;
		projection_RR(V, A, eval, evec);
		cout << V.rows() << " " << V.cols() << " " << evec.rows() << " " << evec.cols() << endl;
		/*orthogonalization(X1, eigenvectors, B);
		orthogonalization(X1, B);
		projection_RR(X1, A, eval, evec);

		Xnew = X1 * evec;*/
		Xnew = V * evec;
		//cout << Xnew << endl;
		system("cls");
		P = X;
		cnv = conv_select(eval, Xnew, shift, LAM, X);
		//system("pause");
		cout << "已收敛特征向量个数：" << cnv << endl;

		if (cnv >= nev) {
			break;
		}

	}
}
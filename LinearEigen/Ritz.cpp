#include<Ritz.h>

Ritz::Ritz(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int q, int r) : LinearEigenSolver(A, B, nev) {
	this->q = q;
	this->r = r;
	this->cgstep = cgstep;
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

void Ritz::compute() {
	double shift = 0;
	int cnv = 0;
	MatrixXd eval, evec, Xnew, tmp, rls;
	SparseMatrix<double> tmpA;
	X1.resize(A.rows(), q * r);
	P.resize(A.rows(), 0);
	while (true) {
		time_t now = time(&now);
		if (timeCheck(start_time, now))
			break;

		++nIter;
		cout << "迭代步：" << nIter << endl;
		cout << "移频：" << shift << endl;
		
		tmp = B * X;
		com_of_mul += B.nonZeros();

		for (int j = 0; j < q; ++j) {
			tmp.col(j) *= LAM(j, 0);
		}
		com_of_mul += q * A.rows();

		X1.leftCols(q) = linearsolver.solveWithGuess(tmp, X);
		com_of_mul += X.cols() * (A.nonZeros() + 4 * A.rows() +
			cgstep * (A.nonZeros() + 7 * A.rows()));

		orthogonalization(X1.leftCols(q), eigenvectors, B);
		com_of_mul += q * eigenvectors.cols() * A.rows() * 2;

		orthogonalization(X1.leftCols(q), B);
		com_of_mul += (q + 1) * q * A.rows();

		for (int i = 1; i < r; ++i) {
			
			tmp = B * X1.middleCols((i - 1) * q, q);
			com_of_mul += B.nonZeros() * q;
				
			for (int j = 0; j < q; ++j) {
				tmp.col(j) *= LAM(j, 0);
			}
			com_of_mul += A.rows() * q;

			X1.middleCols(i * q, q) = linearsolver.solveWithGuess(tmp, X1.middleCols((i - 1) * q, q));
			com_of_mul += tmp.cols() * (A.nonZeros() + 4 * A.rows() +
				cgstep * (A.nonZeros() + 7 * A.rows()));

			orthogonalization(X1.middleCols(i * q, q), eigenvectors, B);
			com_of_mul += q * eigenvectors.cols() * A.rows() * 2;

			orthogonalization(X1.middleCols(i * q, q), X1.leftCols(i * q), B);
			com_of_mul += q * i * q * A.rows() * 2;

			orthogonalization(X1.middleCols(i * q, q), B);
			com_of_mul += (q + 1) * q * A.rows();
		}
		
		cout << X1.cols() << endl;
		projection_RR(X1, A, eval, evec);
		com_of_mul += X1.cols() * A.nonZeros() * X1.cols() + (24 * X1.cols() * X1.cols() * X1.cols());

		cout << X1.rows() << " " << X1.cols() << " " << evec.rows() << " " << evec.cols() << endl;
		/*orthogonalization(X1, eigenvectors, B);
		orthogonalization(X1, B);
		projection_RR(X1, A, eval, evec);

		Xnew = X1 * evec;*/

		Xnew = X1 * evec;
		com_of_mul += A.rows() * X1.cols() * evec.cols();

		//cout << Xnew << endl;
		system("cls");
		/*P = X;*/

		cnv = conv_select(eval, Xnew, shift, LAM, X);
		com_of_mul += (A.nonZeros() + B.nonZeros() + 3 * A.rows()) * LinearEigenSolver::CHECKNUM;
		//system("pause");
		cout << "已收敛特征向量个数：" << cnv << endl;
		/*if (cnv > 0)
			system("pause");*/
		if (cnv >= nev) {
			break;
		}

	}
}
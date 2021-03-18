#include<Ritz.h>

Ritz::Ritz(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int q, int r)
	: LinearEigenSolver(A, B, nev),
	q(q),
	r(r),
	cgstep(cgstep),
	X(MatrixXd::Random(A.rows(), q)) {
	
	orthogonalization(X, B);
	MatrixXd evec;
	projection_RR(X, A, LAM, evec);
	X *= evec;
	linearsolver.compute(A);
	linearsolver.setMaxIterations(cgstep);
	cout << "CG求解器准备完成..." << endl;
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
		
		projection_RR(X1, A, eval, evec);
		com_of_mul += X1.cols() * A.nonZeros() * X1.cols() + (24 * X1.cols() * X1.cols() * X1.cols());

		Xnew = X1 * evec;
		com_of_mul += A.rows() * X1.cols() * evec.cols();

		system("cls");
		cnv = conv_select(eval, Xnew, shift, LAM, X);
		com_of_mul += (A.nonZeros() + B.nonZeros() + 3 * A.rows()) * LinearEigenSolver::CHECKNUM;
		cout << "已收敛特征向量个数：" << cnv << endl;

		if (cnv >= nev) {
			break;
		}

		time_t now = time(&now);
		if (timeCheck(start_time, now))
			break;

	}
}
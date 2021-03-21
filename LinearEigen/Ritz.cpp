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
		
#ifndef DIRECT
	linearsolver.setMaxIterations(cgstep);
#else
	SparseMatrix<double> L = linearsolver.matrixL();
	int* bandwidth = new int[A.rows()];
	memset(bandwidth, 255, sizeof(int) * A.rows());
	for (int k = 0; k < L.cols(); ++k)
		for (SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
			if (bandwidth[it.row()] < 0)
				bandwidth[it.row()] = k;
	
	for (int k = 0; k < A.rows(); ++k) {
		if (bandwidth[k] < 0)
			bandwidth[k] = 0;
		else
			bandwidth[k] = k - bandwidth[k];
		com_of_mul += bandwidth[k] * (bandwidth[k] - 1) / 2;
	}
	com_of_mul += 6 * L.nonZeros() + A.nonZeros();
	delete bandwidth;
#endif // !DIRECT

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
	Map<MatrixXd> X0(&X(0, 0), A.rows(), q);

	while (true) {
		++nIter;
		cout << "迭代步：" << nIter << endl;
		cout << "移频：" << shift << endl;

		new (&X0) Map<MatrixXd>(&X(0, 0), X.rows(), X.cols());
		
		
		for (int i = 0; i < r; ++i) {
			
			coutput << "X0--------------------------------" << endl << X0 << endl;
			tmp = B * X0;
			com_of_mul += B.nonZeros() * q;

#ifndef DIRECT			
			for (int j = 0; j < q; ++j) {
				tmp.col(j) *= LAM(j, 0);
			}
			com_of_mul += A.rows() * q;

			X1.middleCols(i * q, q) = linearsolver.solveWithGuess(tmp, X0);
			com_of_mul += tmp.cols() * (A.nonZeros() + 4 * A.rows() +
				cgstep * (A.nonZeros() + 7 * A.rows()));
#else
			X1.middleCols(i * q, q) = linearsolver.solve(tmp);
			com_of_mul += 2 * L.nonZeros() + 5 * A.rows();
#endif // !DIRECT

			orthogonalization(X1.middleCols(i * q, q), eigenvectors, B);
			orthogonalization(X1.middleCols(i * q, q), X1.leftCols(i * q), B);
			orthogonalization(X1.middleCols(i * q, q), B);
			
			new (&X0) Map<MatrixXd>(&X1(0, i * q), A.rows(), q);
		}
		
		projection_RR(X1, A, eval, evec);

		Xnew = X1 * evec.leftCols(2 * q);
		com_of_mul += A.rows() * X1.cols() * 2 * q;

		system("cls");
		cnv = conv_select(eval, Xnew, shift, LAM, X);
		cout << "Ritz法已收敛特征向量个数：" << cnv << endl;

		if (cnv >= nev) {
			break;
		}

		time_t now = time(&now);
		if (timeCheck(start_time, now))
			break;
	}
	finish();
}
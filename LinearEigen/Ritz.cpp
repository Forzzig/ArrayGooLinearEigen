#include<Ritz.h>

Ritz::Ritz(SparseMatrix<double, RowMajor, __int64>& A, SparseMatrix<double, RowMajor, __int64>& B, int nev, int cgstep, int q, int r)
	: LinearEigenSolver(A, B, nev),
	q(q),
	r(r),
	cgstep(cgstep),
	X(MatrixXd::Random(A.rows(), q)) {
	
	int dep = orthogonalization(X, B);
	while (dep) {
		X.rightCols(dep) = MatrixXd::Random(A.rows(), dep);
		dep = orthogonalization(X, B);
	}
	MatrixXd evec;
	projection_RR(X, A, LAM, evec);
	X *= evec;

	//A为RowMajor，利用对称性获取一个ColMajor稀疏矩阵
	//linearsolver.compute(A.transpose());
	linearsolver.compute(A);
		
#ifndef DIRECT
	linearsolver.setMaxIterations(cgstep);
#else
	/*SparseMatrix<double> L = linearsolver.matrixL();
	L_nnz = L.nonZeros();
	long long* bandwidth = new long long[A.rows()];
	memset(bandwidth, 0, sizeof(long long) * A.rows());
	for (int k = 0; k < L.cols(); ++k)
		for (SparseMatrix<double, RowMajor, __int64>::InnerIterator it(L, k); it; ++it)
			++bandwidth[it.row()];
	
	for (int k = 0; k < A.rows(); ++k)
		com_of_mul += bandwidth[k] * (bandwidth[k] - 1) / 2;
	delete[] bandwidth;*/

	com_of_mul += 50 * A.nonZeros() + A.nonZeros();
	
#endif // !DIRECT

	cout << "CG求解器准备完成..." << endl;
	cout << "初始化完成" << endl;
}

void Ritz::compute() {
	double shift = 0;
	int cnv = 0;
	MatrixXd eval, evec, Xnew, tmp, rls;
	SparseMatrix<double, RowMajor, __int64> tmpA;
	V.resize(A.rows(), q * (r + 1));
	P.resize(A.rows(), 0);
	Map<MatrixXd> X0(&X(0, 0), A.rows(), q);
	Map<MatrixXd> X1(&V(0, 0), A.rows(), q);
	Map<MatrixXd> V0(&V(0, 0), A.rows(), 0);
	while (true) {
		++nIter;
		cout << "迭代步：" << nIter << endl;
		cout << "移频：" << shift << endl;

		new (&X0) Map<MatrixXd>(&X(0, 0), X.rows(), X.cols());
		new (&X1) Map<MatrixXd>(&V(0, 0), V.rows(), X.cols());
		new (&V0) Map<MatrixXd>(&V(0, 0), V.rows(), 0);
		for (int i = 0; i < r; ++i) {
			tmp = B * X0;
			com_of_mul += B.nonZeros() * q;

#ifndef DIRECT			
			for (int j = 0; j < q; ++j) {
				tmp.col(j) *= LAM(j, 0);
			}
			com_of_mul += A.rows() * q;

			X1 = linearsolver.solveWithGuess(tmp, X0);
			com_of_mul += tmp.cols() * (A.nonZeros() + 4 * A.rows() +
				cgstep * (A.nonZeros() + 7 * A.rows()));
#else
			X1 = linearsolver.solve(tmp);
			com_of_mul += 5 * A.nonZeros() + 5 * A.rows();
#endif // !DIRECT

			orthogonalization(X1, eigenvectors, B);
			orthogonalization(X1, V0, B);
			int dep = orthogonalization(X1, B);
			
			new (&V0) Map<MatrixXd>(&V(0, 0), A.rows(), V0.cols() + X1.cols() - dep);
			new (&X0) Map<MatrixXd>(&X1(0, 0), A.rows(), X1.cols() - dep);
			new (&X1) Map<MatrixXd>(&V(0, V0.cols()), A.rows(), X0.cols());
		}
		projection_RR(V0, A, eval, evec);

		Xnew = V0 * evec.leftCols(2 * q);
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
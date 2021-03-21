#include<IterRitz.h>

IterRitz::IterRitz(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int q, int r)
	: LinearEigenSolver(A, B, nev),
	q(q),
	r(r),
	cgstep(cgstep),
	X(MatrixXd::Random(A.rows(), q)),
	V(A.rows(), q * (r + 1)),
	P(A.rows(), 0) {
	
	orthogonalization(X, B);
	MatrixXd evec;
	projection_RR(X, A, Lam, evec);
	X *= evec;
	linearsolver.compute(A);
	linearsolver.setMaxIterations(cgstep);
	cout << "CG求解器准备完成..." << endl;
	cout << "初始化完成" << endl;
}

void IterRitz::compute() {
	double shift = 0;
	MatrixXd eval, evec, Xnew, BX, realX, AX;
	Map<MatrixXd> X1(&V(0, 0), A.rows(), X.cols());
	Map<MatrixXd> V0(&V(0, 0), A.rows(), 0);
	while (true) {
		++nIter;
		cout << "迭代步：" << nIter << endl;
		cout << "移频：" << shift << endl;
		
		new (&X1) Map<MatrixXd>(&V(0, 0), A.rows(), X.cols());
		new (&V0) Map<MatrixXd>(&V(0, 0), A.rows(), 0);

		realX = X;
		BX = B * realX;
		com_of_mul += B.nonZeros() * realX.cols();

		AX = A * realX;
		com_of_mul += realX.cols() * A.nonZeros();

		for (int i = 0; i < r; ++i) {
			for (int j = 0; j < BX.cols(); ++j) {
				BX.col(j) *= Lam(j, 0);
			}
			com_of_mul += A.rows() * BX.cols();

			X1 = linearsolver.solve(BX - AX);
			com_of_mul += BX.cols() * (A.nonZeros() + 4 * A.rows() +
				cgstep * (A.nonZeros() + 7 * A.rows()));

			realX += X1;

			if (i == 0)
				X1 = realX;

			BX = B * realX;
			com_of_mul += B.nonZeros() * realX.cols();

			AX = A * realX;
			com_of_mul += realX.cols() * A.nonZeros();

			for (int j = 0; j < BX.cols(); ++j)
				Lam(j, 0) = (realX.col(j).transpose() * AX.col(j))(0, 0) / (realX.col(j).transpose() * BX.col(j))(0, 0);
			com_of_mul += 2 * A.rows() * BX.cols();

			orthogonalization(X1, eigenvectors, B);
			orthogonalization(X1, V0, B);
			int dep = orthogonalization(X1, B);
			new (&V0) Map<MatrixXd>(&V(0, 0), A.rows(), V0.cols() + X1.cols() - dep);
			new (&X1) Map<MatrixXd>(&V0(A.rows() - 1, V0.cols() - 1) + 1, A.rows(), realX.cols());
		}

		if (P.cols() != X1.cols())
			new (&X1) Map<MatrixXd>(&X1(0, 0), A.rows(), P.cols());
		X1 = P;
		orthogonalization(X1, eigenvectors, B);
		orthogonalization(X1, V0, B);
		int dep = orthogonalization(X1, B);
		new (&V0) Map<MatrixXd>(&V(0, 0), A.rows(), V0.cols() + X1.cols() - dep);
		projection_RR(V0, A, eval, evec);

		int nd = (2 * q < V0.cols()) ? 2 * q : V0.cols();
		Xnew = V0 * evec.leftCols(nd);
		com_of_mul += A.rows() * V.cols() * nd;

		P = X;

		system("cls");
		int prev = eigenvalues.size();
		int cnv = conv_select(eval, Xnew, shift, Lam, X);
		com_of_mul += (A.nonZeros() + B.nonZeros() + 3 * A.rows()) * LinearEigenSolver::CHECKNUM;
		
		cout << "已收敛特征向量个数：" << cnv << endl;

		if (cnv >= nev) {
			break;
		}

		time_t now = time(&now);
		if (timeCheck(start_time, now))
			break;

		if (nd - (cnv - prev) < X.cols())
			X.conservativeResize(A.rows(), nd - (cnv - prev));
		orthogonalization(X, eigenvectors, B);
		orthogonalization(X, B);
	}
	finish();
}
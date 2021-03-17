#include<IterRitz.h>

IterRitz::IterRitz(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int q, int r) : LinearEigenSolver(A, B, nev){
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

void IterRitz::compute() {
	double shift = 0;
	int cnv = 0;
	MatrixXd eval, evec, Xnew, tmp, rls;
	SparseMatrix<double> tmpA;
	X1.resize(A.rows(), q * (r + 1));
	P.resize(A.rows(), 0);
	Map<MatrixXd> V(&X1(0, 0), A.rows(), 0);
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
		
		tmp = X1.leftCols(q);
		rls = A * tmp;
		com_of_mul += q * A.nonZeros();

		for (int j = 0; j < q; ++j)
			LAM(j, 0) = (tmp.col(j).transpose() * rls.col(j))(0, 0);

		orthogonalization(X1.leftCols(q), eigenvectors, B);
		com_of_mul += q * eigenvectors.cols() * A.rows() * 2;

		orthogonalization(X1.leftCols(q), B);
		com_of_mul += (q + 1) * q * A.rows();
		for (int i = 1; i < r; ++i) {

			for (int j = 0; j < q; ++j)
				rls.col(j) -= B * tmp.col(j) * LAM(j, 0);
			com_of_mul += q * (B.nonZeros() + A.rows());

			X1.middleCols(i * q, q) = linearsolver.solve(rls);
			com_of_mul += tmp.cols() * (A.nonZeros() + 4 * A.rows() +
				cgstep * (A.nonZeros() + 7 * A.rows()));

			tmp -= X1.middleCols(i * q, q);
			rls = A * tmp;

			orthogonalization(X1.middleCols(i * q, q), eigenvectors, B);
			com_of_mul += q * eigenvectors.cols() * A.rows() * 2;

			orthogonalization(X1.middleCols(i * q, q), X1.leftCols(i * q), B);
			com_of_mul += q * i * q * A.rows() * 2;

			orthogonalization(X1.middleCols(i * q, q), B);
			com_of_mul += (q + 1) * q * A.rows();
		}

		if (nIter > 1) {
			orthogonalization(P, eigenvectors, B);
			com_of_mul += P.cols() * eigenvectors.cols() * A.rows() * 2;
			
			X1.middleCols(r * q, q) = P;
			int dep = orthogonalization(X1.leftCols(q * (r + 1)), B);
			//com_of_mul += (q * (r + 1) + 1) * q * (r + 1) * A.rows();

			new (&V) Map<MatrixXd>(&X1(0, 0), A.rows(), q * (r + 1) - dep);
		}
		else {
			int dep = orthogonalization(X1.leftCols(q * r), B);
			new (&V) Map<MatrixXd>(&X1(0, 0), A.rows(), q * r - dep);
		}
		cout << V.cols() << endl;
		projection_RR(V, A, eval, evec);
		com_of_mul += V.cols() * A.nonZeros() * V.cols() + (24 * V.cols() * V.cols() * V.cols());

		cout << V.rows() << " " << V.cols() << " " << evec.rows() << " " << evec.cols() << endl;
		
		Xnew = V * evec;
		com_of_mul += A.rows() * V.cols() * evec.cols();

		//cout << Xnew << endl;
		system("cls");
		P = X;
		
		cnv = conv_select(eval, Xnew, shift, LAM, X);
		com_of_mul += (A.nonZeros() + B.nonZeros() + 3 * A.rows()) * LinearEigenSolver::CHECKNUM;
		//system("pause");
		cout << "已收敛特征向量个数：" << cnv << endl;

		if (cnv >= nev) {
			break;
		}

	}
}
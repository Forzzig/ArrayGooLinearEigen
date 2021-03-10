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
	P.resize(A.rows(), 0);
	while (true) {
		++nIter;
		cout << "迭代步：" << nIter << endl;
		cout << "移频：" << shift << endl;
		X1.resize(A.rows(), q * r);
		for (int i = 0; i < r; ++i) {
			if (i == 0) {
				
				tmp = B * X;
				com_of_mul += B.nonZeros();

				for (int j = 0; j < q; ++j) {
					tmp.col(j) *= LAM(j, 0);
				}
				com_of_mul += q * A.rows();

				X1.block(0, 0, A.rows(), q) = linearsolver.solveWithGuess(tmp, X);
				com_of_mul += X.cols() * (A.nonZeros() + 4 * A.rows() +
					cgstep * (A.nonZeros() + 7 * A.rows()));

				tmp = X1.block(0, 0, A.rows(), q);
				rls = tmp.transpose() * A * tmp;
				com_of_mul += q * (A.nonZeros() + A.rows());

				for (int j = 0; j < q; ++j)
					LAM(j, 0) = rls(j, j);

				orthogonalization(X1.block(0, 0, A.rows(), q), eigenvectors, B);
				com_of_mul += q * eigenvectors.cols() * A.rows() * 2;

				orthogonalization(X1.block(0, 0, A.rows(), q), B);
				com_of_mul += (q + 1) * q * A.rows();
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
				com_of_mul += q * (B.nonZeros() + A.rows());

				X1.block(0, i * q, A.rows(), q) = linearsolver.solve(rls);
				com_of_mul += tmp.cols() * (A.nonZeros() + 4 * A.rows() +
					cgstep * (A.nonZeros() + 7 * A.rows()));

				tmp -= X1.block(0, i * q, A.rows(), q);
			}
			orthogonalization(X1.block(0, i * q, A.rows(), q), eigenvectors, B);
			com_of_mul += q * eigenvectors.cols() * A.rows() * 2;

			orthogonalization(X1.block(0, i * q, A.rows(), q), X1.block(0, 0, A.rows(), i * q), B);
			com_of_mul += q * i * q * A.rows() * 2;

			orthogonalization(X1.block(0, i * q, A.rows(), q), B);
			com_of_mul += (q + 1) * q * A.rows();
		}
		orthogonalization(P, X1, B);
		com_of_mul += P.cols() * X1.cols() * A.rows() * 2;

		orthogonalization(P, B);
		com_of_mul += (P.cols() + 1) * P.cols() * A.rows();

		V.resize(A.rows(), P.cols() + X1.cols());
		V << X1, P;
		//orthogonalization(V, eigenvectors, B);
		//orthogonalization(V, B);
		cout << V.cols() << endl;
		projection_RR(V, A, eval, evec);
		com_of_mul += V.cols() * A.nonZeros() * V.cols() + (V.cols() * V.cols() * V.cols());

		cout << V.rows() << " " << V.cols() << " " << evec.rows() << " " << evec.cols() << endl;
		/*orthogonalization(X1, eigenvectors, B);
		orthogonalization(X1, B);
		projection_RR(X1, A, eval, evec);

		Xnew = X1 * evec;*/
		
		Xnew = V * evec;
		com_of_mul += A.rows() * V.cols() * V.cols();

		//cout << Xnew << endl;
		system("cls");
		P = X;
		
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
#include<IterRitz.h>

IterRitz::IterRitz(SparseMatrix<double, RowMajor, __int64>& A, SparseMatrix<double, RowMajor, __int64>& B, int nev, int cgstep, int q, int r)
	: LinearEigenSolver(A, B, nev),
	q(q),
	r(r),
	cgstep(cgstep),
	X(MatrixXd::Random(A.rows(), q)),
	V(A.rows(), q* (r + 2)),
	P(A.rows(), 0),
	Lam(q, 1),
	CA(q* (r + 2), q* (r + 2)),
	CB(q* (r + 2), q* (r + 2)),
	CAB(q* (r + 2), q* (r + 2)),
	CBA(q* (r + 2), q* (r + 2)) {

	int dep = orthogonalization(X, B);
	while (dep) {
		X.rightCols(dep) = MatrixXd::Random(A.rows(), dep);
		dep = orthogonalization(X, B);
	}

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
	VectorXd eval(q * (r + 2));
	MatrixXd evec(q * (r + 2), q * (r + 2)), Xnew(A.rows(), 2 * q), BX(A.rows(), q * (r + 2)), realX(A.rows(), q), AX(A.rows(), q * (r + 2));
	Map<MatrixXd> X1(&V(0, 0), A.rows(), X.cols());
	Map<MatrixXd> V0(&V(0, 0), A.rows(), 0);
	MatrixXd H(q * (r + 2), q * (r + 2));
	while (true) {
		++nIter;
		cout << "迭代步：" << nIter << endl;
		cout << "移频：" << shift << endl;

		new (&X1) Map<MatrixXd>(&V(0, 0), A.rows(), X.cols());
		new (&V0) Map<MatrixXd>(&V(0, 0), A.rows(), 0);

		if (realX.cols() != X.cols())
			realX.resize(NoChange, X.cols());

#pragma omp parallel for
		for (int i = 0; i < X.cols(); ++i) {
			memcpy(&realX(0, i), &X(0, i), A.rows() * sizeof(double));
			BX.col(i).noalias() = B * realX.col(i);
			AX.col(i).noalias() = A * realX.col(i);
		}
		com_of_mul += B.nonZeros() * realX.cols();
		com_of_mul += realX.cols() * A.nonZeros();

		for (int i = 0; i < r; ++i) {
#pragma omp parallel for
			for (int j = 0; j < realX.cols(); ++j) {
				memset(&X1(0, j), 0, A.rows() * sizeof(double));
				linearsolver._solveWithGuess(BX.col(j) * Lam(j, 0) - AX.col(j), X1.col(j));
				realX.col(j) += X1.col(j);

				if (i == 0)
					memcpy(&X1(0, j), &realX(0, j), A.rows() * sizeof(double));
			}
			com_of_mul += A.rows() * BX.cols();
			com_of_mul += BX.cols() * (A.nonZeros() + 4 * A.rows() +
				cgstep * (A.nonZeros() + 7 * A.rows()));

			orthogonalization(X1, eigenvectors, B);
			orthogonalization(X1, V0, B);
			int dep = orthogonalization(X1, B);
			new (&V0) Map<MatrixXd>(&V(0, 0), A.rows(), V0.cols() + X1.cols() - dep);
			new (&X1) Map<MatrixXd>(&V0(A.rows() - 1, V0.cols() - 1) + 1, A.rows(), realX.cols());

			if (i < r - 1) {
#pragma omp parallel for
				for (int j = 0; j < realX.cols(); ++j) {
					BX.col(j).noalias() = B * realX.col(j);
					AX.col(j).noalias() = A * realX.col(j);
					Lam(j, 0) = realX.col(j).dot(AX.col(j)) / realX.col(j).dot(BX.col(j));
				}
				com_of_mul += B.nonZeros() * realX.cols();
				com_of_mul += realX.cols() * A.nonZeros();
				com_of_mul += 2 * A.rows() * BX.cols();
			}
		}
		if (P.cols() != X1.cols())
			new (&X1) Map<MatrixXd>(&X1(0, 0), A.rows(), P.cols());
		if (P.cols())
			memcpy(&X1(0, 0), &P(0, 0), A.rows() * P.cols() * sizeof(double));
		//X1 = P;

		orthogonalization(X1, eigenvectors, B);
		orthogonalization(X1, V0, B);
		int dep = orthogonalization(X1, B);
		new (&V0) Map<MatrixXd>(&V(0, 0), A.rows(), V0.cols() + X1.cols() - dep);

#pragma omp parallel for
		for (int j = 0; j < V0.cols(); ++j) {
			AX.col(j).noalias() = A * V0.col(j);
			BX.col(j).noalias() = B * V0.col(j);
		}
		com_of_mul += A.nonZeros() * V0.cols();
		com_of_mul += B.nonZeros() * V0.cols();

		H.topLeftCorner(V0.cols(), V0.cols()).noalias() = V0.transpose() * AX.leftCols(V0.cols());
		com_of_mul += V0.cols() * A.rows() * V0.cols();

		RR(H.topLeftCorner(V0.cols(), V0.cols()), eval, evec);

		int nd = (2 * q < V0.cols()) ? 2 * q : V0.cols();
		if (Xnew.cols() != nd)
			Xnew.resize(NoChange, nd);
		Xnew.noalias() = V0 * evec.leftCols(nd);
		com_of_mul += A.rows() * V.cols() * nd;

		//if (P.cols() != 2 * X.cols())
		//	P.resize(NoChange, 2 * X.cols());
		//memcpy(&P(0, 0), &X(0, 0), A.rows() * X.cols() * sizeof(double));
		//int Xc = X.cols();
		if (P.cols() != X.cols())
			P.resize(NoChange, X.cols());
		memcpy(&P(0, 0), &X(0, 0), A.rows() * X.cols() * sizeof(double));

		system("cls");
		int prev = eigenvalues.size();
		int cnv = conv_select(eval, Xnew, shift, Lam, X);
		cout << "IterRitz已收敛特征向量个数：" << cnv << endl;

		if (cnv >= nev) {
			break;
		}

		time_t now = time(&now);
		if (timeCheck(start_time, now))
			break;

		if (nd - (cnv - prev) < X.cols())
			X.conservativeResize(NoChange, nd - (cnv - prev));

		/*if (P.cols() != Xc + X.cols())
			P.conservativeResize(NoChange, Xc + X.cols());*/
			//		if (P.cols() != X.cols())
			//			P.resize(NoChange, X.cols());
			//
			//		if (CA.cols() != V0.cols()) {
			//			CA.resize(V0.cols(), V0.cols());
			//			CB.resize(V0.cols(), V0.cols());
			//			CAB.resize(V0.cols(), V0.cols());
			//			CBA.resize(V0.cols(), V0.cols());
			//		}
			//#pragma omp parallel for
			//		for (int j = 0; j < V0.cols(); ++j) {
			//			CA.col(j).noalias() = AX.leftCols(V0.cols()).transpose() * AX.col(j);
			//			CB.col(j).noalias() = BX.leftCols(V0.cols()).transpose() * BX.col(j);
			//			CAB.col(j).noalias() = AX.leftCols(V0.cols()).transpose() * BX.col(j);
			//			CBA.col(j).noalias() = BX.leftCols(V0.cols()).transpose() * AX.col(j);
			//		}
			//		com_of_mul += 4 * V0.cols() * V0.rows() * V0.cols();
			//
			//		vector<int> pos;
			//		vector<int> num;
			//		for (int j = 0; j < X.cols(); ++j) {
			//			int k = j + 1;
			//			while ((k < X.cols()) && ((Lam(k, 0) - Lam(j, 0)) / Lam(j, 0) < ORTH_TOL))
			//				++k;
			//			pos.push_back(j);
			//			num.push_back(k - j);
			//			j = k - 1;
			//		}
			//#pragma omp parallel for
			//		for (int i = 0; i < pos.size(); ++i) {
			//			//refine(Lam(pos[i], 0), V0, P.middleCols(Xc + pos[i], num[i]));
			//			refine(Lam(pos[i], 0), V0, P.middleCols(pos[i], num[i]));
			//		}
	}
	finish();
}
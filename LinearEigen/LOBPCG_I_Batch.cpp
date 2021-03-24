#include<LOBPCG_I_Batch.h>

using namespace std;
LOBPCG_I_Batch::LOBPCG_I_Batch(SparseMatrix<double, RowMajor>& A, SparseMatrix<double, RowMajor>& B, int nev, int cgstep, int batch)
	: LinearEigenSolver(A, B, nev),
	//摆放顺序：XWP
	storage(new double[A.rows() * batch * 3]),
	X(storage, A.rows(), batch),
	W(storage + A.rows() * batch, A.rows(), batch),
	P(storage + A.rows() * batch * 2, A.rows(), 0),

	batch(batch),
	cgstep(cgstep),
	Lam(batch, 1) {

	X = MatrixXd::Random(A.rows(), batch);
	int dep = orthogonalization(X, B);
	while (dep) {
		X.rightCols(dep) = MatrixXd::Random(A.rows(), dep);
		dep = orthogonalization(X, B);
	}
	cout << "X初始化" << endl;

	MatrixXd tmp = A * X;
	for (int i = 0; i < X.cols(); ++i)
		Lam(i, 0) = X.col(i).dot(tmp.col(i));

	//TODO 只有要求解的矩阵不变时才能在这初始化
	linearsolver.compute(A);

	//事先固定CG迭代步数量
	linearsolver.setMaxIterations(cgstep);
	cout << "CG求解器准备完成..." << endl;
	cout << "初始化完成" << endl;
}
LOBPCG_I_Batch::~LOBPCG_I_Batch() {
	delete[] storage;
}

void LOBPCG_I_Batch::compute() {
	MatrixXd eval(batch, 1), evec(batch, batch), AX(A.rows(), batch), BX(A.rows(), batch), Xnew(A.rows(), 2 * batch);
	clock_t t1, t2, T1 = 0, T2 = 0, T3 = 0, T4 = 0, T5 = 0, T6 = 0, T7 = 0, T8 = 0, T9 = 0, T10 = 0;
	long long c1, c2, C1 = 0, C2 = 0, C3 = 0, C4 = 0, C5 = 0, C6 = 0, C7 = 0, C8 = 0, C9 = 0, C10 = 0;
	//所有已计算出来的特征向量和待计算的都依次存在storage里，避免内存拷贝
	Map<MatrixXd> V(storage, A.rows(), X.cols() + W.cols());
	Map<MatrixXd> WP(storage + A.rows() * X.cols(), A.rows(), W.cols());
	while (true) {
		t1 = clock();
		c1 = com_of_mul;

		++nIter;
		cout << "迭代步：" << nIter << endl;

		if (AX.cols() != X.cols())
			AX.resize(NoChange, X.cols());
#pragma omp parallel for
		for (int i = 0; i < X.cols(); ++i) {
			AX.col(i).noalias() = A * X.col(i);
		}
		com_of_mul += A.nonZeros() * X.cols();

		t2 = clock();
		c2 = com_of_mul;
		T1 += t2 - t1;
		C1 += c2 - c1;

		if (BX.cols() != X.cols())
			BX.resize(NoChange, X.cols());
#pragma omp parallel for
		for (int i = 0; i < X.cols(); ++i) {
			//TODO 其实sparse_time_dense_product自带乘标量功能，是可以同时进行的
			BX.col(i).noalias() = B * X.col(i);
			BX.col(i) *= Lam(i, 0);
			BX.col(i) -= AX.col(i);
		}
		com_of_mul += B.nonZeros() * X.cols();
		com_of_mul += A.rows() * X.cols();

		t1 = clock();
		c1 = com_of_mul;
		T2 += t1 - t2;
		C2 += c1 - c2;

		//求解 A*W = A*X - mu*B*X， X为上一步的近似特征向量
		//W = linearsolver.solve(BX);
		memset(&W(0, 0), 0, A.rows() * W.cols() * sizeof(double));

#pragma omp parallel for
		for (int i = 0; i < X.cols(); ++i)
			linearsolver._solveWithGuess(BX.col(i), W.col(i));
		com_of_mul += X.cols() * (A.nonZeros() + 4 * A.rows() +
			cgstep * (A.nonZeros() + 7 * A.rows()));

		t2 = clock();
		c2 = com_of_mul;
		T3 += t2 - t1;
		C3 += c2 - c1;

		//预存上一步的近似特征向量，BX废物利用
		memcpy(&BX(0, 0), &X(0, 0), X.rows() * X.cols() * sizeof(double));
		//BX = X;
		t1 = clock();
		c1 = com_of_mul;
		T4 += t1 - t2;
		C4 += c1 - c2;

		if ((WP.cols() != W.cols() + P.cols()) || (&WP(0, 0) != storage + A.rows() * X.cols()))
			new (&WP) Map<MatrixXd>(storage + A.rows() * X.cols(), A.rows(), W.cols() + P.cols());
		orthogonalization(WP, eigenvectors, B);
		orthogonalization(WP, X, B);

		int dep = orthogonalization(WP, B);

		t2 = clock();
		c2 = com_of_mul;
		T5 += t2 - t1;
		C5 += c2 - c1;

		//正交化后会有线性相关项，剔除
		if (dep)
			new (&WP) Map<MatrixXd>(storage + A.rows() * X.cols(), A.rows(), WP.cols() - dep);

		if (V.cols() != X.cols() + WP.cols()) 
			new (&V) Map<MatrixXd>(storage, A.rows(), X.cols() + WP.cols());

		t1 = clock();
		c1 = com_of_mul;
		T6 += t1 - t2;
		C6 += c1 - c2;

		projection_RR(V, A, eval, evec);

		t2 = clock();
		c2 = com_of_mul;
		T7 += t2 - t1;
		C7 += c2 - c1;

		//子空间V投影下的新的近似特征向量
		int nd = (2 * batch < V.cols()) ? 2 * batch : V.cols();
		Xnew.leftCols(nd).noalias() = V * evec.leftCols(nd);
		com_of_mul += A.rows() * V.cols() * nd;

		system("cls");
		int prev = eigenvalues.size();
		int cnv = conv_select(eval, Xnew, 0, Lam, X);
		cout << "LOBPCG-I已收敛特征向量个数：" << cnv << endl;

		t1 = clock();
		c1 = com_of_mul;
		T8 += t1 - t2;
		C8 += c1 - c2;

		if (cnv >= nev)
			break;

		time_t now = time(&now);
		if (timeCheck(start_time, now))
			break;

		if (nd - (cnv - prev) < X.cols()) {
			int wid = nd - (cnv - prev);
			new (&X) Map<MatrixXd>(storage, A.rows(), wid);
			new (&W) Map<MatrixXd>(storage + A.rows() * X.cols(), A.rows(), wid);
		}
		orthogonalization(X, eigenvectors, B);
		dep = orthogonalization(X, B);
		while (dep) {
			X.rightCols(dep) = MatrixXd::Random(A.rows(), dep);
			dep = orthogonalization(X, B);
		}
		t2 = clock();
		c2 = com_of_mul;
		T9 += t2 - t1;
		C9 += c2 - c1;

		if ((P.cols() != BX.cols()) || (&P(0, 0) != storage + A.rows() * (X.cols() + W.cols()))) 
			new (&P) Map<MatrixXd>(storage + A.rows() * (X.cols() + W.cols()), A.rows(), BX.cols());
		memcpy(&P(0, 0), &BX(0, 0), A.rows() * BX.cols() * sizeof(double));
		//P = BX;

		t1 = clock();
		c1 = com_of_mul;
		T10 += t1 - t2;
		C10 += c1 - c2;
		cout << T1 << " " << C1 << " " << C1 * 1.0 / T1 << endl;
		cout << T2 << " " << C2 << " " << C2 * 1.0 / T2 << endl;
		cout << T3 << " " << C3 << " " << C3 * 1.0 / T3 << endl;
		cout << T4 << " " << C4 << " " << C4 * 1.0 / T4 << endl;
		cout << T5 << " " << C5 << " " << C5 * 1.0 / T5 << endl;
		cout << T6 << " " << C6 << " " << C6 * 1.0 / T6 << endl;
		cout << T7 << " " << C7 << " " << C7 * 1.0 / T7 << endl;
		cout << T8 << " " << C8 << " " << C8 * 1.0 / T8 << endl;
		cout << T9 << " " << C9 << " " << C9 * 1.0 / T9 << endl;
		cout << T10 << " " << C10 << " " << C10 * 1.0 / T10 << endl;
		//system("pause");
	}
	finish();
}
#include<LOBPCG_II_Batch.h>

using namespace std;
LOBPCG_II_Batch::LOBPCG_II_Batch(SparseMatrix<double, RowMajor>& A, SparseMatrix<double, RowMajor>& B, int nev, int cgstep, int batch)
	: LinearEigenSolver(A, B, nev),
	storage(new double[A.rows() * nev * 3]),
	X(storage, A.rows(), batch, OuterStride<>(3 * A.rows())),
	W(storage + A.rows(), A.rows(), batch, OuterStride<>(3 * A.rows())),
	P(storage + 2 * A.rows(), A.rows(), 0, OuterStride<>(3 * A.rows())),
	
	batch(batch),
	Lam(batch, 1),
	cgstep(cgstep) {

	X = MatrixXd::Random(A.rows(), batch);
	int dep = orthogonalization(X, B);
	while (dep) {
		X.rightCols(dep) = MatrixXd::Random(A.rows(), dep);
		dep = orthogonalization(X, B);
	}

	VectorXd tmp(A.rows(), 1);
	for (int i = 0; i < X.cols(); ++i) {
		tmp = A * X.col(i);
		Lam(i, 0) = tmp.dot(X.col(i));
	}
	cout << "X初始化" << endl;

	//只有要求解的矩阵不变时才能在这初始化
	linearsolver.compute(A);

	//事先固定CG迭代步数量
	linearsolver.setMaxIterations(cgstep);
	cout << "CG求解器准备完成..." << endl;
	cout << "初始化完成" << endl;
}

LOBPCG_II_Batch::~LOBPCG_II_Batch() {
	delete[] storage;
}

void LOBPCG_II_Batch::compute() {
	clock_t t1, t2, T1 = 0, T2 = 0, T3 = 0, T4 = 0, T5 = 0, T6 = 0, T7 = 0, T8 = 0, T9 = 0, T10 = 0;
	long long c1, c2, C1 = 0, C2 = 0, C3 = 0, C4 = 0, C5 = 0, C6 = 0, C7 = 0, C8 = 0, C9 = 0, C10 = 0;
	
	VectorXd eval(batch);
	MatrixXd evec(batch, batch), AX(A.rows(), batch), BX(A.rows(), batch);
	vector<Map<MatrixXd>> V, WP;
	vector<SelfAdjointEigenSolver<MatrixXd>> eigsols;
	vector<MatrixXd> tmpAV;
	for (int i = 0; i < batch; ++i) {
		V.push_back(Map<MatrixXd>(storage + A.rows() * 3 * i, A.rows(), 2));
		WP.push_back(Map<MatrixXd>(storage + A.rows() * (3 * i + 1), A.rows(), 1));
		eigsols.push_back(SelfAdjointEigenSolver<MatrixXd>());
		tmpAV.push_back(MatrixXd(A.rows(), 2));
	}
	while (true) {
		t1 = clock();
		c1 = com_of_mul;

		++nIter;
		cout << "迭代步：" << nIter << endl;

		if (AX.cols() != X.cols())
			AX.resize(NoChange, X.cols());
		if (BX.cols() != X.cols())
			BX.resize(NoChange, X.cols());

		long long mults = 0;
#pragma omp parallel for reduction(+:mults)
		for (int i = 0; i < X.cols(); ++i) {
			AX.col(i).noalias() = A * X.col(i);

			//TODO 其实sparse_time_dense_product自带乘标量功能，是可以同时进行的
			BX.col(i).noalias() = B * X.col(i);
			memset(&W(0, i), 0, A.rows() * sizeof(double));
			linearsolver._solveWithGuess(AX.col(i) - BX.col(i) * Lam(i, 0), W.col(i));
		
			//预存上一步的近似特征向量，BX废物利用
			memcpy(&BX(0, i), &X(0, i), A.rows() * sizeof(double));
			
			//对每组XPW分别求解RR
			orthogonalization(WP[i], eigenvectors, B);
			orthogonalization(WP[i], X.col(i), B);
			int dep = orthogonalization(WP[i], B);
			
			if (dep) {
				int WPold = WP[i].cols();
				new (&WP[i]) Map<MatrixXd>(storage + A.rows() * (3 * i + 1), A.rows(), WPold - dep);
				new (&V[i]) Map<MatrixXd>(storage + A.rows() * 3 * i, A.rows(), WPold - dep + 1);
			}
			
			if (tmpAV[i].cols() != V[i].cols())
				tmpAV[i].resize(NoChange, V[i].cols());
			tmpAV[i].noalias() = A * V[i];
			mults += A.nonZeros() * V[i].cols();

			eigsols[i].compute(V[i].transpose() * tmpAV[i]);
			mults += 24 * V[i].cols() * V[i].cols() * V[i].cols();

			X.col(i) = V[i] * eigsols[i].eigenvectors().col(0);
			mults += A.rows() * V[i].cols();

			if (dep || (nIter == 1)) {
				new (&WP[i]) Map<MatrixXd>(storage + A.rows() * (3 * i + 1), A.rows(), 2);
				new (&V[i]) Map<MatrixXd>(storage + A.rows() * 3 * i, A.rows(), 3);
			}
		}
		com_of_mul += A.nonZeros() * X.cols();
		com_of_mul += B.nonZeros() * X.cols();
		com_of_mul += A.rows() * X.cols();
		com_of_mul += X.cols() * (A.nonZeros() + 4 * A.rows() +
			cgstep * (A.nonZeros() + 7 * A.rows()));
		com_of_mul += mults;

		t2 = clock();
		c2 = com_of_mul;
		T1 += t2 - t1;
		C1 += c2 - c1;

		t1 = clock();
		c1 = com_of_mul;
		T2 += t1 - t2;
		C2 += c1 - c2;

		t2 = clock();
		c2 = com_of_mul;
		T3 += t2 - t1;
		C3 += c2 - c1;

		t1 = clock();
		c1 = com_of_mul;
		T4 += t1 - t2;
		C4 += c1 - c2;

		t2 = clock();
		c2 = com_of_mul;
		T5 += t2 - t1;
		C5 += c2 - c1;

		//对X再做一次RR问题
		int dep = orthogonalization(X, B);
		if (dep) 
			new (&X) Map<MatrixXd, Unaligned, OuterStride<>>(storage, A.rows(), X.cols() - dep, OuterStride<>(3 * A.rows()));

		t1 = clock();
		c1 = com_of_mul;
		T6 += t1 - t2;
		C6 += c1 - c2;

		projection_RR(X, A, eval, evec);

		t2 = clock();
		c2 = com_of_mul;
		T7 += t2 - t1;
		C7 += c2 - c1;

		X *= evec;
		com_of_mul += A.rows() * X.cols() * X.cols();

		system("cls");
		int prev = eigenvalues.size();
		int Xw = X.cols();
		int cnv = conv_select(eval, X, 0, Lam, X);
		cout << "LOBPCG-II已收敛特征向量个数：" << cnv << endl;

		t1 = clock();
		c1 = com_of_mul;
		T8 += t1 - t2;
		C8 += c1 - c2;

		if (cnv >= nev)
			break;

		time_t now = time(&now);
		if (timeCheck(start_time, now))
			break;

		int wid = batch < A.rows() - eigenvalues.size() ? batch : A.rows() - eigenvalues.size();
		if (X.cols() != wid) {
			new (&X) Map<MatrixXd, Unaligned, OuterStride<>>(storage, A.rows(), wid, OuterStride<>(3 * A.rows()));
			new (&W) Map<MatrixXd, Unaligned, OuterStride<>>(storage + A.rows(), A.rows(), wid, OuterStride<>(3 * A.rows()));
			new (&P) Map<MatrixXd, Unaligned, OuterStride<>>(storage + A.rows() * 2, A.rows(), wid, OuterStride<>(3 * A.rows()));
		}
		else if (P.cols() != X.cols()) {
			new (&P) Map<MatrixXd, Unaligned, OuterStride<>>(storage + A.rows() * 2, A.rows(), X.cols(), OuterStride<>(3 * A.rows()));
		}

		int xwid = wid < Xw - (cnv - prev) ? wid : Xw - (cnv - prev);
		X.rightCols(X.cols() - xwid) = MatrixXd::Random(A.rows(), X.cols() - xwid);

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

		P.leftCols(xwid) = BX.middleCols(cnv - prev, xwid);
		P.rightCols(P.cols() - xwid) = MatrixXd::Random(A.rows(), P.cols() - xwid);
		
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
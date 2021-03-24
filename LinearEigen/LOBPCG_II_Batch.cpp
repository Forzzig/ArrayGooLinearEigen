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
	VectorXd eval(batch);
	MatrixXd evec(batch, batch), BX(A.rows(), batch);
	vector<Map<MatrixXd>> V, WP, AWP;
	vector<Matrix3d> H;
	vector<SelfAdjointEigenSolver<MatrixXd>> eigsols;
	double* tmp = new double[A.rows() * 3 * batch];
	Map<MatrixXd, Unaligned, OuterStride<>> AX(tmp, A.rows(), batch, OuterStride<>(3 * A.rows()));
	for (int i = 0; i < batch; ++i) {
		V.push_back(Map<MatrixXd>(storage + A.rows() * 3 * i, A.rows(), 2));
		WP.push_back(Map<MatrixXd>(storage + A.rows() * (3 * i + 1), A.rows(), 1));
		AWP.push_back(Map<MatrixXd>(tmp + A.rows() * (3 * i + 1), A.rows(), 1));
		H.push_back(Matrix3d());
		eigsols.push_back(SelfAdjointEigenSolver<MatrixXd>());
	}
	while (true) {
		++nIter;
		cout << "迭代步：" << nIter << endl;

		if (AX.cols() != X.cols())
			new (&AX) Map<MatrixXd, Unaligned, OuterStride<>>(tmp, A.rows(), X.cols(), OuterStride<>(3 * A.rows()));
			//AX.resize(NoChange, X.cols());
		AX.noalias() = A * X;
		com_of_mul += A.nonZeros() * X.cols();

		if (BX.cols() != X.cols())
			BX.resize(NoChange, X.cols());
		BX.noalias() = B * X;
		com_of_mul += B.nonZeros() * X.cols();

#pragma omp parallel for
		for (int i = 0; i < X.cols(); ++i)
			BX.col(i) *= Lam(i, 0);
		com_of_mul += A.rows() * X.cols();

		//求解 A*W = A*X - mu*B*X， X为上一步的近似特征向量
		W = linearsolver.solve(AX - BX);
		com_of_mul += X.cols() * (A.nonZeros() + 4 * A.rows() +
			cgstep * (A.nonZeros() + 7 * A.rows()));

		//预存上一步的近似特征向量，BX废物利用
#pragma omp parallel for
		for (int i = 0; i < X.cols(); ++i)
			memcpy(&BX(0, i), &X(0, i), A.rows() * sizeof(double));
		//BX = X;

		//对每组XPW分别求解RR
#pragma omp parallel for
		for (int i = 0; i < X.cols(); ++i) {
			orthogonalization(WP[i], eigenvectors, B);
			orthogonalization(WP[i], X.col(i), B);
			int dep = orthogonalization(WP[i], B);
			
			if (dep) {
				int WPold = WP[i].cols();
				new (&WP[i]) Map<MatrixXd>(storage + A.rows() * (3 * i + 1), A.rows(), WPold - dep);
				new (&AWP[i]) Map<MatrixXd>(tmp + A.rows() * (3 * i + 1), A.rows(), WPold - dep);
				new (&V[i]) Map<MatrixXd>(storage + A.rows() * 3 * i, A.rows(), WPold - dep + 1);
			}
			
			AWP[i].noalias() = A * WP[i];
			com_of_mul += A.nonZeros() * WP[i].cols();
			
			H[i](0, 0) = X.col(i).dot(AX.col(i));
			com_of_mul += A.rows();

			H[i].block(0, 1, 1, WP[i].cols()).noalias() = X.col(i).transpose() * AWP[i];
			H[i].block(1, 0, WP[i].cols(), 1) = H[i].block(0, 1, 1, WP[i].cols()).transpose();
			com_of_mul += A.rows() * WP[i].cols();

			H[i].block(1, 1, WP[i].cols(), WP[i].cols()).noalias() = WP[i].transpose() * AWP[i];
			com_of_mul += WP[i].cols() * A.rows() * WP[i].cols();
		
			eigsols[i].compute(H[i].block(0, 0, 1 + WP[i].cols(), 1 + WP[i].cols()));
			com_of_mul += 24 * (1 + WP[i].cols()) * (1 + WP[i].cols()) * (1 + WP[i].cols());

			X.col(i) = V[i] * eigsols[i].eigenvectors().leftCols(1);
			com_of_mul += A.rows() * V[i].cols();

			if (dep || (nIter == 1)) {
				new (&WP[i]) Map<MatrixXd>(storage + A.rows() * (3 * i + 1), A.rows(), 2);
				new (&AWP[i]) Map<MatrixXd>(tmp + A.rows() * (3 * i + 1), A.rows(), 2);
				new (&V[i]) Map<MatrixXd>(storage + A.rows() * 3 * i, A.rows(), 3);
			}
		}

		//对X再做一次RR问题
		int dep = orthogonalization(X, B);
		if (dep) 
			new (&X) Map<MatrixXd, Unaligned, OuterStride<>>(storage, A.rows(), X.cols() - dep, OuterStride<>(3 * A.rows()));

		projection_RR(X, A, eval, evec);

		X *= evec;
		com_of_mul += A.rows() * X.cols() * X.cols();

		system("cls");
		int prev = eigenvalues.size();
		int Xw = X.cols();
		int cnv = conv_select(eval, X, 0, Lam, X);
		cout << "LOBPCG-II已收敛特征向量个数：" << cnv << endl;

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

//#pragma omp parallel for
		for (int i = 0; i < xwid; ++i)
			memcpy(&P(0, i), &BX(0, cnv - prev + i), A.rows() * sizeof(double));
		//P.leftCols(xwid) = BX.middleCols(cnv - prev, xwid);
		P.rightCols(P.cols() - xwid) = MatrixXd::Random(A.rows(), P.cols() - xwid);
	}
	finish();
}
#include<LOBPCG_II_Batch.h>

using namespace std;
LOBPCG_II_Batch::LOBPCG_II_Batch(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int batch)
	: LinearEigenSolver(A, B, nev),
	storage(new double[A.rows() * nev * 3]),
	X(storage, A.rows(), batch, OuterStride<>(3 * A.rows())),
	W(storage + A.rows(), A.rows(), batch, OuterStride<>(3 * A.rows())),
	P(storage + 2 * A.rows(), A.rows(), 0, OuterStride<>(3 * A.rows())),
	
	batch(batch),
	Lam(batch, 1),
	cgstep(cgstep){

	X = MatrixXd::Random(A.rows(), batch);
	orthogonalization(X, B);
	for (int i = 0; i < X.cols(); ++i)
		Lam(i, 0) = X.col(i).transpose() * A * X.col(i);
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

	MatrixXd eval, evec, tmp, AX, BX;
	Map<MatrixXd> V(storage, A.rows(), 2), WP(storage + A.rows(), A.rows(), 1);
	while (true) {
		++nIter;
		cout << "迭代步：" << nIter << endl;

		AX = A * X;
		com_of_mul += A.nonZeros() * X.cols();

		BX = B * X;
		com_of_mul += B.nonZeros() * X.cols();

		tmp = BX;
		for (int i = 0; i < X.cols(); ++i)
			tmp.col(i) *= Lam(i, 0);
		com_of_mul += A.rows() * X.cols();

		//求解 A*W = A*X - mu*B*X， X为上一步的近似特征向量
		W = linearsolver.solve(AX - tmp);
		com_of_mul += X.cols() * (A.nonZeros() + 4 * A.rows() +
			cgstep * (A.nonZeros() + 7 * A.rows()));

		//预存上一步的近似特征向量
		tmp = X;

		//对每组XPW分别求解RR
		for (int i = 0; i < X.cols(); ++i) {
			if (nIter == 1) {
				new (&WP) Map<MatrixXd>(storage + A.rows() * (3 * i + 1), A.rows(), 1);
				new (&V) Map<MatrixXd>(storage + A.rows() * 3 * i, A.rows(), 2);
			}
			else {
				new (&WP) Map<MatrixXd>(storage + A.rows() * (3 * i + 1), A.rows(), 2);
				new (&V) Map<MatrixXd>(storage + A.rows() * 3 * i, A.rows(), 3);
			}

			orthogonalization(WP, eigenvectors, B);
			orthogonalization(WP, X.col(i), B);
			int dep = orthogonalization(WP, B);
			
			int Vold = V.cols();
			new (&V) Map<MatrixXd>(storage + A.rows() * 3 * i, A.rows(), Vold - dep);

			projection_RR(V, A, eval, evec);

			X.col(i) = V * evec.leftCols(1);
			com_of_mul += A.rows() * V.cols();
		}

		//对X再做一次RR问题
		orthogonalization(X, B); //TODO 这里应该不会有线性相关

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
		new (&X) Map<MatrixXd, Unaligned, OuterStride<>>(storage, A.rows(), wid, OuterStride<>(3 * A.rows()));
		new (&W) Map<MatrixXd, Unaligned, OuterStride<>>(storage + A.rows(), A.rows(), wid, OuterStride<>(3 * A.rows()));		
		new (&P) Map<MatrixXd, Unaligned, OuterStride<>>(storage + A.rows() * 2, A.rows(), wid, OuterStride<>(3 * A.rows()));
		
		int xwid = wid < Xw - (cnv - prev) ? wid : Xw - (cnv - prev);
		X.rightCols(X.cols() - xwid) = MatrixXd::Random(A.rows(), X.cols() - xwid);

		orthogonalization(X, eigenvectors, B);
		orthogonalization(X, B); //TODO 应该不会有相关列

		P.leftCols(xwid) = tmp.middleCols(cnv - prev, xwid);
		P.rightCols(P.cols() - xwid) = MatrixXd::Random(A.rows(), P.cols() - xwid);
	}
	finish();
}
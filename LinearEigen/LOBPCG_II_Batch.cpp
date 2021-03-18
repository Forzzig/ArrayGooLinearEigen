#include<LOBPCG_II_Batch.h>

using namespace std;
LOBPCG_II_Batch::LOBPCG_II_Batch(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int batch)
	: LinearEigenSolver(A, B, nev),
	storage(new double[A.rows() * nev * 3]),
	X(storage + A.rows(), A.rows(), batch, OuterStride<>(3 * A.rows())),
	W(storage, A.rows(), batch, OuterStride<>(3 * A.rows())),
	P(storage + 2 * A.rows(), A.rows(), 0, OuterStride<>(3 * A.rows())),
	batch(batch),
	cgstep(cgstep){

	X = MatrixXd::Random(A.rows(), batch);
	orthogonalization(X, B);
	cout << "X��ʼ��" << endl;

	//ֻ��Ҫ���ľ��󲻱�ʱ���������ʼ��
	linearsolver.compute(A);

	//���ȹ̶�CG����������
	linearsolver.setMaxIterations(cgstep);
	cout << "CG�����׼�����..." << endl;
	cout << "��ʼ�����" << endl;
}


void LOBPCG_II_Batch::compute() {

	MatrixXd eval, evec, tmp, tmpA, mu;
	Map<MatrixXd> V(storage, A.rows(), 3), v1(storage, 0, 0);
	while (true) {
		++nIter;
		cout << "��������" << nIter << endl;

		tmpA = A * X;
		com_of_mul += A.nonZeros() * X.cols();

		tmp = B * X;
		com_of_mul += B.nonZeros() * X.cols();

		for (int i = 0; i < X.cols(); ++i) {
			mu = X.col(i).transpose() * tmpA.col(i);
			tmp.col(i) *= mu(0, 0);
		}
		com_of_mul += (X.cols() + 1) * A.rows() * X.cols();

		//��� A*W = A*X - mu*B*X�� XΪ��һ���Ľ�����������
		W = linearsolver.solve(tmpA - tmp);
		com_of_mul += X.cols() * (A.nonZeros() + 4 * A.rows() +
			cgstep * (A.nonZeros() + 7 * A.rows()));

		//Ԥ����һ���Ľ�����������
		tmp = X;

		//��ÿ��XPW�ֱ����RR
		for (int i = 0; i < X.cols(); ++i) {
			if (nIter == 1)
				new (&V) Map<MatrixXd>(storage + A.rows() * 3 * i, A.rows(), 2);
			else
				new (&V) Map<MatrixXd>(storage + A.rows() * 3 * i, A.rows(), 3);

			orthogonalization(V, eigenvectors, B);
			com_of_mul += V.cols() * eigenvectors.cols() * A.rows() * 2;

			int dep = orthogonalization(V, B);
			com_of_mul += (V.cols() + 1) * V.cols() * A.rows();

			int Vold = V.cols();
			new (&V) Map<MatrixXd>(storage + A.rows() * 3 * i, A.rows(), Vold - dep);

			projection_RR(V, A, eval, evec);
			com_of_mul += V.cols() * A.nonZeros() * V.cols() + (24 * V.cols() * V.cols() * V.cols());

			X.col(i) = V * evec.block(0, 0, V.cols(), 1);
			com_of_mul += A.rows() * V.cols();
		}

		//��X����һ��RR����
		orthogonalization(X, eigenvectors, B);
		com_of_mul += X.cols() * eigenvectors.cols() * A.rows() * 2;

		orthogonalization(X, B); //TODO ����Ӧ�ò������������
		com_of_mul += (X.cols() + 1) * X.cols() * A.rows();

		projection_RR(X, A, eval, evec);
		com_of_mul += X.cols() * A.nonZeros() * X.cols() + (24 * X.cols() * X.cols() * X.cols());

		tmpA = X * evec;
		com_of_mul += A.rows() * X.cols() * X.cols();

		system("cls");
		int prev = eigenvalues.size();
		int cnv = conv_select(eval, tmpA, 0, eval, tmpA);
		cout << "��������������������" << cnv << endl;
		com_of_mul += (A.nonZeros() + B.nonZeros() + 3 * A.rows()) * LinearEigenSolver::CHECKNUM;

		if (cnv >= nev)
			break;

		time_t now = time(&now);
		if (timeCheck(start_time, now))
			break;

		int wid = batch < A.rows() - eigenvalues.size() ? batch : A.rows() - eigenvalues.size();
		new (&W) Map<MatrixXd, Unaligned, OuterStride<>>(storage, A.rows(), wid, OuterStride<>(3 * A.rows()));
		new (&X) Map<MatrixXd, Unaligned, OuterStride<>>(storage + A.rows(), A.rows(), wid, OuterStride<>(3 * A.rows()));
		new (&P) Map<MatrixXd, Unaligned, OuterStride<>>(storage + A.rows() * 2, A.rows(), wid, OuterStride<>(3 * A.rows()));
		
		int xwid = wid < tmpA.cols() - (cnv - prev) ? wid : tmpA.cols() - (cnv - prev);
		X.leftCols(xwid) = tmpA.leftCols(xwid);
		X.rightCols(X.cols() - xwid) = MatrixXd::Random(A.rows(), X.cols() - xwid);

		orthogonalization(X, B);
		com_of_mul += (X.cols() + 1) * X.cols() * A.rows();

		P.leftCols(xwid) = tmp.middleCols(cnv - prev, xwid);
		P.rightCols(P.cols() - xwid) = MatrixXd::Random(A.rows(), P.cols() - xwid);
	}
}
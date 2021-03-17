#include<LOBPCG_I_Batch.h>

using namespace std;
LOBPCG_I_Batch::LOBPCG_I_Batch(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int batch)
	: LinearEigenSolver(A, B, nev),
	storage(new double[A.rows() * nev * 3]),
	X(storage + A.rows() * batch, A.rows(), batch),
	P(storage + A.rows() * batch * 2, A.rows(), 0),
	W(storage, A.rows(), batch),
	batch(batch) {

	X = MatrixXd::Random(A.rows(), batch);
	orthogonalization(X, B);
	cout << "X��ʼ��" << endl;

	//TODO ֻ��Ҫ���ľ��󲻱�ʱ���������ʼ��
	linearsolver.compute(A);

	//���ȹ̶�CG����������
	this->cgstep = cgstep;
	linearsolver.setMaxIterations(cgstep);
	cout << "CG�����׼�����..." << endl;
	cout << "��ʼ�����" << endl;
}


void LOBPCG_I_Batch::compute() {
	MatrixXd eval, evec, tmp, tmpA, mu, vl;

	//�����Ѽ�����������������ʹ�����Ķ����δ���storage������ڴ濽��
	Map<MatrixXd> V(storage, A.rows(), X.cols() + W.cols());
	while (true) {
		time_t now = time(&now);
		if (now - start_time > time_tol)
			break;

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
		/*cout << W.cols() << " " << X.cols() << " " << P.cols()<< endl;*/

		new (&V) Map<MatrixXd>(storage, A.rows(), W.cols() + X.cols() + P.cols());

		//ȫ����������������ֿ����׳�����
		orthogonalization(V, eigenvectors, B);
		com_of_mul += V.cols() * eigenvectors.cols() * A.rows() * 2;

		int dep = orthogonalization(V, B);
		com_of_mul += (V.cols() + 1) * V.cols() * A.rows();

		//��������������������޳�
		new (&V) Map<MatrixXd>(storage, A.rows(), W.cols() + X.cols() + P.cols() - dep);
		cout << V.cols() << endl;

		projection_RR(V, A, eval, evec);
		com_of_mul += V.cols() * A.nonZeros() * V.cols() + (24 * V.cols() * V.cols() * V.cols());

		system("cls");
		cout << V.rows() << " " << V.cols() << " " << evec.rows() << " " << evec.cols() << endl;

		//�ӿռ�VͶӰ�µ��µĽ�����������
		int nd = (2 * batch < V.cols()) ? 2 * batch : V.cols();
		tmpA = V * evec.block(0, 0, V.cols(), nd);
		com_of_mul += A.rows() * V.cols() * nd;

		int prev = eigenvalues.size();
		int cnv = conv_select(eval, tmpA, 0, eval, tmpA);
		com_of_mul += (A.nonZeros() + B.nonZeros() + 3 * A.rows()) * LinearEigenSolver::CHECKNUM;
		if (cnv >= nev)
			break;

		int wid = batch < A.rows() - eigenvalues.size() ? batch : A.rows() - eigenvalues.size();
		new (&W) Map<MatrixXd>(storage, A.rows(), wid);
		new (&X) Map<MatrixXd>(storage + A.rows() * wid, A.rows(), wid);
		new (&P) Map<MatrixXd>(storage + A.rows() * wid * 2, A.rows(), tmp.cols());
		
		X = tmpA.leftCols(wid);
		P = tmp;
	}
}
#include<LOBPCG_II.h>
#include<iostream>

using namespace std;
LOBPCG_II::LOBPCG_II(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep)
	: LinearEigenSolver(A, B, nev),
	storage(new double[A.rows() * nev * 3]),
	X(storage + A.rows(), A.rows(), nev, OuterStride<>(3 * A.rows())),
	W(storage, A.rows(), nev, OuterStride<>(3 * A.rows())),
	P(storage + 2 * A.rows(), A.rows(), 0, OuterStride<>(3 * A.rows())) {
	
	X = MatrixXd::Random(A.rows(), nev);
	orthogonalization(X, B);
	cout << "X��ʼ��" << endl;

	//ֻ��Ҫ���ľ��󲻱�ʱ���������ʼ��
	linearsolver.compute(A);

	//���ȹ̶�CG����������
	this->cgstep = cgstep;
	linearsolver.setMaxIterations(cgstep);
	cout << "CG�����׼�����..." << endl;
	cout << "��ʼ�����" << endl;
}


void LOBPCG_II::compute() {
	vector<int> cnv;
	MatrixXd eval, evec, tmp, tmpA, mu;
	Map<MatrixXd> V(storage, A.rows(), 3), v1(storage, 0, 0);
	while (true) {
		++nIter;
		cout << "��������" << nIter << endl;

		tmpA = A * X;
		com_of_mul += A.nonZeros() * X.cols();

		tmp = B * X;
		com_of_mul += B.nonZeros() * X.cols();

		for (int i = 0; i < nev; ++i) {
			mu = X.col(i).transpose() * tmpA.col(i);
			tmp.col(i) *= mu(0, 0);
		}
		com_of_mul += (X.cols() + 1) * A.rows() * X.cols();

		//��� A*W = A*X - mu*B*X�� XΪ��һ���Ľ�����������
		W = linearsolver.solve(tmpA - tmp);
		com_of_mul += X.cols() * (A.nonZeros() + 4 * A.rows() +
			cgstep * (A.nonZeros() + 7 * A.rows()));

		/*coutput << "W---------------------------------" << endl << W << endl;
		coutput << "WXP--------------------------------" << endl << Map<MatrixXd>(storage, A.rows(), nev * 3) << endl;*/

		//Ԥ����һ���Ľ�����������
		tmp = X;
		/*cout << W.cols() << " " << X.cols() << " " << P.cols()<< endl;*/

		//��ÿ��XPW�ֱ����RR
		for (int i = 0; i < nev; ++i) {
			if (nIter == 1)
				new (&V) Map<MatrixXd>(storage + A.rows() * 3 * i, A.rows(), 2);
			else
				new (&V) Map<MatrixXd>(storage + A.rows() * 3 * i, A.rows(), 3);
			
			/*coutput << "nIter: " << nIter << "---------------------------" << endl;
			coutput << "i: " << i << "-----------------------------" << endl;
			coutput << "V--------------------------------" << endl << V << endl;*/

			int dep = orthogonalization(V, B);
			com_of_mul += (V.cols() + 1) * V.cols() * A.rows();

			int Vold = V.cols();
			new (&V) Map<MatrixXd>(storage + A.rows() * 3 * i, A.rows(), Vold - dep);
			
			projection_RR(V, A, eval, evec);
			com_of_mul += V.cols() * A.nonZeros() * V.cols() + (24 * V.cols() * V.cols() * V.cols());

			X.col(i) = V * evec.block(0, 0, V.cols(), 1);
			com_of_mul += A.rows() * V.cols();
		}
		
		new (&P) Map<MatrixXd, Unaligned, OuterStride<>>(storage + 2 * A.rows(), A.rows(), 0, OuterStride<>(3 * A.rows()));
		P = tmp;

		////ȫ����������������ֿ����׳�����
		//int dep = orthogonalization(V, B);
		//com_of_mul += (V.cols() + 1) * V.cols() * A.rows();

		////��������������������޳�
		//new (&V) Map<MatrixXd>(storage, A.rows(), W.cols() + X.cols() + P.cols() - dep);
		//cout << V.cols() << endl;

		//projection_RR(V, A, eval, evec);
		//com_of_mul += V.cols() * A.nonZeros() * V.cols() + (24 * V.cols() * V.cols() * V.cols());

		//system("cls");
		//cout << V.rows() << " " << V.cols() << " " << evec.rows() << " " << evec.cols() << endl;

		////�ӿռ�VͶӰ�µ��µĽ�����������
		//X = V * evec.block(0, 0, V.cols(), nev);
		//com_of_mul += A.rows() * V.cols() * nev;

		////P����һ���Ľ�����������
		//new (&P) Map<MatrixXd>(storage + A.rows() * (W.cols() + X.cols()), A.rows(), nev);
		//P = tmp;

		//��X����һ��RR����
		orthogonalization(X, B); //TODO ����Ӧ�ò������������
		com_of_mul += (X.cols() + 1) * X.cols() * A.rows();

		projection_RR(X, A, eval, evec);
		com_of_mul += X.cols() * A.nonZeros() * X.cols() + (24 * X.cols() * X.cols() * X.cols());
		
		X *= evec;
		com_of_mul += A.rows() * X.cols() * X.cols();

		system("cls");
		cnv = conv_check(eval, X, 0);
		cout << "��������������������" << cnv.size() << endl;
		com_of_mul += (A.nonZeros() + B.nonZeros() + 3 * A.rows()) * LinearEigenSolver::CHECKNUM;

		if (cnv.size() >= nev) {
			eigenvectors = X;
			for (int i = 0; i < nev; ++i)
				eigenvalues.push_back(eval(i, 0));
			break;
		}
	}
}
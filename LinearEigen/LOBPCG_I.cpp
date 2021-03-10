#include<LOBPCG_I.h>
#include<iostream>

using namespace std;
LOBPCG_I::LOBPCG_I(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep)
	: LinearEigenSolver(A, B, nev),
		storage(new double[A.rows() * nev * 3]),
		X(storage + A.rows() * nev, A.rows(), nev),
		P(storage + A.rows() * nev * 2, A.rows(), 0),
		W(storage, A.rows(), nev) {
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


void LOBPCG_I::compute() {
	vector<int> cnv;
	MatrixXd eval, evec, tmp, tmpA, mu;

	//�����Ѽ�����������������ʹ�����Ķ����δ���storage������ڴ濽��
	Map<MatrixXd> V(storage, A.rows(), X.cols() + W.cols()), vl(storage, 0, 0);

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

		//Ԥ����һ���Ľ�����������
		tmp = X;
		/*cout << W.cols() << " " << X.cols() << " " << P.cols()<< endl;*/

		//��Ϊ��ȡ��������������������V��storageͷ����ʼ
		new (&V) Map<MatrixXd>(storage, A.rows(), W.cols() + X.cols() + P.cols());
		
		//ȫ����������������ֿ����׳�����
		int dep = orthogonalization(V, B);
		com_of_mul += (V.cols() + 1) * V.cols() * A.rows();

		//��������������������޳�
		new (&V) Map<MatrixXd>(storage, A.rows(), W.cols() + X.cols() + P.cols() - dep);
		cout << V.cols() << endl;
		
		projection_RR(V, A, eval, evec);
		com_of_mul += V.cols() * A.nonZeros() * V.cols() + (V.cols() * V.cols() * V.cols());

		system("cls");
		cout << V.rows() << " " << V.cols() << " " << evec.rows() << " " << evec.cols() << endl;
		
		//�ӿռ�VͶӰ�µ��µĽ�����������
		X = V * evec.block(0, 0, V.cols(), nev);
		com_of_mul += A.rows() * V.cols() * V.cols();
		
		//P����һ���Ľ�����������
		new (&P) Map<MatrixXd>(storage + A.rows() * (W.cols() + X.cols()), A.rows(), nev);
		P = tmp;

		//����v1��Ϊ�˷�����Ƽ�������ĸ���
		new (&vl) Map<MatrixXd>(&(eval(0, 0)), nev, 1);
		
		cnv = conv_check(vl, X, 0);
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
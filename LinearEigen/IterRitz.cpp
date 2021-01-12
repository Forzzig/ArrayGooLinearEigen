#include<IterRitz.h>

IterRitz::IterRitz(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int q, int r) : LinearEigenSolver(A, B, nev){
	this->q = q;
	this->r = r;
	X = MatrixXd::Random(A.rows(), q);
	orthogonalization(X, B);
	MatrixXd evec;
	projection_RR(X, A, LAM, evec);
	X = X * evec;
	linearsolver.compute(A);
	linearsolver.setMaxIterations(cgstep);
	cout << "CG�����׼�����..." << endl;
	nIter = 0;
	cout << "��ʼ�����" << endl;
}

void IterRitz::compute() {
	double shift = 0;
	int cnv = 0;
	MatrixXd eval, evec, Xnew, tmp, rls;
	SparseMatrix<double> tmpA;
	P.resize(A.rows(), 0);
	while (true) {
		++nIter;
		cout << "��������" << nIter << endl;
		cout << "��Ƶ��" << shift << endl;
		X1.resize(A.rows(), q * r);
		for (int i = 0; i < r; ++i) {
			if (i == 0) {
				tmp = B * X;
				for (int j = 0; j < q; ++j) {
					tmp.col(j) *= LAM(j, 0);
				}
				X1.block(0, 0, A.rows(), q) = linearsolver.solveWithGuess(tmp, X);
				
				tmp = X1.block(0, 0, A.rows(), q);
				rls = tmp.transpose() * A * tmp;
				for (int j = 0; j < q; ++j)
					LAM(j, 0) = rls(j, j);
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
				X1.block(0, i * q, A.rows(), q) = linearsolver.solve(rls); 
				tmp -= X1.block(0, i * q, A.rows(), q);
			}
			
			orthogonalization(X1, i * q, (i + 1) * q, 0, i * q, B);
			orthogonalization(X1, i * q, (i + 1) * q, B);
		}
		V.resize(A.rows(), P.cols() + X1.cols());
		V << X1, P;
		orthogonalization(V, eigenvectors, B);
		orthogonalization(V, B);
		cout << V.cols() << endl;
		projection_RR(V, A, eval, evec);
		cout << V.rows() << " " << V.cols() << " " << evec.rows() << " " << evec.cols() << endl;
		/*orthogonalization(X1, eigenvectors, B);
		orthogonalization(X1, B);
		projection_RR(X1, A, eval, evec);

		Xnew = X1 * evec;*/
		Xnew = V * evec;
		//cout << Xnew << endl;
		system("cls");
		P = X;
		cnv = conv_check(eval, Xnew, shift, LAM, X);
		//system("pause");
		cout << "��������������������" << cnv << endl;
		/*if (nIter == 8) {
			cout << Xnew << endl << "-----------------" << endl;
			cout << X << endl << "--------------------" << endl;
			cout << LAM << endl;
			system("pause");
		}*/
		/*if (cnv > 0)
			system("pause");*/
		if (cnv >= nev) {
			break;
		}
		/*if (eigenvalues.size())
			shift = ((eval.col(0)[nev - X0.cols() + cnv - 1] - shift) - 100 * (eigenvalues[0])) / 99;
		else
			shift = ((eval.col(0)[nev - 1] - shift) - 100 * (eval.col(0)[0] - shift)) / 99;*/

	}
}
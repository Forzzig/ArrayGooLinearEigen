#include<BJD.h>

BJD::BJD(SparseMatrix<double, RowMajor, __int64>& A, SparseMatrix<double, RowMajor, __int64>& B, int nev, int restart, int batch, int gmres_size, int gmres_restart) 
	: LinearEigenSolver(A, B, nev),
	restart(restart),
	gmres_size(gmres_size),
	gmres_restart(gmres_restart), 
	nRestart(0),
	batch(batch),
	V(A.rows(), batch * restart),
	/*W(A.rows(), batch * restart),*/
	HA(batch * restart, batch * restart),
	/*HB(batch * restart, batch * restart),*/
	Atmp(A.rows(), batch),
	Btmp(A.rows(), batch),
	AV(A.rows(), batch * restart)/*,
	BV(A.rows(), batch * restart),
	tau(0),*/
	/*tmpAinv(A.rows())*/{

	Map<MatrixXd> V1(&V(0, 0), A.rows(), batch);
	V1 = MatrixXd::Random(A.rows(), batch);
	int dep = orthogonalization(V1, B);
	while (dep) {
		V1.rightCols(dep) = MatrixXd::Random(A.rows(), dep);
		dep = orthogonalization(V1, B);
	}
	
	Map<MatrixXd> AV1(&AV(0, 0), A.rows(), batch);
	/*Map<MatrixXd> BV1(&BV(0, 0), A.rows(), batch);
	Map<MatrixXd> W1(&W(0, 0), A.rows(), batch);*/
	AV1.noalias() = A * V1;
	/*BV1.noalias() = B * V1;
	W1 = AV1 - BV1 * tau;*/

	//HA.block(0, 0, V1.cols(), V1.cols()) = W1.transpose() * AV1;
	HA.block(0, 0, V1.cols(), V1.cols()) = V1.transpose() * AV1;
	/*HB.block(0, 0, V1.cols(), V1.cols()) = W1.transpose() * BV1;*/

	for (int i = 0; i < batch; ++i) {
		Y.push_back(MatrixXd(A.rows(), batch));
		v.push_back(MatrixXd(batch, batch));
	}

#ifdef DIAG_PRECOND
	for (int i = 0; i < batch; ++i)
		Ainv.push_back(vector<double>(A.rows()));
#endif // DIAG_PRECOND
}

BJD::~BJD() {
	//ILU需要
    //delete[] ia;
    //delete[] ja;
    //delete[] a;
}

void BJD::genCRS(SparseMatrix<double, RowMajor, __int64>& A, int* ia, int* ja, double* a) {
    
    int* tmpia = new int[A.nonZeros()];
    memset(ia, 0, sizeof(int) * (A.rows() + 1));

    int i = 0;
    for (int k = 0; k < A.cols(); ++k)
        for (SparseMatrix<double, RowMajor, __int64>::InnerIterator it(A, k); it; ++it) {
            tmpia[i] = it.row();
            ja[i] = it.col();
            a[i] = it.value();
            ++i;
        }
    CRSsort(tmpia, ja, a, A.nonZeros());

    for (int i = 0; i < A.nonZeros(); ++i) {
        ia[tmpia[i]] = i;
        while (i < A.nonZeros() - 1) {
            if (tmpia[i] != tmpia[i + 1])
                break;
            ++i;
        }
    }
    ia[A.rows()] = A.nonZeros();
    delete[] tmpia;
}

void BJD::CRSsort(int* ia, int* ja, double* a, int n) {
    if (n <= 1)
        return;
    int l = 0, r = n - 1, m = r / 2;
    int row = ia[m];
    int col = ja[m];
    while (l < r) {
        while ((ia[l] < row) || (ia[l] == row) && (ja[l] < col)) ++l;
        while ((ia[r] > row) || (ia[r] == row) && (ja[r] > col)) --r;
        if (l <= r) {
            swap(ia[l], ia[r]);
            swap(ja[l], ja[r]);
            swap(a[l], a[r]);
            ++l;
            --r;
        }
    }
    CRSsort(ia, ja, a, r + 1);
    CRSsort(ia + l, ja + l, a + l, n - l);
}

//带自动扩容
void BJD::CRSsubtrac(int*& ia, int*& ja, double*& a, int& nnz, SparseMatrix<double, RowMajor, __int64>& B, double eff) {
    //记录增加的数
    int* row_add = new int[B.nonZeros()];
    int* col_add = new int[B.nonZeros()];
    double* value_add = new double[B.nonZeros()];
    int num = 0;
    for (int k = 0; k < B.cols(); ++k) {
        for (SparseMatrix<double, RowMajor, __int64>::InnerIterator it(B, k); it; ++it) {
            int l = ia[it.row()], r = ia[it.row() + 1] - 1, mid;
            while (l <= r) {
                mid = (l + r) / 2;
                if (ja[mid] == it.col())
                    break;
                else if (ja[mid] > it.col())
                    r = mid - 1;
                else
                    l = mid + 1;
            }
            if (l <= r) {
                //找到了！
                a[mid] -= eff * it.value();
            }
            else {
                //没有找到，需要扩容
                row_add[num] = it.row();
                col_add[num] = it.col();
                value_add[num] = -eff * it.value();
                ++num;
            }
        }
    }

    if (num > 0) {
        //使其相邻元素聚集起来
        CRSsort(row_add, col_add, value_add, num);

        int* newia = new int[B.rows() + 1];
        int* newja = new int[nnz + num];
        double* newa = new double[nnz + num];
        int i = 0; //原有元素指示器
        int lasti = 0; //等原矩阵元素凑够一大段一起copy
        int j = 0; //新增元素指示器
        int k = 0; //新数组指示器
        for (int row = 0; row < B.rows(); ++row) {
            newia[row] = k;
            for ( ; i < ia[row + 1]; ++i) {
                if ((j < num) && (row_add[j] == row) && (col_add[j] < ja[i])) {
                    memcpy(newja + k, ja + lasti, sizeof(int) * (i - lasti));
                    memcpy(newa + k, a + lasti, sizeof(double) * (i - lasti));
                    k += i - lasti;
                    lasti = i;
                    newja[k] = col_add[j];
                    newa[k] = value_add[j];
                    ++j;
                    ++k;
                }
            }
        }
        memcpy(newja + k, ja + lasti, sizeof(int) * (i - lasti));
        memcpy(newa + k, a + lasti, sizeof(double) * (i - lasti));
        k += i - lasti;
        lasti = i;
        newia[B.rows() + 1] = k;

        delete[] ia;
        delete[] ja;
        delete[] a;
        ia = newia;
        ja = newja;
        a = newa;
        nnz += num;
    }

    delete[] row_add;
    delete[] col_add;
    delete[] value_add;

    com_of_mul += A.nonZeros() + B.nonZeros();
}

void BJD::compute() {
	clock_t t1, t2, T1 = 0, T2 = 0, T3 = 0, T4 = 0, T5 = 0, T6 = 0, T7 = 0, T8 = 0, T9 = 0, T10 = 0;
	long long c1, c2, C1 = 0, C2 = 0, C3 = 0, C4 = 0, C5 = 0, C6 = 0, C7 = 0, C8 = 0, C9 = 0, C10 = 0;

	VectorXd eval(restart * batch);
	MatrixXd evec(restart * batch, restart * batch), ui(A.rows(), batch), ri(A.rows(), batch), Au(A.rows(), batch);
	Map<MatrixXd> Vj(&V(0, 0), A.rows(), batch);
	/*Map<MatrixXd> Wj(&W(0, 0), A.rows(), batch);*/
	Map<MatrixXd> AVj(&AV(0, 0), A.rows(), batch);
	/*Map<MatrixXd> BVj(&BV(0, 0), A.rows(), batch);*/

	Map<MatrixXd> X(&V(0, Vj.cols()), A.rows(), ri.cols());
	/*Map<MatrixXd> WX(&W(0, Vj.cols()), A.rows(), ri.cols());*/
	Map<MatrixXd> AX(&AV(0, Vj.cols()), A.rows(), ri.cols());
	/*Map<MatrixXd> BX(&BV(0, Vj.cols()), A.rows(), ri.cols());*/
	int nd = batch;
	while (true) {
		++nRestart;
		int prev = eigenvalues.size();
		for (int i = 1; i <= restart; ++i) {
			t1 = clock();
			c1 = com_of_mul;

			++nIter;
			cout << "第" << nRestart << "轮重启：" << endl;
			cout << "迭代步：" << i << endl;
			cout << "总迭代步：" << nIter << endl;

			//generalized_RR(HA.block(0, 0, Vj.cols(), Vj.cols()), HB.block(0, 0, Vj.cols(), Vj.cols()), eval, evec);
			RR(HA.block(0, 0, Vj.cols(), Vj.cols()), eval, evec);
			
			t2 = clock();
			c2 = com_of_mul;
			T1 += t2 - t1;
			C1 += c2 - c1;

			if (ui.cols() != nd)
				ui.resize(NoChange, nd);
#pragma omp parallel for
			for (int j = 0; j < ui.cols(); ++j)
				ui.col(j).noalias() = Vj * evec.col(j);
			com_of_mul += A.rows() * Vj.cols() * nd;

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

			int cnv = conv_select(eval, ui, 0, eval, ui);
			cout << "BJD已收敛特征向量个数：" << cnv << endl;

			if (cnv > prev)
				break;

			if (i == restart)
				break;

			time_t now = time(&now);
			if (timeCheck(start_time, now))
				break;

			t2 = clock();
			c2 = com_of_mul;
			T5 += t2 - t1;
			C5 += c2 - c1;

			if (ri.cols() != ui.cols())
				ri.resize(NoChange, ui.cols());
			new (&X) Map<MatrixXd>(&V(0, Vj.cols()), A.rows(), ri.cols());
#pragma omp parallel for
			for (int j = 0; j < ui.cols(); ++j) {
				ri.col(j).noalias() = B * ui.col(j);
				ri.col(j) *= eval(j, 0);

				Au.col(j).noalias() = A * ui.col(j);
				ri.col(j) -= Au.col(j);
				
				memset(&X(0, j), 0, A.rows() * sizeof(double));
				PMGMRES(A, B, ri.col(j), ui, X.col(j), eval(j, 0), j);
			}
			com_of_mul += ui.cols() * (B.nonZeros() + A.rows());
			com_of_mul += A.nonZeros() * ui.cols();

			t1 = clock();
			c1 = com_of_mul;
			T6 += t1 - t2;
			C6 += c1 - c2;

			orthogonalization(X, eigenvectors, B);
			orthogonalization(X, Vj, B);
			int dep = orthogonalization(X, B);
			new (&X) Map<MatrixXd>(&V(0, Vj.cols()), A.rows(), ri.cols() - dep);
			/*new (&WX) Map<MatrixXd>(&W(0, Vj.cols()), A.rows(), X.cols());*/
			new (&AX) Map<MatrixXd>(&AV(0, Vj.cols()), A.rows(), X.cols());
			/*new (&BX) Map<MatrixXd>(&BV(0, Vj.cols()), A.rows(), X.cols());*/

			t2 = clock();
			c2 = com_of_mul;
			T7 += t2 - t1;
			C7 += c2 - c1;
			
#pragma omp parallel for
			for (int j = 0; j < X.cols(); ++j) {
				AX.col(j).noalias() = A * X.col(j);
				/*BX.col(j).noalias() = B * X.col(j);
				
				WX.col(j).noalias() = AX.col(j) - BX.col(j) * tau;*/
				//HA.block(0, Vj.cols() + j, Vj.cols(), 1).noalias() = Wj.transpose() * AX.col(j);
				HA.block(0, Vj.cols() + j, Vj.cols(), 1).noalias() = V.transpose() * AX.col(j);
				//HA.block(Vj.cols() + j, 0, 1, Vj.cols()) = HA.block(0, Vj.cols() + j, Vj.cols(), 1).transpose();
				HA.block(Vj.cols() + j, 0, 1, Vj.cols()) = HA.block(0, Vj.cols() + j, Vj.cols(), 1).transpose();
				/*HB.block(0, Vj.cols() + j, Vj.cols(), 1).noalias() = Wj.transpose() * BX.col(j);
				HB.block(Vj.cols() + j, 0, 1, Vj.cols()) = HB.block(0, Vj.cols() + j, Vj.cols(), 1).transpose();*/
			}
			com_of_mul += X.cols() * (A.nonZeros() + Vj.cols() * X.rows());
			
			/*HA.block(Vj.cols(), Vj.cols(), X.cols(), X.cols()).noalias() = WX.transpose() * AX;
			HB.block(Vj.cols(), Vj.cols(), X.cols(), X.cols()).noalias() = WX.transpose() * BX;*/
			HA.block(Vj.cols(), Vj.cols(), X.cols(), X.cols()).noalias() = X.transpose() * AX;
			com_of_mul += X.cols() * A.rows() * X.cols();
			
			t1 = clock();
			c1 = com_of_mul;
			T8 += t1 - t2;
			C8 += c1 - c2;

			t2 = clock();
			c2 = com_of_mul;
			T9 += t2 - t1;
			C9 += c2 - c1;

			new (&Vj) Map<MatrixXd>(&V(0, 0), A.rows(), Vj.cols() + X.cols());
			/*new (&Wj) Map<MatrixXd>(&W(0, 0), A.rows(), Vj.cols());*/
			new (&AVj) Map<MatrixXd>(&AV(0, 0), A.rows(), Vj.cols());
			/*new (&BVj) Map<MatrixXd>(&BV(0, 0), A.rows(), Vj.cols());*/

			system("cls");

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
		if (eigenvalues.size() >= nev)
			break;
		
		time_t now = time(&now);
		if (timeCheck(start_time, now))
			break;

		int left = nd - (eigenvalues.size() - prev);
		nd = (batch < A.rows() - eigenvalues.size()) ? batch : A.rows() - eigenvalues.size();
		new (&Vj) Map<MatrixXd>(&V(0, 0), A.rows(), nd);
		Vj.leftCols(left) = ui.leftCols(left);
		Vj.rightCols(nd - left) = MatrixXd::Random(A.rows(), nd - left);
		orthogonalization(Vj, eigenvectors, B);
		int dep = orthogonalization(Vj, B);
		while (dep) {
			Vj.rightCols(dep) = MatrixXd::Random(A.rows(), dep);
			dep = orthogonalization(Vj, B);
		}
		
		/*new (&Wj) Map<MatrixXd>(&W(0, 0), A.rows(), Vj.cols());*/
		new (&AVj) Map<MatrixXd>(&AV(0, 0), A.rows(), Vj.cols());
		/*new (&BVj) Map<MatrixXd>(&BV(0, 0), A.rows(), Vj.cols());*/

#pragma omp parallel for
		for (int j = 0; j < Vj.cols(); ++j) {
			AVj.col(j).noalias() = A * Vj.col(j);
			/*BVj.col(j).noalias() = B * Vj.col(j);

			Wj.col(j) = AVj.col(j) - BVj.col(j) * tau;*/
		}
		com_of_mul += A.nonZeros() * Vj.cols();

		/*HA.block(0, 0, Vj.cols(), Vj.cols()).noalias() = Wj.transpose() * AVj;
		HB.block(0, 0, Vj.cols(), Vj.cols()).noalias() = Wj.transpose() * BVj;*/
		HA.block(0, 0, Vj.cols(), Vj.cols()).noalias() = Vj.transpose() * AVj;
		com_of_mul += Vj.cols() * A.rows() * Vj.cols();
	}
	finish();
}

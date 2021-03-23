#include<BJD.h>

BJD::BJD(SparseMatrix<double, RowMajor>& A, SparseMatrix<double, RowMajor>& B, int nev, int restart, int batch, int gmres_size, int gmres_restart) 
	: LinearEigenSolver(A, B, nev),
	restart(restart),
	gmres_size(gmres_size),
	gmres_restart(gmres_restart), 
	nRestart(0),
	batch(batch),
	V(A.rows(), batch* restart),
	WA(A.rows(), batch* restart),
	H(batch* restart, batch* restart)/*,
	tmpAinv(A.rows()) */{

	Map<MatrixXd> V1(&V(0, 0), A.rows(), batch);
	V1 = MatrixXd::Random(A.rows(), batch);
	orthogonalization(V1, B);
	
	Map<MatrixXd> W1(&WA(0, 0), A.rows(), batch);
	W1 = A * V1;

	/*WB.resize(A.rows(), batch * restart);
	Map<MatrixXd> W2(&WB(0, 0), A.rows(), batch);
	W2 = B * V1;
	HA.resize(batch * restart, batch * restart);
	HA.block(0, 0, batch, batch) = W1.transpose() * W1;
	HB.resize(batch * restart, batch * restart);
	HB.block(0, 0, batch, batch) = W2.transpose() * W2;
	HAB.resize(batch * restart, batch * restart);
	HAB.block(0, 0, batch, batch) = W1.transpose() * W2;*/
	H.block(0, 0, batch, batch) = W1.transpose() * V1;
    
	//ILU需要
    //ia = new int[A.rows() + 1];
    //ja = new int[A.nonZeros()];
    //a = new double[A.nonZeros()];

    //nnz = A.nonZeros();
    //genCRS(A, ia, ja, a);
}

BJD::~BJD() {
	//ILU需要
    //delete[] ia;
    //delete[] ja;
    //delete[] a;
}

void BJD::genCRS(SparseMatrix<double, RowMajor>& A, int* ia, int* ja, double* a) {
    
    int* tmpia = new int[A.nonZeros()];
    memset(ia, 0, sizeof(int) * (A.rows() + 1));

    int i = 0;
    for (int k = 0; k < A.cols(); ++k)
        for (SparseMatrix<double, RowMajor>::InnerIterator it(A, k); it; ++it) {
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
void BJD::CRSsubtrac(int*& ia, int*& ja, double*& a, int& nnz, SparseMatrix<double, RowMajor>& B, double eff) {
    //记录增加的数
    int* row_add = new int[B.nonZeros()];
    int* col_add = new int[B.nonZeros()];
    double* value_add = new double[B.nonZeros()];
    int num = 0;
    for (int k = 0; k < B.cols(); ++k) {
        for (SparseMatrix<double, RowMajor>::InnerIterator it(B, k); it; ++it) {
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
	VectorXd eval(restart * batch);
	MatrixXd evec(restart * batch, restart * batch), ui(A.rows(), batch), ri(A.rows(), batch);
	Map<MatrixXd> Vj(&V(0, 0), A.rows(), batch);
	Map<MatrixXd> WAj(&WA(0, 0), A.rows(), batch);
	/*Map<MatrixXd> WBj(&WB(0, 0), A.rows(), batch);*/

	Map<MatrixXd> X(&V(0, Vj.cols()), A.rows(), ri.cols());
	Map<MatrixXd> tmpWA(&WA(0, Vj.cols()), A.rows(), X.cols());
	MatrixXd tmpeval(batch, 1), tmpevec(A.rows(), batch);
	int nd = batch;
	/*long long t1, t2;
	long long tRR = 0, tCnv = 0, tGMR = 0, tOrt = 0, tAX = 0, tH = 0;*/
	while (true) {
		++nRestart;
		int prev = eigenvalues.size();
		for (int i = 1; i <= restart; ++i) {
			
			++nIter;
			system("cls");
			cout << "第" << nRestart << "轮重启：" << endl;
			cout << "迭代步：" << i << endl;
			//t1 = clock();

			//generalized_RR(HA.block(0, 0, Vj.cols(), Vj.cols()), HB.block(0, 0, Vj.cols(), Vj.cols()), HAB.block(0, 0, Vj.cols(), Vj.cols()), 0, eval, evec);
			RR(H.block(0, 0, Vj.cols(), Vj.cols()), eval, evec);
			
			//t2 = clock();
			//tRR += t2 - t1;

			//注意eval长度是偏长的
			if (ui.cols() != nd)
				ui.resize(NoChange, nd);
			ui.noalias() = Vj * evec.leftCols(nd);
			com_of_mul += A.rows() * Vj.cols() * nd;

			if (ri.cols() != ui.cols())
				ri.resize(NoChange, ui.cols());
			ri.noalias() = B * ui;
#pragma omp parallel for
			for (int j = 0; j < ri.cols(); ++j) {
				ri.col(j) *= eval(j, 0);
			}
			com_of_mul += ui.cols() * (B.nonZeros() + A.rows());

			ri -= A * ui;
			com_of_mul += A.nonZeros() * ui.cols();

			int cnv = conv_select(eval, ui, 0, tmpeval, tmpevec);
			cout << "BJD已收敛特征向量个数：" << cnv << endl;

			//t1 = clock();
			//tCnv += t1 - t2;

			if (cnv > prev)
				break;

			if (i == restart)
				break;

			time_t now = time(&now);
			if (timeCheck(start_time, now))
				break;

			new (&X) Map<MatrixXd>(&V(0, Vj.cols()), A.rows(), ri.cols());
			X = MatrixXd::Zero(A.rows(), ri.cols());

			//L_GMRES(A, B, ri, ui, X, eval, gmres_size);
            PMGMRES(ri, ui, X, eval);

			//t2 = clock();
			//tGMR += t2 - t1;

			orthogonalization(X, eigenvectors, B);
			orthogonalization(X, Vj, B);
			
			int dep = orthogonalization(X, B);

			//t1 = clock();
			//tOrt += t1 - t2;

			new (&X) Map<MatrixXd>(&V(0, Vj.cols()), A.rows(), ri.cols() - dep);
			new (&tmpWA) Map<MatrixXd>(&WA(0, Vj.cols()), A.rows(), X.cols());
			tmpWA.noalias() = A * X;
			com_of_mul += A.nonZeros() * X.cols();

			//t2 = clock();
			//tAX += t2 - t1;

			/*Map<MatrixXd> tmpWB(&WB(0, Vj.cols()), A.rows(), X.cols());
			tmpWB = B * X;*/

			/*HA.block(Vj.cols(), 0, X.cols(), Vj.cols()) = tmpWA.transpose() * WAj;
			HA.block(0, Vj.cols(), Vj.cols(), X.cols()) = HA.block(Vj.cols(), 0, X.cols(), Vj.cols()).transpose();
			HA.block(Vj.cols(), Vj.cols(), X.cols(), X.cols()) = tmpWA.transpose() * tmpWA;
			HB.block(Vj.cols(), 0, X.cols(), Vj.cols()) = tmpWB.transpose() * WBj;
			HB.block(0, Vj.cols(), Vj.cols(), X.cols()) = HB.block(Vj.cols(), 0, X.cols(), Vj.cols()).transpose();
			HB.block(Vj.cols(), Vj.cols(), X.cols(), X.cols()) = tmpWB.transpose() * tmpWB;
			HAB.block(Vj.cols(), 0, X.cols(), Vj.cols()) = tmpWA.transpose() * WBj;
			HAB.block(0, Vj.cols(), Vj.cols(), X.cols()) = HAB.block(Vj.cols(), 0, X.cols(), Vj.cols()).transpose();
			HAB.block(Vj.cols(), Vj.cols(), X.cols(), X.cols()) = tmpWA.transpose() * tmpWB;*/
			H.block(Vj.cols(), 0, X.cols(), Vj.cols()).noalias() = tmpWA.transpose() * Vj;
			H.block(0, Vj.cols(), Vj.cols(), X.cols()) = H.block(Vj.cols(), 0, X.cols(), Vj.cols()).transpose();
			H.block(Vj.cols(), Vj.cols(), X.cols(), X.cols()).noalias() = tmpWA.transpose() * X;
			com_of_mul += X.cols() * A.rows() * Vj.cols() + X.cols() * A.rows() * X.cols();

			//t1 = clock();
			//tH += t1 - t2;

			new (&Vj) Map<MatrixXd>(&V(0, 0), A.rows(), Vj.cols() + X.cols());
			new (&WAj) Map<MatrixXd>(&WA(0, 0), A.rows(), WAj.cols() + X.cols());
			/*new (&WBj) Map<MatrixXd>(&WB(0, 0), A.rows(), WBj.cols() + X.cols());*/
		}
		if (eigenvalues.size() >= nev)
			break;
		
		time_t now = time(&now);
		if (timeCheck(start_time, now))
			break;

		int left = nd - (eigenvalues.size() - prev);
		nd = (batch < A.rows() - eigenvalues.size()) ? batch : A.rows() - eigenvalues.size();
		new (&Vj) Map<MatrixXd>(&V(0, 0), A.rows(), nd);
		Vj.leftCols(left) = tmpevec.leftCols(left);
		Vj.rightCols(nd - left) = MatrixXd::Random(A.rows(), nd - left);
		orthogonalization(Vj, eigenvectors, B);
		orthogonalization(Vj, B);
		
		new (&WAj) Map<MatrixXd>(&WA(0, 0), A.rows(), Vj.cols());
		/*new (&WBj) Map<MatrixXd>(&WB(0, 0), A.rows(), Vj.cols());*/

		WAj.noalias() = A * Vj;
		com_of_mul += A.nonZeros() * Vj.cols();

		/*WBj = B * Vj;
		HA.block(0, 0, Vj.cols(), Vj.cols()) = WAj.transpose() * WAj;
		HB.block(0, 0, Vj.cols(), Vj.cols()) = WBj.transpose() * WBj;
		HAB.block(0, 0, Vj.cols(), Vj.cols()) = WAj.transpose() * WBj;*/
		H.block(0, 0, Vj.cols(), Vj.cols()).noalias() = WAj.transpose() * Vj;
		com_of_mul += Vj.cols() * A.rows() * Vj.cols();
	}
	/*cout << "RR   , Cnv   , GMR   , Ort   , AX   , H" << endl;
	cout << tRR << ", " << tCnv << ", " << tGMR << ", " << tOrt << ", " << tAX << ", " << tH << endl;
	system("pause");*/
	finish();
}

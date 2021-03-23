#ifndef __BJD_H__
#define __BJD_H__
#include<LinearEigenSolver.h>
#include<mgmres.hpp>

using namespace Eigen;
class BJD : public LinearEigenSolver {
public:
	int restart, gmres_size, gmres_restart, nRestart;
	MatrixXd V, WA,/* WB, HA, HB, HAB,*/ H;
	MatrixXd Y, v;
	SparseMatrix<double, RowMajor> tmpA;
    //vector<double> tmpAinv;
	int batch;

    //ILU需要
	//int* ia, * ja;
	//double* a;
	//int nnz;
	void CRSsort(int* ia, int* ja, double* a, int n);
	void genCRS(SparseMatrix<double, RowMajor>& A, int* ia, int* ja, double* a);

	void CRSsubtrac(int*& ia, int*& ja, double*& a, int& nnz, SparseMatrix<double, RowMajor>& B, double eff);

	BJD(SparseMatrix<double, RowMajor>& A, SparseMatrix<double, RowMajor>& B, int nev, int restart, int batch, int gmres_size, int gmres_restart);
	~BJD();

    /*
	//用来实现求解V'(A-tauB)'AV = lam*V'(A-tauB)'BV的，暂时不用
	template<typename Derived_HA, typename Derived_HB, typename Derived_HAB>
	void generalized_RR(Derived_HA& HA, Derived_HB& HB, Derived_HAB& HAB, double tau, MatrixXd& eigenvalues, MatrixXd& eigenvectors) {
		GeneralizedSelfAdjointEigenSolver<MatrixXd> ges;
		ges.compute(HA - tau * HAB.transpose(), HAB - tau * HB);
		com_of_mul += HAB.rows() * HAB.cols() + HB.rows() * HB.cols() + 24 * HA.cols() * HA.cols() * HA.cols();
		eigenvalues = ges.eigenvalues();
		eigenvectors = ges.eigenvectors();
	}
    */

    /*
	//设定好用来求补空间的与U对应的Y和v
	template<typename Derived_U>
	void Minv_set(Derived_U& U) {
		//  [Ki, U;]^(-1)
		//   UT, 0;      * r = z
		//   K1 = Ki^(-1)        
		Y = K1 * U;
		com_of_mul += K1.nonZeros() * U.cols();
		
		v = U.transpose() * Y;
		com_of_mul += U.cols() * U.rows() * Y.cols();

		v = v.inverse();
		com_of_mul += v.cols() * v.cols() * v.cols() * 2 / 3 + 7 * v.cols() * v.cols() - v.cols() - v.cols();
	}
    */

	template<typename Derived_U>
	void Minv_setU(Derived_U& U, double* l, int* ua) {
		/*[Ki, U;]^(-1)
		   UT, 0;      * r = z
		   K1 = Ki^(-1)        */

        //ILU需要
		Y.resize(U.rows(), U.cols());
		double* u = new double[U.rows()];
		double* y = new double[U.rows()];
		Map<MatrixXd> u0(u, U.rows(), 1);
		Map<MatrixXd> y0(y, U.rows(), 1);
		for (int i = 0; i < U.cols(); ++i) {
			u0 = U.col(i);
			lus_cr(A.rows(), nnz, ia, ja, l, ua, u, y);
			Y.col(i) = y0;
		}
		com_of_mul += nnz * U.cols();

		v.noalias() = U.transpose() * Y;
		com_of_mul += U.cols() * U.rows() * Y.cols();

		v = v.inverse();
		com_of_mul += v.cols() * v.cols() * v.cols() * 2 / 3 + 7 * v.cols() * v.cols() - v.cols() - v.cols();

		delete[] u;
		delete[] y;
	}

	template<typename Derived_U, typename Derived_r, typename Derived_z>
	void Minv_mulU(Derived_U& U, Derived_r& r, Derived_z& z, double* l, int* ua) {
		/*[Ki, U;]^(-1)
		   UT, 0;      * r = z
		   K1 = Ki^(-1)        */

        //ILU需要
		double* r0 = new double[A.rows()];
		double* z0 = new double[A.rows()];
		Map<MatrixXd> R(r0, A.rows(), 1);
		Map<MatrixXd> Z(z0, A.rows(), 1);
		R = r.topRows(A.rows());
		lus_cr(A.rows(), nnz, ia, ja, l, ua, r0, z0);
		z.topRows(A.rows()) = Z;
		com_of_mul += nnz;
        
		z.bottomRows(U.cols()) = v * (r.bottomRows(U.cols()) + U.transpose() * z.topRows(A.rows()));
		com_of_mul += v.rows() * v.cols() + U.cols() * U.rows();

		z.topRows(A.rows()) -= Y * z.bottomRows(U.cols());
		com_of_mul += Y.rows() * Y.cols();

        delete[] r0;
        delete[] z0;
	}
	
	template<typename Derived_U>
	void Minv_set(Derived_U& U) {
		/*[Ki, U;]^(-1)
		   UT, 0;      * r = z
		   K1 = Ki^(-1)        */

		Y.resize(U.rows(), U.cols());
		for (int i = 0; i < U.rows(); ++i)
			Y.row(i).noalias() = U.row(i) * tmpAinv[i];
		com_of_mul += U.rows() * U.cols();

		v.noalias() = U.transpose() * Y;
		com_of_mul += U.cols() * U.rows() * Y.cols();

		v = v.inverse();
		com_of_mul += v.cols() * v.cols() * v.cols() * 2 / 3 + 7 * v.cols() * v.cols() - v.cols() - v.cols();
	}

	template<typename Derived_U, typename Derived_r, typename Derived_z>
	void Minv_mul(Derived_U& U, Derived_r& r, Derived_z& z) {
		/*[Ki, U;]^(-1)
		   UT, 0;      * r = z
		   K1 = Ki^(-1)        */

		for (int i = 0; i < A.rows(); ++i)
			z.row(i) = tmpAinv[i] * r.row(i);

		z.bottomRows(U.cols()) = v * (r.bottomRows(U.cols()) + U.transpose() * z.topRows(A.rows()));
		com_of_mul += v.rows() * v.cols() + U.cols() * U.rows();

		z.topRows(A.rows()) -= Y * z.bottomRows(U.cols());
		com_of_mul += Y.rows() * Y.cols();
	}

	//左乘A对应的增广矩阵
	template<typename Derived_U, typename Derived_r, typename Derived_z>
	void A_mul(Derived_U& U, Derived_r& r, Derived_z& z) {
		z.topRows(tmpA.rows()) = tmpA * r.topRows(tmpA.rows()) + U * r.bottomRows(U.cols());
		com_of_mul += tmpA.nonZeros() + U.rows() * U.cols();
		
		z.bottomRows(U.cols()) = U.transpose() * r.topRows(tmpA.rows());
		com_of_mul += U.cols() * U.rows();
	}
	
	/*
    //左预处理GMRES，BJD特化版本
	template<typename Derived_rhs, typename Derived_ss, typename Derived_sol, typename Derived_eval>
	void L_GMRES(SparseMatrix<double, RowMajor>& A, SparseMatrix<double, RowMajor>& B, Derived_rhs& b, Derived_ss& U, Derived_sol& X, Derived_eval& lam, int m) {
		//(求解：（I-UUT）（A-λB）z = b)
		K1.resize(A.rows(), A.cols());
		K1.reserve(A.rows());
		for (int j = 0; j < A.rows(); ++j)
			K1.insert(j, j) = 1;

		MatrixXd x(A.rows() + U.cols(), 1);
		MatrixXd r(x.rows(), 1);
		MatrixXd tpw(x.rows(), 1);
		MatrixXd V(x.rows(), m);
		MatrixXd H(m + 1, m);
		MatrixXd w(x.rows(), 1);
		MatrixXd y;

		for (int i = 0; i < b.cols(); ++i) {

			//注意K逆的生成
			if (i == 0) {
				tmpA = A;
				tmpA -= lam(0, 0) * B;
			}
			else {
				tmpA -= (lam(i, 0) - lam(i - 1, 0)) * B;
			}
			com_of_mul += B.nonZeros();

			//TODO 避免0对角元,也可用其他方法
			//先取预处理矩阵为对角阵
			for (int j = 0; j < A.rows(); ++j)
				if (abs(tmpA.coeff(j, j)) < LinearEigenSolver::ORTH_TOL)
					K1.coeffRef(j, j) = 1 / LinearEigenSolver::ORTH_TOL;
				else
					K1.coeffRef(j, j) = 1 / tmpA.coeff(j, j);
			com_of_mul += A.rows() * 5;

			Minv_set(U);
			x.topRows(tmpA.rows()) = X.col(i);
			x.bottomRows(U.cols()) = MatrixXd::Zero(U.cols(), 1);

			//由于CG和GMRES不能等价，因此用CG中的cgstep近似控制一下计算量
			//这样cgstep仍然代表大矩阵乘法进行的次数，m为GMRES扩展子空间的大小，maxit为GMRES重启次数
			int maxit = cgstep / m;
			for (int it = 0; it < maxit; ++it) {

				A_mul(U, x, r);
				r.topRows(tmpA.rows()) -= b.col(i);
				tpw = -r;
				Minv_mul(U, tpw, r);

				double beta = r.norm();
				V.col(0) = r / beta;
				com_of_mul += 2 * A.rows() + 2 * U.cols();

				//错误就在这里，细心的你能看到吗？
				for (int j = 0; j < m; ++j) {
					A_mul(U, V.col(j), tpw);
					Minv_mul(U, tpw, w);
					for (int k = 0; k <= j; ++k) {
						H(k, j) = (w.transpose() * V.col(k))(0, 0);
						w -= H(k, j) * V.col(k);
					}
					H(j + 1, j) = w.norm();
					if (j < m - 1)
						V.col(j + 1) = w / H(j + 1, j);
					com_of_mul += (j + 1) * V.rows() * 2 + V.rows();

					if (H(j + 1, j) < LinearEigenSolver::ORTH_TOL) {
						H.conservativeResize(j + 2, j + 1);
						V.conservativeResize(x.rows(), j + 1);
						break;
					}
				}

				//求解y
				y = MatrixXd::Zero(H.rows(), 1);
				y(0, 0) = beta;
				for (int j = 0; j < H.rows() - 1; ++j) {
					double tmp = sqrt(H(j, j) * H(j, j) + H(j + 1, j) * H(j + 1, j));
					double s = H(j + 1, j) / tmp;
					double c = H(j, j) / tmp;
					Matrix2d omega;
					omega << c, s,
						-s, c;
					H.block(j, j, 2, H.cols() - j).applyOnTheLeft(omega);
					y(j + 1, 0) = -s * y(j, 0);
					y(j, 0) *= c;
				}
				com_of_mul += H.rows() * 30 + 2 * H.rows() * H.rows();

				for (int j = H.rows() - 2; j >= 0; --j) {
					y(j, 0) /= H(j, j);
					for (int k = j - 1; k >= 0; --k) {
						y(k, 0) -= H(k, j) * y(j, 0);
					}
				}
				com_of_mul += (H.rows() + 1) * H.rows() / 2 + 5 * H.rows();

				x += V * y.topRows(V.cols());
				com_of_mul += V.rows() * V.cols();

				if (abs(y(y.rows() - 1, 0)) < LinearEigenSolver::ORTH_TOL)
					break;
			}
			X.col(i) = x.topRows(A.rows());
		}
	}
    */

    template<typename Derived_rhs, typename Derived_ss, typename Derived_sol, typename Derived_eval>
    void PMGMRES(Derived_rhs& b, Derived_ss& U, Derived_sol& X, Derived_eval& lam) {
        //CRSsubtrac(ia, ja, a, nnz, B, 0);

        double av;
        double* c;
        double delta = 1.0e-03;
        double* g;
        double* h;
        double htmp;
        int i;
        int itr;
        int itr_used;
        int j;
        int k;
        int k_copy;
        double mu;
        double* r;
        double rho;
        double rho_tol;
        double* s;
        double* v;
        int verbose = 0;
        double* y;
        //double* l;
        //int* ua;

        int n = A.rows() + U.cols();
        c = new double[gmres_size + 1];
        g = new double[gmres_size + 1];
        h = new double[(gmres_size + 1) * gmres_size];
        r = new double[n];
        s = new double[gmres_size + 1];
        v = new double[(gmres_size + 1) * n];
        y = new double[gmres_size + 1];
        //l = new double[ia[A.rows()] + 1];
        //ua = new int[A.rows()];

        Map<MatrixXd> R(r, n, 1);
        Map<MatrixXd> V(v, n, gmres_size + 1);
        Map<MatrixXd> Y(y, gmres_size + 1, 1);

        MatrixXd x(A.rows() + U.cols(), 1);
        tmpA = A;
        for (int index = 0; index < b.cols(); ++index) {
            if (index == 0) {
                //CRSsubtrac(ia, ja, a, nnz, B, lam(0, 0));
                tmpA -= lam(0, 0) * B;
            }
            else {
                //CRSsubtrac(ia, ja, a, nnz, B, lam(index, 0) - lam(index - 1, 0));
                tmpA -= (lam(index, 0) - lam(index - 1, 0)) * B;
            }
            //for (int row = 0; row < A.rows(); ++row) {
            //    double diag = tmpA.coeff(row, row);
            //    if (abs(diag) > ORTH_TOL)
            //        tmpAinv[row] = 1.0 / diag;
            //    else
            //        tmpAinv[row] = ((diag < 0) ? -1.0 : 1.0) / ORTH_TOL;
            //}

            itr_used = 0;

            x.topRows(A.rows()) = X.col(index);
            x.bottomRows(U.cols()) = MatrixXd::Zero(U.cols(), 1);

            //ILU分解
            //diagonal_pointer_cr(A.rows(), nnz, ia, ja, ua);
            //ilu_cr(A.rows(), nnz, ia, ja, a, ua, l);
            com_of_mul += 2 * tmpA.nonZeros();

            //Minv_set(U);
            if (verbose)
            {
                cout << "\n";
                cout << "PMGMRES_ILU_CR\n";
                cout << "  Number of unknowns = " << n << "\n";
            }

            for (itr = 0; itr < gmres_restart; itr++)
            {
                A_mul(U, x, R);

                R *= -1;
                R.topRows(A.rows()) += b.col(index);
                com_of_mul += A.rows();

                //Minv_mul(U, R, R);

                //rho = sqrt(r8vec_dot(n, r, r));
                rho = R.norm();

                if (verbose)
                {
                    cout << "  ITR = " << itr << "  Residual = " << rho << "\n";
                }

                if (itr == 0)
                {
                    rho_tol = rho * LinearEigenSolver::ORTH_TOL;
                }

                V.col(0) = R / rho;

                g[0] = rho;
                for (i = 1; i < gmres_size + 1; i++)
                {
                    g[i] = 0.0;
                }

                for (i = 0; i < gmres_size + 1; i++)
                {
                    for (j = 0; j < gmres_size; j++)
                    {
                        h[i * (gmres_size)+j] = 0.0;
                    }
                }

                for (k = 0; k < gmres_size; k++)
                {
                    k_copy = k;

                    A_mul(U, V.col(k), V.col(k + 1));
                    //Minv_mul(U, V.col(k + 1), V.col(k + 1));

                    //av = sqrt(r8vec_dot(n, v + (k + 1) * n, v + (k + 1) * n));
                    av = V.col(k + 1).norm();
                    com_of_mul += n;

                    for (j = 0; j <= k; j++)
                    {
                        //h[j * gmres_size + k] = r8vec_dot(n, v + (k + 1) * n, v + j * n);
                        h[j * gmres_size + k] = (V.col(k + 1).transpose() * V.col(j))(0, 0);
                        V.col(k + 1) -= h[j * gmres_size + k] * V.col(j);
                    }
                    com_of_mul += n * (k + 1) * 2;

                    //h[(k + 1) * gmres_size + k] = sqrt(r8vec_dot(n, v + (k + 1) * n, v + (k + 1) * n));
                    h[(k + 1) * gmres_size + k] = V.col(k + 1).norm();
                    com_of_mul += n;

                    if ((av + delta * h[(k + 1) * gmres_size + k]) == av)
                    {
                        for (j = 0; j < k + 1; j++)
                        {
                            //htmp = r8vec_dot(n, v + (k + 1) * n, v + j * n);
                            htmp = V.col(k + 1).transpose() * V.col(j);
                            com_of_mul += n;

                            h[j * gmres_size + k] = h[j * gmres_size + k] + htmp;
                            V.col(k + 1) = V.col(k + 1) - htmp * V.col(j);
                            com_of_mul += n;
                        }
                        //h[(k + 1) * gmres_size + k] = sqrt(r8vec_dot(n, v + (k + 1) * n, v + (k + 1) * n));
                        h[(k + 1) * gmres_size + k] = V.col(k + 1).norm();
                        com_of_mul += n;
                    }

                    if (h[(k + 1) * gmres_size + k] != 0.0)
                    {
                        V.col(k + 1) = V.col(k + 1) / h[(k + 1) * gmres_size + k];
                        com_of_mul += 5 * n;
                    }

                    if (0 < k)
                    {
                        for (i = 0; i < k + 2; i++)
                        {
                            y[i] = h[i * gmres_size + k];
                        }
                        for (j = 0; j < k; j++)
                        {
                            mult_givens(c[j], s[j], j, y);
                        }
                        for (i = 0; i < k + 2; i++)
                        {
                            h[i * gmres_size + k] = y[i];
                        }
                    }
                    mu = sqrt(h[k * gmres_size + k] * h[k * gmres_size + k] + h[(k + 1) * gmres_size + k] * h[(k + 1) * gmres_size + k]);
                    c[k] = h[k * gmres_size + k] / mu;
                    s[k] = -h[(k + 1) * gmres_size + k] / mu;
                    h[k * gmres_size + k] = c[k] * h[k * gmres_size + k] - s[k] * h[(k + 1) * gmres_size + k];
                    h[(k + 1) * gmres_size + k] = 0.0;
                    mult_givens(c[k], s[k], k, g);

                    rho = fabs(g[k + 1]);

                    itr_used = itr_used + 1;

                    if (verbose)
                    {
                        cout << "  K   = " << k << "  Residual = " << rho << "\n";
                    }

                    if (rho <= rho_tol && rho <= sqrt(LinearEigenSolver::ORTH_TOL))
                    {
                        break;
                    }
                }

                k = k_copy;

                y[k] = g[k] / h[k * gmres_size + k];
                for (i = k - 1; 0 <= i; i--)
                {
                    y[i] = g[i];
                    for (j = i + 1; j < k + 1; j++)
                    {
                        y[i] = y[i] - h[i * gmres_size + j] * y[j];
                    }
                    y[i] = y[i] / h[i * gmres_size + i];
                }
                com_of_mul += k * (k + 1) / 2;

                x += V.leftCols(k + 1) * Y.topRows(k + 1);
                com_of_mul += V.rows() * (k + 1);

                if (rho <= rho_tol && rho <= sqrt(LinearEigenSolver::ORTH_TOL))
                {
                    break;
                }
            }

            X.col(index) = x.topRows(A.rows());

            if (verbose)
            {
                cout << "\n";;
                cout << "PMGMRES_ILU_CR:\n";
                cout << "  Iterations = " << itr_used << "\n";
                cout << "  Final residual = " << rho << "\n";
            }
        }
        //CRSsubtrac(ia, ja, a, nnz, B, -lam(b.cols() - 1, 0));
        //
        //  Free memory.
        //
        delete[] c;
        delete[] g;
        delete[] h;
        delete[] r;
        delete[] s;
        delete[] v;
        delete[] y;
        //delete[] l;
        //delete[] ua;
    }
	void compute();
};

#endif

#include<mtxio.h>

SparseMatrix<double, RowMajor>& mtxio::getSparseMatrix(string filename, string filepath) {
        return mtxio::getSparseMatrix(filepath + filename);
    }
SparseMatrix<double, RowMajor>& mtxio::getSparseMatrix(string filename) {
        
        ifstream fin(filename);
        int M, N, L;
        //Ignore headers and comments
        while (fin.peek() == '%')
            fin.ignore(2048, '\n');

        fin >> M >> N >> L;
        // Eigen::setNbThreads(8);

        SparseMatrix<double, RowMajor>* mat = new SparseMatrix<double, RowMajor>;
        SparseMatrix<double, RowMajor>& matrix = *mat;
        matrix.resize(M, N);
        matrix.reserve(L * 2 - M);
        vector<Eigen::Triplet<double>> triple;
        for (int i = 0; i < L; ++i) {
            int m, n;
            double data;
            fin >> m >> n >> data;
            triple.push_back(Triplet<double>(m - 1, n - 1, data));// m - 1 and n - 1 to set index start from 0
            if (m != n)
                triple.push_back(Triplet<double>(n - 1, m - 1, data));
        }
        fin.close();
        matrix.setFromTriplets(triple.begin(), triple.end());

        //SparseVector<double, RowMajor> vec(N);

        //cout << matrix << endl;
        return matrix;
    }
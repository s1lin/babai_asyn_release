/** \file
 * \brief Computation of overdetermined integer linear models
 * \author Shilei Lin
 * This file is part of 
 *   CILS is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   CILS is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with   If not, see <http://www.gnu.org/licenses/>.
 */

namespace cils {

    template<typename scalar, typename index>
    class CILS_OLM {
    private:

        void init_R_A() {
            R_A.resize(n / 2 * (1 + n), false);
            index idx = 0;
            for (index row = 0; row < R.size1(); row++) {
                for (index col = row; col < R.size2(); col++) {
                    R_A[idx] = R(row, col);
                    idx++;
                }
            }
        }

    public:
        index n, m, is_constrained, upper, qam, offset, search_iter;
        b_vector z_hat{}, R_A{}, y_bar{}, y{};
        si_vector d;
        b_matrix R{}, A{};

        CILS_OLM() {}

        CILS_OLM(CILS<scalar, index> &cils, b_vector &z_hat, b_matrix &R, b_vector &y_bar) {
            this->n = cils.n;
            this->m = cils.m;
            this->upper = cils.upper;
            this->search_iter = cils.search_iter;
            this->qam = cils.qam;
            this->offset = cils.offset;
            this->is_constrained = cils.is_constrained;
            this->R.assign(R);
            this->y_bar.assign(y_bar);
            this->z_hat.assign(z_hat);
            this->y.assign(cils.y);
            this->A.assign(cils.A);
            this->d = cils.d;
            init_R_A();
        }

        void reset(b_matrix &R, b_vector &y_bar, index upper, index block_size, index is_constrained, index qam = 3) {
            this->n = R.size1();
            this->m = R.size2();
            this->upper = upper;
            this->is_constrained = is_constrained;
            this->R.assign(R);
            this->y_bar.assign(y_bar);
            this->z_hat.resize(n);
            this->z_hat.clear();
            this->y.assign(y);
            this->qam = qam;
            d.resize(n / block_size);
            std::fill(d.begin(), d.end(), block_size);
            for (index i = d.size() - 2; i >= 0; i--) {
                d[i] += d[i + 1];
            }
            init_R_A();
        }

        returnType<scalar, index>
        pbnp(const index n_t, const index nstep) {

            index idx = 0, ni, nj, diff = 0, c_i, t = 0;
            index z_p[n] = {}, ct[nstep] = {}, df[nstep] = {}, z_n;
            b_vector y_b;
            y_b = y_bar;
            bool flag = false;
            scalar sum = 0;

#pragma omp parallel default(shared) num_threads(n_t)
            {}

            scalar run_time = omp_get_wtime();
            if (nstep != 0) {
                c_i = round(y_b[n - 1] / R_A[n / 2 * (1 + n) - 1]);
                z_n = !is_constrained ? c_i : max(min((index) c_i, upper), 0);
                z_hat[n - 1] = z_n;
                idx = n - 2;

#pragma omp parallel default(shared) num_threads(n_t) private(sum, ni, nj, c_i) firstprivate(z_n, t)
                {
//#pragma omp barrier
//#pragma omp for schedule(static, 4) nowait
//                    for (index i = 0; i < n; i++) {
//                        y_b[i] -= z_n * R_A(i, n - 1, n);
//                    }
                    for (t = 0; t < nstep && !flag; t++) {
#pragma omp for nowait schedule(dynamic)
                        for (index i = 1; i < n; i++) {
                            ni = n - 1 - i;
                            if (!flag && ni <= idx) {
                                sum = y_b[ni];
                                nj = ni * n - (ni * (ni + 1)) / 2;
#pragma omp simd reduction(- : sum)
                                for (index j = n - 1; j >= n - i; j--) { //j < n - 1; j++) {
                                    sum -= R_A(nj, j) * z_hat[j]; //[nj + j]
                                }

                                c_i = round(sum / R_A(nj, ni));
                                z_hat[ni] = !is_constrained ? c_i : max(min((index) c_i, upper), 0);
#pragma omp atomic
                                df[t] += z_p[ni] != z_hat[ni];

                                z_p[ni] = z_hat[ni];
                                if (idx == ni) {
#pragma omp atomic
                                    idx--;
                                }
                            }
#pragma omp atomic
                            ct[t]++;
                        }
                        if (!flag) {
                            flag = (df[t] <= 100 || idx <= n_t) && ct[t] == n - 1;
                        }
                    }
                }
            }
            run_time = omp_get_wtime() - run_time;
            helper::display<index, index>(df, nstep, "df");
            cout << idx << endl;
            returnType < scalar, index > reT = {{}, run_time, (scalar) t};
            return reT;
        }

        returnType<scalar, index>
        pbnp2(const index n_t, const index nstep, const index init) {

            index idx = 0, ni, nj, diff = 0, c_i, t = 0, s = 0;
            index z_p[n] = {}, ct[nstep] = {}, df[nstep] = {}, z_n, delta[n] = {};
            b_vector y_b;
            y_b = y_bar;
            bool flag = false;

            scalar sum = 0;

            auto lock = new omp_lock_t[n]();
            for (index i = 0; i < n; i++) {
                omp_init_lock((&lock[i]));
            }

#pragma omp parallel default(shared) num_threads(n_t)
            {}

            scalar run_time = omp_get_wtime();
            if (nstep != 0) {
                c_i = round(y_b[n - 1] / R_A[n / 2 * (1 + n) - 1]);
                z_n = !is_constrained ? c_i : max(min((index) c_i, upper), 0);
                z_hat[n - 1] = z_n;
                idx = n - 2;

#pragma omp parallel default(shared) num_threads(n_t) private(sum, ni, nj, c_i) firstprivate(z_n, t)
                {
                    for (t = 0; t < nstep && !flag; t++) {
#pragma omp for nowait schedule(dynamic)
                        for (index i = 1; i < n; i++) {
                            ni = n - 1 - i;
                            if (!flag && ni <= idx && s < 700) {
                                omp_set_lock(&lock[i]);
                                sum = y_b[ni];
                                nj = ni * n - (ni * (ni + 1)) / 2;
#pragma omp simd reduction(- : sum)
                                for (index j = n - 1; j >= n - i; j--) {
                                    sum -= R_A(nj, j) * z_hat[j];
                                }
                                c_i = round(sum / R_A(nj, ni));
                                z_hat[ni] = !is_constrained ? c_i : max(min((index) c_i, upper), 0);
                                delta[ni] = z_p[ni] != z_hat[ni];
                                z_p[ni] = z_hat[ni];
                                if (idx == ni) {
#pragma omp atomic
                                    idx--;
                                }
                                omp_unset_lock(&lock[i]);
#pragma omp atomic
                                s++;
                            }
                        }
                        if (!flag || idx <= n_t || s < 700) {
#pragma omp for nowait
                            for (index i = 1; i < idx; i++) {
                                ni = n - 1 - i;
                                diff += delta[ni];
#pragma omp atomic
                                ct[t]++;
                            }
                            if (ct[t] >= idx - 2) {
//                                if (init == 1)
//                                    flag = diff == 0;
//                                if (init != 1)
                                flag = diff <= 100 || idx <= n_t;
#pragma omp atomic write
                                diff = 0;
                            }
                        } else
                            flag = true;
                    }
                }
            }
            run_time = omp_get_wtime() - run_time;

//            helper::display<index, index>(ct, nstep, "ct");
//            cout << diff << "," << idx << t << endl;
            for (index i = 0; i < n; i++) {
                omp_destroy_lock(&lock[i]);
            }
            delete[] lock;
            return {{}, run_time, (scalar) s};
        }

        returnType<scalar, index> bnp() {
            scalar sum = 0;
            scalar time = omp_get_wtime();
            for (index i = n - 1; i >= 0; i--) {
                for (index j = i + 1; j < n; j++) {
                    sum += R(i, j) * z_hat[j];
                }
                scalar c_i = round((y_bar[i] - sum) / R(i, i));
                z_hat[i] = !is_constrained ? c_i : max(min((index) c_i, upper), 0);
                sum = 0;
            }
            time = omp_get_wtime() - time;
            return {{}, time, 0};
        }


        returnType<scalar, index> rbb(index k) {
            b_matrix domain, edges, range;
            b_vector dd, x_temp(n, 0);
            double g, bsum, r, sumw;
            int aoffset, b_k, i, idx, nn, nx;

            // Input:  R is upper triangular
            //         y is the target REAL vector in the lattice.
            //         l is the lower bound of the box
            //         u is the upper bound of the box
            //         k is the number of runs to obtain the optimal one
            // Output: x is an estimate
            // 'random_babai:9' g = 10/min(abs(diag(R)))^2;
            sumw = INFINITY;
            for (i = 0; i < n; i++) {
                if (abs(R(i, i)) < sumw)
                    sumw = abs(R(i, i));
            }
            g = 10.0 / (sumw * sumw);
            //  This is a parameter used in the algorithm
            // 'random_babai:10' nn = size(R,2);
            // 'random_babai:11' x = l;
            // 'random_babai:12' r = norm(y-R*x);
            r = helper::find_residual<scalar, index>(R, x_temp, y);
            // 'random_babai:13' x_temp = zeros(nn,1);
            // 'random_babai:15' for j = 1:k
            scalar time = omp_get_wtime();
            for (int j = 0; j < k; j++) {
                int lastBlockLength;
                // 'random_babai:16' for i = nn:-1:1
                for (i = n - 1; i >= 0; i--) {
                    scalar sum = 0;
                    for (int t = i + 1; t < n; t++) {
                        sum += R(i, t) * z_hat[t];
                    }
                    scalar c = round((y_bar[i] - sum) / R(i, i));
                    z_hat[i] = !is_constrained ? c : max(min((index) c, upper), 0);
                    // 'random_babai:22' [domain,range] = klein_dist(g*R(i,i)^2,c,l(i),u(i));
                    sumw = R[i + R.size1() * i];
                    sumw = g * (sumw * sumw);
                    bsum = 0;
                    // Input: g is the parameter controlling the distribution. As g->0 the
                    // distribution tends to a discrete uniform.
                    //        c is the real-valued input to be rounded.
                    //        bl and bu are the lower and upper ends of the constraint
                    //        interval
                    //
                    // Output: domain - candidates with probability > 0 (in MATLAB)
                    //         range - probabilities of each element in the domain

                    // 'klein_dist:11' domain = zeros(1,bu-bl+1);
                    nx = upper + 1.0;
                    domain.resize(1, nx, false);
                    // 'klein_dist:12' range = zeros(1,bu-bl+1);
                    range.resize(1, nx, false);

                    // 'klein_dist:13' index=0;
                    // 'klein_dist:14' for i=bl:bu
                    lastBlockLength = static_cast<int>(upper + (1.0 - 0));
                    for (nx = 0; nx < lastBlockLength; nx++) {
                        double d_i = bsum + static_cast<double>(nx);
                        // 'klein_dist:15' index = index+1;
                        // 'klein_dist:16' s = 0;
                        double s = 0.0;
                        // 'klein_dist:17' for j = bl:bu
                        for (idx = 0; idx < lastBlockLength; idx++) {
                            double b_j = bsum + static_cast<double>(idx);
                            // 'klein_dist:18' s = s+exp(-g*(2*c-j-i)*(i-j));
                            s += std::exp(-sumw * ((2.0 * c - b_j) - d_i) * (d_i - b_j));
                        }
                        // 'klein_dist:20' prob = 1/s;
                        range[nx] = 1.0 / s;
                        // 'klein_dist:21' domain(index) = i;
                        domain[nx] = d_i;
                        // 'klein_dist:22' range(index) = prob;
                    }
                    // 'random_babai:23' x_temp(i) = randsample(domain,1,true,range);
                    if (range.size2() != 0) {
                        if (range.size2() <= 1024) {
                            nx = range.size2();
                            lastBlockLength = 0;
                            aoffset = 1;
                        } else {
                            nx = 1024;
                            aoffset = range.size2() / 1024;
                            lastBlockLength = range.size2() - (aoffset << 10);
                            if (lastBlockLength > 0) {
                                aoffset++;
                            } else {
                                lastBlockLength = 1024;
                            }
                        }
                        sumw = range[0];
                        for (b_k = 2; b_k <= nx; b_k++) {
                            sumw += range[b_k - 1];
                        }
                        for (int ib{2}; ib <= aoffset; ib++) {
                            nx = (ib - 1) << 10;
                            bsum = range[nx];
                            if (ib == aoffset) {
                                idx = lastBlockLength;
                            } else {
                                idx = 1024;
                            }
                            for (b_k = 2; b_k <= idx; b_k++) {
                                bsum += range[(nx + b_k) - 1];
                            }
                            sumw += bsum;
                        }
                        edges.resize(1, domain.size2() + 1);
                        edges[0] = 0.0;
                        edges[domain.size2()] = 1.0;
                        lastBlockLength = domain.size2();
                        for (idx = 0; idx <= lastBlockLength - 2; idx++) {
                            edges[idx + 1] = std::fmin(edges[idx] + range[idx] / sumw, 1.0);
                        }
                    } else {
                        edges.resize(1, 1);
                        edges[0] = 0.0;
                    }
                    std::random_device rd;
                    std::mt19937 e2(rd());
                    std::uniform_real_distribution<> dist(0, 1);

                    if (range.size2() == 0) {
                        sumw = dist(e2);
                        sumw = std::floor(sumw * static_cast<double>(domain.size2())) + 1.0;
                    } else {
                        sumw = dist(e2);
                        idx = 0;
                        if (!std::isnan(sumw)) {
                            if ((sumw >= edges[0]) && (sumw < edges[edges.size2() - 1])) {
                                nx = edges.size2();
                                idx = 1;
                                aoffset = 2;
                                while (nx > aoffset) {
                                    lastBlockLength = (idx >> 1) + (nx >> 1);
                                    if (((idx & 1) == 1) && ((nx & 1) == 1)) {
                                        lastBlockLength++;
                                    }
                                    if (sumw >= edges[lastBlockLength - 1]) {
                                        idx = lastBlockLength;
                                        aoffset = lastBlockLength + 1;
                                    } else {
                                        nx = lastBlockLength;
                                    }
                                }
                            }
                            if (sumw == edges[edges.size2() - 1]) {
                                idx = edges.size2();
                            }
                        }
                        sumw = idx;
                    }
                    x_temp[i] = domain[static_cast<int>(sumw) - 1];
                }
                // 'random_babai:25' r_temp = norm(y-R*x_temp);
                bsum = helper::find_residual<scalar, index>(R, x_temp, y);
                // 'random_babai:26' if r_temp < r
                if (bsum < r) {
                    // 'random_babai:27' x = x_temp;
                    z_hat.assign(x_temp);
                    // 'random_babai:28' r = r_temp;
                    r = bsum;
                }
            }
            time = omp_get_wtime() - time;
            return {{}, time, 0};
        }

        returnType<scalar, index> prbb(index k, index n_t) {
            b_matrix domain, edges, range;
            b_vector dd, x_temp(n, 0);
            double g, bsum, r, sumw;
            int aoffset, b_k, i, idx, nx;

            // Input:  R is upper triangular
            //         y is the target REAL vector in the lattice.
            //         l is the lower bound of the box
            //         u is the upper bound of the box
            //         k is the number of runs to obtain the optimal one
            // Output: x is an estimate
            // 'random_babai:9' g = 10/min(abs(diag(R)))^2;
            sumw = INFINITY;
            for (i = 0; i < n; i++) {
                if (abs(R(i, i)) < sumw)
                    sumw = abs(R(i, i));
            }
            g = 10.0 / (sumw * sumw);
            //  This is a parameter used in the algorithm
            // 'random_babai:10' nn = size(R,2);
            // 'random_babai:11' x = l;
            // 'random_babai:12' r = norm(y-R*x);
            r = helper::find_residual<scalar, index>(R, x_temp, y);
            // 'random_babai:13' x_temp = zeros(nn,1);
            // 'random_babai:15' for j = 1:k

            scalar time = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_t) private(domain, edges, range, dd, bsum, sumw, i, idx, nx, aoffset) firstprivate(x_temp)
            {
#pragma omp for nowait
                for (int j = 0; j < k; j++) {
                    int lastBlockLength;
                    // 'random_babai:16' for i = nn:-1:1
                    for (i = n - 1; i >= 0; i--) {
                        scalar sum = 0;
                        for (int t = i + 1; t < n; t++) {
                            sum += R(i, t) * z_hat[t];
                        }
                        scalar c = round((y_bar[i] - sum) / R(i, i));
                        z_hat[i] = !is_constrained ? c : max(min((index) c, upper), 0);
                        // 'random_babai:22' [domain,range] = klein_dist(g*R(i,i)^2,c,l(i),u(i));
                        sumw = R[i + R.size1() * i];
                        sumw = g * (sumw * sumw);
                        bsum = 0;
                        // Input: g is the parameter controlling the distribution. As g->0 the
                        // distribution tends to a discrete uniform.
                        //        c is the real-valued input to be rounded.
                        //        bl and bu are the lower and upper ends of the constraint
                        //        interval
                        //
                        // Output: domain - candidates with probability > 0 (in MATLAB)
                        //         range - probabilities of each element in the domain

                        // 'klein_dist:11' domain = zeros(1,bu-bl+1);
                        nx = upper + 1.0;
                        domain.resize(1, nx, false);
                        // 'klein_dist:12' range = zeros(1,bu-bl+1);
                        range.resize(1, nx, false);

                        // 'klein_dist:13' index=0;
                        // 'klein_dist:14' for i=bl:bu
                        lastBlockLength = static_cast<int>(upper + (1.0 - 0));
                        for (nx = 0; nx < lastBlockLength; nx++) {
                            double d_i = bsum + static_cast<double>(nx);
                            // 'klein_dist:15' index = index+1;
                            // 'klein_dist:16' s = 0;
                            double s = 0.0;
                            // 'klein_dist:17' for j = bl:bu
                            for (idx = 0; idx < lastBlockLength; idx++) {
                                double b_j = bsum + static_cast<double>(idx);
                                // 'klein_dist:18' s = s+exp(-g*(2*c-j-i)*(i-j));
                                s += std::exp(-sumw * ((2.0 * c - b_j) - d_i) * (d_i - b_j));
                            }
                            // 'klein_dist:20' prob = 1/s;
                            range[nx] = 1.0 / s;
                            // 'klein_dist:21' domain(index) = i;
                            domain[nx] = d_i;
                            // 'klein_dist:22' range(index) = prob;
                        }
                        // 'random_babai:23' x_temp(i) = randsample(domain,1,true,range);
                        if (range.size2() != 0) {
                            if (range.size2() <= 1024) {
                                nx = range.size2();
                                lastBlockLength = 0;
                                aoffset = 1;
                            } else {
                                nx = 1024;
                                aoffset = range.size2() / 1024;
                                lastBlockLength = range.size2() - (aoffset << 10);
                                if (lastBlockLength > 0) {
                                    aoffset++;
                                } else {
                                    lastBlockLength = 1024;
                                }
                            }
                            sumw = range[0];
                            for (b_k = 2; b_k <= nx; b_k++) {
                                sumw += range[b_k - 1];
                            }
                            for (int ib{2}; ib <= aoffset; ib++) {
                                nx = (ib - 1) << 10;
                                bsum = range[nx];
                                if (ib == aoffset) {
                                    idx = lastBlockLength;
                                } else {
                                    idx = 1024;
                                }
                                for (b_k = 2; b_k <= idx; b_k++) {
                                    bsum += range[(nx + b_k) - 1];
                                }
                                sumw += bsum;
                            }
                            edges.resize(1, domain.size2() + 1);
                            edges[0] = 0.0;
                            edges[domain.size2()] = 1.0;
                            lastBlockLength = domain.size2();
                            for (idx = 0; idx <= lastBlockLength - 2; idx++) {
                                edges[idx + 1] = std::fmin(edges[idx] + range[idx] / sumw, 1.0);
                            }
                        } else {
                            edges.resize(1, 1);
                            edges[0] = 0.0;
                        }
                        std::random_device rd;
                        std::mt19937 e2(rd());
                        std::uniform_real_distribution<> dist(0, 1);

                        if (range.size2() == 0) {
                            sumw = dist(e2);
                            sumw = std::floor(sumw * static_cast<double>(domain.size2())) + 1.0;
                        } else {
                            sumw = dist(e2);
                            idx = 0;
                            if (!std::isnan(sumw)) {
                                if ((sumw >= edges[0]) && (sumw < edges[edges.size2() - 1])) {
                                    nx = edges.size2();
                                    idx = 1;
                                    aoffset = 2;
                                    while (nx > aoffset) {
                                        lastBlockLength = (idx >> 1) + (nx >> 1);
                                        if (((idx & 1) == 1) && ((nx & 1) == 1)) {
                                            lastBlockLength++;
                                        }
                                        if (sumw >= edges[lastBlockLength - 1]) {
                                            idx = lastBlockLength;
                                            aoffset = lastBlockLength + 1;
                                        } else {
                                            nx = lastBlockLength;
                                        }
                                    }
                                }
                                if (sumw == edges[edges.size2() - 1]) {
                                    idx = edges.size2();
                                }
                            }
                            sumw = idx;
                        }
                        x_temp[i] = domain[static_cast<int>(sumw) - 1];
                    }
                    // 'random_babai:25' r_temp = norm(y-R*x_temp);
                    bsum = helper::find_residual<scalar, index>(R, x_temp, y);
                    // 'random_babai:26' if r_temp < r
                    if (bsum < r) {
                        // 'random_babai:27' x = x_temp;
                        z_hat.assign(x_temp);
                        // 'random_babai:28' r = r_temp;
                        r = bsum;
                    }
                }
            }
            time = omp_get_wtime() - time;
            return {{}, time, 0};
        }

        returnType<scalar, index> sic(const index nstep) {
            b_vector sum(n, 0), z(z_hat);
            b_vector a_t(n, 0);
            y_bar.resize(m);
            y_bar.assign(y);
            for (index i = 0; i < n; i++) {
                b_vector a = column(A, i);
                a_t[i] = inner_prod(a, a);
            }
            index t = 0;
            scalar time = omp_get_wtime();
            for (t = 0; t < nstep; t++) {
                index diff = 0;
                for (index i = n - 1; i >= 0; i--) {
                    for (index j = 0; j < i; j++) {
                        sum += column(A, j) * z_hat[j];
                    }
                    for (index j = i + 1; j < n; j++) {
                        sum += column(A, j) * z_hat[j];
                    }
                    y_bar = y - sum;
                    scalar c_i = round(inner_prod(column(A, i), y_bar) / a_t[i]);
                    z_hat[i] = !is_constrained ? c_i : max(min((index) c_i, upper), 0);
                    diff += z_hat[i] - z[i];
                    sum.clear();
                }
                //Converge:
                if (diff == 0) {
                    break;
                } else {
                    z.assign(z_hat);
                }
            }

            time = omp_get_wtime() - time;
            return {{}, time, t};
        }

        returnType<scalar, index> psic(const index nstep, const index n_t) {
            b_vector sum(n, 0), z(z_hat);
            b_vector a_t(n, 0), y_b(y);
            index diff[nstep], flag = 0, num_iter = 0;
            for (index i = 0; i < n; i++) {
                b_vector a = column(A, i);
                a_t[i] = inner_prod(a, a);
            }

            scalar time = omp_get_wtime();

#pragma omp parallel default(shared) num_threads(n_t) firstprivate(sum, y_b)
            {
                for (index t = 0; t < nstep && !flag; t++) {
#pragma omp for schedule(dynamic) nowait
                    for (index i = n - 1; i >= 0; i--) {
                        for (index j = 0; j < i; j++) {
                            sum += column(A, j) * z_hat[j];
                        }
                        for (index j = i + 1; j < n; j++) {
                            sum += column(A, j) * z_hat[j];
                        }
                        y_bar = y - sum;
                        scalar c_i = round(inner_prod(column(A, i), y_bar) / a_t[i]);
                        z_hat[i] = !is_constrained ? c_i : max(min((index) c_i, upper), 0);
                        diff[t] += z_hat[i] - z[i];
                        sum.clear();
                    }
                    if (diff[t] == 0 && t > 2) {
                        num_iter = t;
                        flag = 1;
                    } else {
                        z.assign(z_hat);
                    }
                }
            }

            time = omp_get_wtime() - time;
            return {{}, time, num_iter};
        }

        returnType<scalar, index> backsolve() {
            scalar sum = 0;
            scalar time = omp_get_wtime();
            for (index i = n - 1; i >= 0; i--) {
                for (index j = i + 1; j < n; j++) {
                    sum += R(i, j) * z_hat[j];
                }
                z_hat[i] = (y_bar[i] - sum) / R(i, i);
                sum = 0;
            }
            time = omp_get_wtime() - time;
            return {{}, time, 0};
        }

        returnType<scalar, index> bocb() {
            scalar sum = 0;
            index ds = d.size(), n_dx_q_0, n_dx_q_1;
            b_vector y_b(n, 0);
            CILS_SECH_Search<scalar, index> search(n, n, qam, search_iter);

            scalar run_time = omp_get_wtime();
            if (ds == 1) {
                search.ch(0, n, 1, R, y_b, z_hat);
                run_time = omp_get_wtime() - run_time;
                return {{}, run_time, 0};
            }

            for (index i = 0; i < ds; i++) {
                n_dx_q_1 = d[i];
                n_dx_q_0 = i == ds - 1 ? 0 : d[i + 1];

                for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                    sum = 0;
                    for (index col = n_dx_q_1; col < n; col++) {
                        sum += R(row, col) * z_hat[col];
                    }
                    y_b[row] = y_bar[row] - sum;
                }
                if (is_constrained)
                    search.ch(n_dx_q_0, n_dx_q_1, 1, R, y_b, z_hat);
                else
                    search.se(n_dx_q_0, n_dx_q_1, 1, R, y_b, z_hat);
//                for (index tt = n_dx_q_0; tt < n_dx_q_1; tt++){
//                    cout << z_hat[tt] << ", ";
//                }
//                cout << endl;
            }
//            cout << z_hat;
            run_time = omp_get_wtime() - run_time;
            returnType<scalar, index> reT = {{}, run_time, 0};
            return reT;
        }

        returnType<scalar, index> bocb2() {
            scalar sum = 0;
            index ds = d.size(), n_dx_q_0, n_dx_q_1, row_n;
            b_vector y_b(n);
//            cout << R;
            CILS_SECH_Search<scalar, index> search(n, n, qam, search_iter);
            scalar run_time = omp_get_wtime();
            if (ds == 1) {
                search.ch(0, n, 1, R, y_b, z_hat);
                run_time = omp_get_wtime() - run_time;
                return {{}, run_time, 0};
            }
            run_time = omp_get_wtime();

            for (index i = 0; i < ds; i++) {
                n_dx_q_1 = d[i];
                n_dx_q_0 = i == ds - 1 ? 0 : d[i + 1];
//                cout << ((n_dx_q_0 - 1) * (2 * n - n_dx_q_0)) / 2 << ", " << (n_dx_q_0 - 1) * (n - n_dx_q_0 / 2);
                row_n = ((n_dx_q_0 - 1) * (2 * n - n_dx_q_0)) / 2;
                for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                    sum = 0;
                    row_n += n - row;
//#pragma omp simd reduction(+:sum)
                    for (index col = n_dx_q_1; col < n; col++) {
                        sum += R_A[col + row_n] * z_hat[col];
                    }
                    y_b[row] = y_bar[row] - sum;
                }
//                cout << y_b;
                if (is_constrained)
                    search.ch(n_dx_q_0, n_dx_q_1, 1, R, y_b, z_hat);
//                    search.mch(n_dx_q_0, n_dx_q_1, R_A, y_b, z_hat,
//                        INFINITY, true, false, INFINITY);
                else
                    search.se(n_dx_q_0, n_dx_q_1, 1, R, y_b, z_hat);
            }
//            cout << z_hat;
            run_time = omp_get_wtime() - run_time;
            returnType < scalar, index > reT = {{}, run_time, 0};
            return reT;
        }


        returnType<scalar, index> pbocb(const index n_proc, const index nstep, const index init) {
            index ds = d.size();
            CILS_SECH_Search<scalar, index> search(m, n, qam, search_iter);
            scalar run_time2 = omp_get_wtime();
            if (ds == 1) {
                search.mch2(0, n, 0, 1, R_A, y_bar, z_hat);
                run_time2 = omp_get_wtime() - run_time2;
                return {{}, run_time2, 0};
            } else if (m == 44) {
                return bocb2();
            }

            index diff = 0, num_iter = 0, flag = 0, temp, R_S_1[ds] = {};
            index test, row_n, check = 0, r, _nswp = nstep, end = 0;
            index n_dx_q_2, n_dx_q_0;
            scalar sum = 0, start;
            scalar run_time = 0, run_time3 = 0;

            b_vector y_B(n);

#pragma omp parallel default(none) num_threads(n_proc)
            {}

            run_time2 = omp_get_wtime();
            n_dx_q_2 = d[0];
            n_dx_q_0 = d[1];

            if (is_constrained)
                search.mch2(n_dx_q_0, n_dx_q_2, 0, 1, R_A, y_bar, z_hat);
            else
                search.se(n_dx_q_0, n_dx_q_2, 1, R, y_bar, z_hat);


            R_S_1[0] = 1;
            end = 1;
#pragma omp parallel default(shared) num_threads(n_proc) private(n_dx_q_2, n_dx_q_0, sum, temp, check, test, row_n) firstprivate(y_B)
            {
                for (index j = 0; j < _nswp && !flag; j++) {
//                omp_set_lock(&lock[j - 1]);
//                omp_unset_lock(&lock[j - 1]);
#pragma omp for schedule(runtime) nowait
                    for (index i = 1; i < ds; i++) {
                        if (!flag && end <= i) {//  front >= i   &&
                            n_dx_q_2 = d[i];
                            n_dx_q_0 = i == ds - 1 ? 0 : d[i + 1];
                            check = i == end;
                            row_n = ((n_dx_q_0 - 1) * (2 * n - n_dx_q_0)) / 2;
                            for (index row = n_dx_q_0; row < n_dx_q_2; row++) {
                                sum = 0;
                                row_n += n - row;
#pragma omp simd reduction(+:sum)
                                for (index col = n_dx_q_2; col < n; col++) {
                                    sum += R_A[col + row_n] * z_hat[col];
                                }
                                y_B[row] = y_bar[row] - sum;
                            }
//                        test = 0;
//                        for (index row = 0; row < i; row++){
//                            test += R_S_1[i];
//                        }
//                        omp_set_lock(&lock[i - 1]);

//                        check = check || R_S_1[i - 1];
                            if (is_constrained) {
                                R_S_1[i] = search.mch2(n_dx_q_0, n_dx_q_2, i, 0, R_A, y_B, z_hat);
//                                R_S_1[i] = search.ch(n_dx_q_0, n_dx_q_2, check, R, y_B, z_hat);
                            }
//                            else
//                                R_S_1[i] = search.mse(n_dx_q_0, n_dx_q_2, R_A, y_B, z_hat, check ? INFINITY : 1, true,
//                                                      false, INFINITY);
//                            omp_unset_lock(&lock[j]);
                            if (check) { //!R_S_1[i] &&
                                end = i + 1;
                                R_S_1[i] = 1;
                            }
#pragma omp atomic
                            diff += R_S_1[i];
//                            if (mode != 0) {
                            flag = ((diff) >= ds - offset) && j > 0;
//                            }
                        }
                        num_iter = j;

                    }
                }

#pragma omp single
                {
                    run_time3 = omp_get_wtime() - run_time2;
                }
//#pragma omp flush
            }
            run_time2 = omp_get_wtime() - run_time2;
#pragma parallel omp cancellation point

            returnType < scalar, index > reT;


            scalar time = 0; //(run_time3 + run_time2) * 0.5;
            if (init == -1) {
                time = qam == 1 ? run_time2 + run_time : run_time2 + run_time;
            } else {
                time = qam == 1 ? run_time2 : run_time2 * 0.5 + run_time3 * 0.5;
            }
//            if (mode == 0)
//                reT = {{run_time3}, time, (scalar) diff + end};
//            else {
            return {{run_time3}, time, (scalar) num_iter + 1};
//            cout << "n_proc:" << n_proc << "," << "init:" << init << "," << diff << "," << end << ",Ratio:"
//                 << (index) (run_time2 / run_time3) << "," << run_time << "||";
//            cout.flush();
//            }
//            for (index i = 0; i < ds; i++) {
//                omp_destroy_lock(&lock[i]);
//            }
        }

        returnType<scalar, index> pbocb_test(const index n_t, const index nstep, const index init, const index T_) {

            index ds = d.size();
            index diff = 0, num_iter = 0, flag = 0, temp;
            b_vector ct(nstep, 0), df(nstep, 0), delta(ds, 0);
            index row_n, check = 0, r, idx = 0, s = 0;
            index n_dx_q_1, n_dx_q_0, case2 = 0;
            scalar sum = 0, start;
            scalar run_time = 0, run_time3 = 0;

            cils::searchType<scalar, index> reT;
            n_dx_q_1 = d[0];
            n_dx_q_0 = d[1];
            b_vector y_B(n), y_old(n), T(ds, 1), beta(n, 0);
            y_B.clear();
            y_old.clear();

            CILS_SECH_Search<scalar, index> search(m, n, qam, search_iter);


#pragma omp parallel default(none) num_threads(n_t)
            {}

            scalar run_time2 = omp_get_wtime();
            if (is_constrained)
                search.mch(n_dx_q_0, n_dx_q_1, R_A, y_bar, z_hat, (int) INFINITY, true, false, INFINITY);
            else
                search.mse(n_dx_q_0, n_dx_q_1, R_A, y_bar, z_hat, (int) INFINITY, true, false, INFINITY);

            delta[0] = 0;
            idx = 1;

#pragma omp parallel default(shared) num_threads(n_t) private(n_dx_q_1, n_dx_q_0, sum, temp, check, row_n, case2, reT)
            {
                for (index t = 0; t < nstep && !flag; t++) {
#pragma omp for schedule(dynamic) nowait
                    for (index i = 1; i < ds; i++) {
                        if (!flag && idx <= i) {
                            n_dx_q_1 = d[i];
                            n_dx_q_0 = i == ds - 1 ? 0 : d[i + 1];
                            check = i == idx;
                            case2 = true;
                            row_n = (n_dx_q_0 - 1) * (n - n_dx_q_0 / 2);
                            for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                                sum = 0;
                                row_n += n - row;
#pragma omp simd reduction(+:sum)
                                for (index col = n_dx_q_1; col < n; col++) {
                                    sum += R_A[col + row_n] * z_hat[col];
                                }
                                y_B[row] = y_bar[row] - sum;
                                case2 = case2 && y_old[row] == y_B[row];
                                y_old[row] = y_B[row];
                            }

                            if (check) {
//                                    search.se(n_dx_q_0, n_dx_q_1, 1, R, y_B, z_hat);
//                                    search.mse(n_dx_q_0, n_dx_q_1, i, check, R_A, y_B, z_hat);
                                if (is_constrained)
                                    search.mch(n_dx_q_0, n_dx_q_1, R_A, y_B, z_hat, (int) INFINITY,
                                               true, false, INFINITY);
                                else
                                    search.mse(n_dx_q_0, n_dx_q_1, R_A, y_B, z_hat, (int) INFINITY,
                                               true, false, INFINITY);

                                idx = i + 1;
                                delta[i] = 0;
                            } else {
                                if (case2 || T[i] != 0) {
                                    if (is_constrained)
                                        reT = search.mch(n_dx_q_0, n_dx_q_1, R_A, y_B, z_hat, T_, t == 0, case2,
                                                         beta[i]);
                                    else
                                        reT = search.mse(n_dx_q_0, n_dx_q_1, R_A, y_B, z_hat, T_, t == 0, case2,
                                                         beta[i]);
                                    beta[i] = reT.beta;
                                    delta[i] = reT.diff;
                                    T[i] = reT.T_r;
                                } else {
                                    delta[i] = 0;
                                }
                            }

                        }
#pragma omp atomic
                        s++;
                    }

                    if (!flag && t > 1) {
                        num_iter = t;
#pragma omp for nowait
                        for (index i = 1; i < ds; i++) {
#pragma omp atomic
                            diff += delta[i];
#pragma omp atomic
                            ct[t]++;
                        }
                        if (ct[t] >= ds - 1) {
                            flag = diff <= 3 || idx <= n_t;
#pragma omp atomic write
                            diff = 0;
                        }
                    } else {
                        flag = true;
                    }


                }

#pragma omp single
                {
                    run_time3 = omp_get_wtime() - run_time2;
                }
            }
            run_time2 = omp_get_wtime() - run_time2;
            scalar time = 0;
            if (init == -1) {
                time = qam == 1 ? run_time2 + run_time : run_time2 + run_time;
            } else {
                time = qam == 1 ? run_time2 : run_time2 * 0.5 + run_time3 * 0.5;
            }
//            cout << delta;
//            cout << ct;
            return {{run_time3}, time, (scalar) num_iter + 1};
        }


    };
}
/*
 *
        returnType<scalar, index> bocb_CPU() {
            index ds = d.size(), n_dx_q_0, n_dx_q_1;
            b_vector y_b(n);

            sd_vector time(2 * ds, 0);
            CILS_SECH_Search<scalar, index> search(m, n, qam, search_iter);
            //special cases:
            if (ds == 1) {
                if (d[0] == 1) {
                    z_hat[0] = round(y_bar[0] / R(0, 0));
                    return {{}, 0, 0};
                } else {
                    for (index i = 0; i < n; i++) {
                        y_b[i] = y_bar[i];
                    }
//                if (is_constrained)
//                    ils_search_obils(0, n, y_b, z_hat);
//                else
//                    se(0, n, y_b, z_hat);
                    return {{}, 0, 0};
                }
            }
            scalar start = omp_get_wtime();
            for (index i = 0; i < ds; i++) {
                n_dx_q_1 = d[i];
                n_dx_q_0 = i == ds - 1 ? 0 : d[i + 1];

                for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                    scalar sum = 0;
                    for (index col = n_dx_q_1; col < n; col++) {
                        sum += R(row, col) * z_hat[col];
                    }
                    y_b[row] = y_bar[row] - sum;
                }
            }
//        std::random_shuffle(r_d.begin(), r_d.end());


            for (index i = 0; i < ds; i++) {
                n_dx_q_1 = d[i];
                n_dx_q_0 = i == ds - 1 ? 0 : d[i + 1];
                index time_index = i;

                time[time_index] = omp_get_wtime();
                if (is_constrained)
                    time[time_index + ds] = search.ch(n_dx_q_0, n_dx_q_1, 1, R, y_b, z_hat);
                else
                    time[time_index + ds] = search.se(n_dx_q_0, n_dx_q_1, 1, R, y_b, z_hat);

                time[time_index] = omp_get_wtime() - time[time_index];
            }

            scalar run_time = omp_get_wtime() - start;


            //Matlab Partial Reduction needs to do the permutation
            returnType < scalar, index > reT = {time, run_time, 0};
            return reT;
        }

 */
//                scalar prod_time = omp_get_wtime();
//                prod_time = omp_get_wtime() - prod_time;
//                scalar prod_time2 = omp_get_wtime();
//                for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
//                    sum = 0;
//                    for (index col = n_dx_q_1; col < n; col++) {
//                        sum += R(row, col) * z_hat[col];
//                    }
//                    y_b[row] = y_bar[row] - sum;
//                }
//                prod_time2 = omp_get_wtime() - prod_time2;
//                cout << "Ratio:" << prod_time / prod_time2 <<" ";


//                                cout << sum << endl;
//                                sum = y_bar[ni];
//                                index nj = ni * n - (ni * (ni + 1)) / 2;
//#pragma omp simd reduction(- : sum)
//                                for (index col = n - 1; col >= n - i; col--){ //col < n - 1; col++) {
//                                    sum -= R_A[nj + col] * z_hat[col];
//                                }
//                                cout << sum << endl;
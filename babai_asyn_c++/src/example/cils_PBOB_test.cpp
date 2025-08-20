#include "../source/CILS.cpp"
#include "../source/CILS_Reduction.cpp"
#include "../source/CILS_SECH_Search.cpp"
#include "../source/CILS_OLM.cpp"

void init_z_hat(b_vector &z_hat, b_vector &x, int init, double mean) {
    z_hat.clear();
    if (init == 1) std::fill(z_hat.x.begin(), z_hat.x.end(), round(mean));
    if (init == 2) z_hat.assign(x);
}

template<typename scalar, typename index>
long test_PBNP(int size_n, bool is_local) {

    time_t t0 = time(nullptr);
    struct tm *lt = localtime(&t0);
    char time_str[20];
    sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
            lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
            lt->tm_hour, lt->tm_min, lt->tm_sec
    );
    printf("====================[ TEST | BNP | %s ]==================================\n", time_str);
    cout.flush();

    index d = 0, l = 0, num_trial = 200, k, c = 0, n = size_n;
    scalar t_qr[6][200][6][2] = {}, t_aspl[6][200][6][2] = {}, t_itr[6][200][6][2] = {};
    scalar t_bnp[6][200][6][2], t_ber[6][200][6][2] = {}, run_time;

    cils::CILS<scalar, index> cils;
    cils::CILS_Reduction<scalar, index> reduction(cils), reduction2(cils);

    cils.is_local = is_local;
    b_vector x_ser, x_lll, x_r;
    for (int t = 0; t < num_trial; t++) {
        run_time = omp_get_wtime();
        index s = 0;
        for (int snr = 0; snr <= 50; snr += 10) {
            k = 0;
            for (int qam = 1; qam <= 3; qam += 2) {
                x_ser.resize(n, false);
                x_lll.resize(n, false);

                printf("--------ITER: %d, SNR: %d, QAM: %d, n: %d-------\n", t + 1, snr, (int) pow(4, qam), n);
                cils::init_PBNP(cils, n, snr, qam, c);
                cils::returnType<scalar, index> reT;
                reduction.reset(cils);
                reT = reduction.aspl();
                t_qr[s][t][0][k] = reT.run_time;
                t_aspl[s][t][0][k] = reT.info;
                printf("ASPL: QR: %8.4f, LLL: %8.4f, TOTAL:%8.4f\n",
                       reT.run_time, reT.info, reT.info + reT.run_time);

                reduction.reset(cils);
                reT = reduction.plll();
                t_qr[s][t][1][k] = reT.run_time;
                t_aspl[s][t][1][k] = reT.info;
                printf("PLLL: QR: %8.4f, LLL: %8.4f, TOTAL:%8.4f\n",
                       reT.run_time, reT.info, reT.info + reT.run_time);

                cils::CILS_OLM<scalar, index> olm(cils, x_ser, reduction.R, reduction.y);

                l = 1;
                for (index n_proc = 5; n_proc <= 25; n_proc += 5) {
                    l++;
                    reduction2.reset(cils);
                    reT = reduction2.paspl(n_proc);
                    t_qr[s][t][l][k] = reT.run_time;
                    t_aspl[s][t][l][k] = reT.info;
                    printf("PASPL: CORE: %3d, QR: %8.4f, LLL: %8.4f, TOTAL:%8.4f, "
                           "SPUQR: %8.4f, SPUASPL: %8.4f, SPUPLLL: %8.4f, SPUTOTAL:%8.4f\n",
                           n_proc, reT.run_time, reT.info, reT.info + reT.run_time,
                           t_qr[s][t][0][k] / reT.run_time, t_aspl[s][t][0][k] / reT.info,
                           t_aspl[s][t][1][k] / reT.info,
                           (t_qr[s][t][0][k] + t_aspl[s][t][0][k]) / (reT.run_time + reT.info)
                    );
                }

                scalar r = helper::find_residual<scalar, index>(cils.A, cils.x_t, cils.y);
                init_z_hat(olm.z_hat, x_r, 1, (scalar) cils.upper / 2.0);

                reT = olm.bnp();
                projection(reduction.Z, olm.z_hat, x_lll, 0, cils.upper);
                t_ber[s][t][0][k] = helper::find_bit_error_rate<scalar, index>(x_lll, cils.x_t, cils.qam);
                t_bnp[s][t][0][k] = reT.run_time;
                scalar res = helper::find_residual<scalar, index>(cils.A, x_lll, cils.y);
                printf("BNP: BER: %8.5f, RES: %8.4f, TIME: %8.4f\n", t_ber[s][t][0][k], res,
                       t_bnp[s][t][0][k]);

                l = 0;
                scalar total = t_bnp[s][t][0][k] + t_qr[s][t][0][k] + t_aspl[s][t][0][k];
                for (index n_proc = 5; n_proc <= 25; n_proc += 5) {
                    l++;
                    init_z_hat(olm.z_hat, x_r, 1, (int) cils.upper / 2);
                    reT = olm.pbnp2(n_proc, 10, 1);
                    projection(reduction.Z, olm.z_hat, x_lll, 0, cils.upper);
                    t_ber[s][t][l][k] = helper::find_bit_error_rate<scalar, index>(x_lll, cils.x_t, cils.qam);
                    t_bnp[s][t][l][k] = reT.run_time;
                    t_itr[s][t][l][k] = reT.info;
                    res = helper::find_residual<scalar, index>(cils.A, x_lll, cils.y);
                    printf("PBNP: CORE: %3d, ITER: %4d, BER: %8.5f, RES: %8.4f, TIME: %8.4f, "
                           "BNP SPU: %8.4f, TOTAL SPU: %8.4f\n",
                           n_proc, (int) reT.info, t_ber[s][t][l][k], res,
                           t_bnp[s][t][l][k],
                           t_bnp[s][t][0][k] / t_bnp[s][t][l][k],
                           total / (t_bnp[s][t][l][k] + t_qr[s][t][l][k] + t_aspl[s][t][l][k]));
                }
                k++;
            }
            s++;
        }
        run_time = omp_get_wtime() - run_time;
        printf("++++++++++++++++++++++++++++++++++++++\n Trial %d, Elapsed Time: %.5fs. \n"
               "++++++++++++++++++++++++++++++++++++++\n", t, run_time);
        cout.flush();
        printf("\n---------------------\nITER:%d\n---------------------\n", t);
    }

    printf("End of current TASK.\n");
    printf("-------------------------------------------\n");

    return 0;

}

template<typename scalar, typename index>
long test_PBOB(int n, int nob, int c, int T_, bool is_local) {

    time_t t0 = time(nullptr);
    struct tm *lt = localtime(&t0);
    char time_str[20];
    sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
            lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
            lt->tm_hour, lt->tm_min, lt->tm_sec
    );
    printf("====================[ TEST | BOB | %s ]==================================\n", time_str);
    cout.flush();

    index d = 0, l = 0, num_trial = 200, constrain = c != 0, qam = 3;
    scalar t_qr[4][200][10][3] = {}, t_aspl[4][200][10][3] = {}, t_bnp[4][200][10][3] = {}, t_ber[4][200][10][3] = {}, run_time;
    index SNRs[4] = {10, 20, 30, 40};

    cils::CILS<scalar, index> cils;
    cils::CILS_Reduction<scalar, index> reduction(cils), reduction2(cils);
    cils::returnType<scalar, index> reT;
    cils.is_local = is_local;
    cils.is_constrained = constrain;
    cils.block_size = n / nob;
    cils.spilt_size = 0;
//    if(cils.block_size == 20)
//        cils.spilt_size = 2;

    b_vector x_ser, x_lll, x_r;

    for (int t = 0; t < num_trial; t++) {
        for (int s = 0; s < 4; s++) {
            run_time = omp_get_wtime();
            int k = 0;
            cils::init_PBNP(cils, n, SNRs[s], qam, c);
            cils.block_size = n / nob;
            cils.spilt_size = 0;
            cils.init_d();

            x_ser.resize(n, false);
            x_lll.resize(n, false);
            x_r.resize(n, false);

            printf("--------ITER: %d, SNR: %d, QAM: %d, SIZE: %d-------\n", t + 1, SNRs[s], qam, n);

            reduction.reset(cils);
            if (constrain)
                reT = reduction.aspl_p();
            else
                reT = reduction.aspl();
            t_qr[s][t][0][k] = reT.run_time;
            t_aspl[s][t][0][k] = reT.info;
            if (constrain) {
                reduction.reset(cils);
                reduction.mgs_qr();
            }

            if (constrain)
                printf("ASPL-P: QR: %8.4f, LLL: %8.4f, TOTAL:%8.4f\n",
                       reT.run_time, reT.info, reT.info + reT.run_time);
            else
                printf("ASPL: QR: %8.4f, LLL: %8.4f, TOTAL:%8.4f\n",
                       reT.run_time, reT.info, reT.info + reT.run_time);

            cils::CILS_OLM<scalar, index> olm(cils, x_ser, reduction.R, reduction.y);

            l = 0;
            for (index n_proc = 5; n_proc <= 15; n_proc += 5) {
                l++;
                reduction2.reset(cils);
                if (constrain)
                    reT = reduction2.paspl_p(n_proc == 0 ? 1 : n_proc);
                else
                    reT = reduction2.paspl(n_proc);

                t_qr[s][t][l][k] = reT.run_time;
                t_aspl[s][t][l][k] = reT.info;
                if (constrain)
                    printf("PASPL-P: CORE: %3d, QR: %8.4f, LLL: %8.4f, TOTAL:%8.4f, "
                           "SPUQR: %8.4f, SPUASPL: %8.4f, 3, SPUTOTAL:%8.4f\n",
                           n_proc, reT.run_time, reT.info, reT.info + reT.run_time,
                           t_qr[s][t][0][k] / reT.run_time, t_aspl[s][t][0][k] / reT.info,
                           t_aspl[s][t][1][k] / reT.info,
                           (t_qr[s][t][0][k] + t_aspl[s][t][0][k]) / (reT.run_time + reT.info));
                else
                    printf("PASPL: CORE: %3d, QR: %8.4f, LLL: %8.4f, TOTAL:%8.4f, "
                           "SPUQR: %8.4f, SPUASPL: %8.4f, SPUPLLL: %8.4f, SPUTOTAL:%8.4f\n",
                           n_proc, reT.run_time, reT.info, reT.info + reT.run_time,
                           t_qr[s][t][0][k] / reT.run_time, t_aspl[s][t][0][k] / reT.info,
                           t_aspl[s][t][1][k] / reT.info,
                           (t_qr[s][t][0][k] + t_aspl[s][t][0][k]) / (reT.run_time + reT.info));
            }
            cout.flush();

            olm.z_hat.assign(0);
            reT = olm.bnp();
            for (int i = 0; i < n; i++) {
                x_ser[i] = olm.z_hat[i];
            }
            projection(reduction.Z, olm.z_hat, x_r, 0, cils.upper);

            scalar ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_r, qam);
            t_bnp[s][t][0][0] = reT.run_time;
            t_bnp[s][t][0][1] = reT.run_time;
            t_bnp[s][t][0][2] = reT.run_time;
            t_ber[s][t][0][0] = ber;
            t_ber[s][t][0][1] = ber;
            t_ber[s][t][0][2] = ber;
            printf("BNP: BER: %8.4f, TIME: %8.4f\n", ber, t_bnp[s][t][0][k]);
            cout.flush();

            cout << "------------------block size 10, T = 1---------\n";
            reT = olm.bocb();
            projection(reduction.Z, olm.z_hat, x_lll, 0, cils.upper);
            ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_lll, qam);
            t_bnp[s][t][1][k] = reT.run_time;
            t_bnp[s][t][1][1] = reT.run_time;
            t_ber[s][t][1][k] = ber;
            t_ber[s][t][1][1] = ber;
            printf("BOB: BER: %8.4f, TIME: %8.4f\n", ber, t_bnp[s][t][1][k]);
            cout.flush();

            l = 1;
            scalar total = t_bnp[s][t][1][k] + t_qr[s][t][0][k] + t_aspl[s][t][0][k];
            for (index n_proc = 3; n_proc <= 9; n_proc += 3) {
                l++;
                for (int i = 0; i < n; i++) {
                    olm.z_hat[i] = x_ser[i];
                }
//                init_z_hat(olm.z_hat, x_r, 1, (int) cils.upper / 2);
                reT = olm.pbocb_test(n_proc, 10, 0, 1);
                projection(reduction.Z, olm.z_hat, x_lll, 0, cils.upper);
                ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_lll, qam);
                t_bnp[s][t][l][k] = reT.run_time;
                t_ber[s][t][l][k] = ber;
                printf("PBOB: CORE: %3d, ITER: %4d, BER: %8.4f, TIME: %8.4f, "
                       "BOB SPU: %8.4f, TOTAL SPU: %8.4f\n",
                       n_proc, (int) reT.info, ber,
                       t_bnp[s][t][l][k], t_bnp[s][t][1][k] / t_bnp[s][t][l][k],
                       total / (t_bnp[s][t][l][k] + t_qr[s][t][l - 1][k] + t_aspl[s][t][l - 1][k]));
            }


            cout << "------------------block size 10 T=" << T_ << "---------\n";
            k = l = 1;
            total = t_bnp[s][t][1][0] + t_qr[s][t][0][0] + t_aspl[s][t][0][0];

            for (index n_proc = 3; n_proc <= 9; n_proc += 3) {
                l++;
                for (int i = 0; i < n; i++) {
                    olm.z_hat[i] = x_ser[i];
                }
                reT = olm.pbocb_test(n_proc, 10, 0, T_);
                projection(reduction.Z, olm.z_hat, x_lll, 0, cils.upper);
                ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_lll, qam);
                t_bnp[s][t][l][k] = reT.run_time;
                t_ber[s][t][l][k] = ber;
                printf("PBOB: CORE: %3d, ITER: %4d, BER: %8.4f, TIME: %8.4f, "
                       "BOB SPU: %8.4f, TOTAL SPU: %8.4f\n",
                       n_proc, (int) reT.info, ber,
                       t_bnp[s][t][l][k], t_bnp[s][t][1][k] / t_bnp[s][t][l][k],
                       total / (t_bnp[s][t][l][k] + t_qr[s][t][l - 1][k] + t_aspl[s][t][l - 1][k]));
            }


            cout << "------------------block size 20 T=1---------\n";
            cils.block_size = n / 10;
            cils.spilt_size = 0;
            cils.init_d();
            cils::CILS_OLM<scalar, index> olm2(cils, x_ser, reduction.R, reduction.y);
            k = 2;
            init_z_hat(olm2.z_hat, x_r, 1, (scalar) cils.upper / 2.0);

            reT = olm2.bocb();
            projection(reduction.Z, olm2.z_hat, x_lll, 0, cils.upper);
            ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_lll, qam);
            t_bnp[s][t][1][k] = reT.run_time;
            t_ber[s][t][1][k] = ber;
            printf("BOB: BER: %8.4f, TIME: %8.4f\n", ber, t_bnp[s][t][1][k]);
            cout.flush();

            l = 1;
            total = t_bnp[s][t][1][1] + t_qr[s][t][0][0] + t_aspl[s][t][0][0];
            for (index n_proc = 3; n_proc <= 9; n_proc += 3) {
                l++;
                for (int i = 0; i < n; i++) {
                    olm.z_hat[i] = x_ser[i];
                }
                reT = olm2.pbocb_test(n_proc, 10, 0, 1);
                projection(reduction.Z, olm2.z_hat, x_lll, 0, cils.upper);
                ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_lll, qam);
                t_bnp[s][t][l][k] = reT.run_time;
                t_ber[s][t][l][k] = ber;
                printf("PBOB: CORE: %3d, ITER: %4d, BER: %8.4f, TIME: %8.4f, "
                       "BOB SPU: %8.4f, TOTAL SPU: %8.4f\n",
                       n_proc, (int) reT.info, ber,
                       t_bnp[s][t][l][k], t_bnp[s][t][1][k] / t_bnp[s][t][l][k],
                       total / (t_bnp[s][t][l][k] + t_qr[s][t][l - 1][0] + t_aspl[s][t][l - 1][0]));
            }
        }

        run_time = omp_get_wtime() - run_time;
        printf("++++++++++++++++++++++++++++++++++++++\n Trial %d, Elapsed Time: %.5fs. \n"
               "++++++++++++++++++++++++++++++++++++++\n", t, run_time);
        cout.flush();
        printf("\n---------------------\nITER:%d\n---------------------\n", t);
    }

    printf("End of current TASK.\n");
    printf("-------------------------------------------\n");

    return 0;

}

template<typename scalar, typename index>
long test_CH(int n, int nob, int c, bool is_local) {

    time_t t0 = time(nullptr);
    struct tm *lt = localtime(&t0);
    char time_str[20];
    sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
            lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
            lt->tm_hour, lt->tm_min, lt->tm_sec
    );
    printf("====================[ TEST | BOB | %s ]==================================\n", time_str);
    cout.flush();

    index d = 0, l = 0, num_trial = 200, constrain = c != 0, qam = 3;
    cils::CILS<scalar, index> cils;
    cils::returnType<scalar, index> reT;

    cils.is_local = is_local;
    cils.is_constrained = constrain;
//    cils.block_size = n / nob;
//    cils.spilt_size = 2;

    b_vector x_ser, x_lll, x_r;
    cils::init_PBNP(cils, n, 0, qam, 1);

    x_ser.resize(n, false);
    x_lll.resize(n, false);

    printf("--------ITER: %d, SNR: %d, QAM: %d, SIZE: %d-------\n", 0, 0, qam, n);

    cils::CILS_SECH_Search<scalar, index> search(n, n, qam, 1e6);
    cils::CILS_OLM<scalar, index> olm(cils, x_ser, cils.B, cils.y);
    cout << cils.y;
    init_z_hat(x_ser, x_r, 1, (scalar) cils.upper / 2.0);
    search.ch(0, n, 1, cils.B, cils.y, x_ser);
    projection(cils.A, x_ser, x_lll, 0, cils.upper);
    scalar ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_lll, qam);
    cout << x_lll;
    cout << ber;
    //    printf("CH: BER: %8.4f, TIME: %8.4f\n", ber, reT.run_time);
    cout.flush();

    init_z_hat(x_ser, x_r, 1, (scalar) cils.upper / 2.0);
    search.mch(0, n, olm.R_A, cils.y, x_ser, (int) INFINITY, true, false, INFINITY);
    projection(cils.A, x_ser, x_lll, 0, cils.upper);
    ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_lll, qam);
    cout << x_lll;
    cout << cils.x_t;
    cout << ber;
//    printf("MCH: BER: %8.4f, TIME: %8.4f\n", ber, reT.run_time);
    cout.flush();

    printf("End of current TASK.\n");
    printf("-------------------------------------------\n");

    return 0;

}

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
long test_PBOB(int s, bool is_local) {

    time_t t0 = time(nullptr);
    struct tm *lt = localtime(&t0);
    char time_str[20];
    sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
            lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
            lt->tm_hour, lt->tm_min, lt->tm_sec
    );
    printf("====================[ TEST | BNP | %s ]==================================\n", time_str);
    cout.flush();

    index d = 0, l = 0, num_trial = 200, k, c = 1;
    scalar t_qr[6][200][6][2] = {}, t_aspl[6][200][6][2] = {}, t_bnp[6][200][6][2], run_time;

    cils::CILS<scalar, index> cils;
    cils::CILS_Reduction<scalar, index> reduction(cils), reduction2(cils);

    cils.is_local = is_local;
    index sizes[5] = {50, 100, 200, 300, 500};

    b_vector x_ser, x_lll, x_r;
    for (int t = 0; t < num_trial; t++) {

        run_time = omp_get_wtime();
        index n = sizes[s];
        cils::init_PBNP(cils, n, 30, 3, 0);
        cils::returnType<scalar, index> reT;
        for (k = 1; k <= 1; k++) {
            cils.is_constrained = 0;
            x_ser.resize(n, false);
            x_lll.resize(n, false);

            printf("--------ITER: %d, SNR: %d, constrain: %d, n: %d-------\n", t + 1, 30, k, n);

            reduction.reset(cils);
            reT = reduction.aspl();
            t_qr[s][t][0][k] = reT.run_time;
            t_aspl[s][t][0][k] = reT.info;
            if (k)
                printf("ASPL: QR: %8.4f, LLL: %8.4f, TOTAL:%8.4f\n",
                       reT.run_time, reT.info, reT.info + reT.run_time);
            else
                printf("ASPL-P: QR: %8.4f, LLL: %8.4f, TOTAL:%8.4f\n",
                       reT.run_time, reT.info, reT.info + reT.run_time);
            cils::CILS_OLM<scalar, index> olm(cils, x_ser, reduction.R, reduction.y);

            l = 0;
            for (index n_proc = 5; n_proc <= 25; n_proc += 5) {
                l++;
                reduction2.reset(cils);
                if (k)
                    reT = reduction2.paspl_p(n_proc);
                else
                    reT = reduction2.paspl(n_proc);

                t_qr[s][t][l][k] = reT.run_time;
                t_aspl[s][t][l][k] = reT.info;
                if (k)
                    printf("PASPL: CORE: %3d, QR: %8.4f, LLL: %8.4f, TOTAL:%8.4f, "
                           "SPUQR: %8.4f, SPUASPL: %8.4f, 3, SPUTOTAL:%8.4f\n",
                           n_proc, reT.run_time, reT.info, reT.info + reT.run_time,
                           t_qr[s][t][0][k] / reT.run_time, t_aspl[s][t][0][k] / reT.info,
                           t_aspl[s][t][1][k] / reT.info,
                           (t_qr[s][t][0][k] + t_aspl[s][t][0][k]) / (reT.run_time + reT.info));
                else
                    printf("PASPL-P: CORE: %3d, QR: %8.4f, LLL: %8.4f, TOTAL:%8.4f, "
                           "SPUQR: %8.4f, SPUASPL: %8.4f, SPUPLLL: %8.4f, SPUTOTAL:%8.4f\n",
                           n_proc, reT.run_time, reT.info, reT.info + reT.run_time,
                           t_qr[s][t][0][k] / reT.run_time, t_aspl[s][t][0][k] / reT.info,
                           t_aspl[s][t][1][k] / reT.info,
                           (t_qr[s][t][0][k] + t_aspl[s][t][0][k]) / (reT.run_time + reT.info));
            }

            scalar r = helper::find_residual<scalar, index>(cils.A, cils.x_t, cils.y);
            init_z_hat(olm.z_hat, x_r, 1, (scalar) cils.upper / 2.0);

            reT = olm.bnp();
            projection(reduction.Z, olm.z_hat, x_lll, 0, cils.upper);
            t_bnp[s][t][0][k] = reT.run_time;
            scalar res = helper::find_residual<scalar, index>(cils.A, x_lll, cils.y);
            printf("BNP: RES: %8.4f, TIME: %8.4f\n", res, t_bnp[s][t][0][k]);

            l = 0;
            scalar total = t_bnp[s][t][0][k] + t_qr[s][t][0][k] + t_aspl[s][t][0][k];
            for (index n_proc = 5; n_proc <= 25; n_proc += 5) {
                l++;
                init_z_hat(olm.z_hat, x_r, 1, (int) cils.upper / 2);
                reT = olm.pbnp2(n_proc, 10, 1);
                projection(reduction.Z, olm.z_hat, x_lll, 0, cils.upper);
                t_bnp[s][t][l][k] = reT.run_time;
                res = helper::find_residual<scalar, index>(cils.A, x_lll, cils.y);
                printf("PBNP: CORE: %3d, ITER: %4d, RES: %8.4f, TIME: %8.4f, "
                       "BNP SPU: %8.4f, TOTAL SPU: %8.4f\n",
                       n_proc, (int) reT.info, res,
                       t_bnp[s][t][l][k], t_bnp[s][t][0][k] / t_bnp[s][t][l][k],
                       total / (t_bnp[s][t][l][k] + t_qr[s][t][l][k] + t_aspl[s][t][l][k]));
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

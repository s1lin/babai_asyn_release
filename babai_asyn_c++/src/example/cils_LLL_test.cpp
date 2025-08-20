#include "../source/CILS.cpp"
#include "../source/CILS_Reduction.cpp"

template<typename scalar, typename index>
long plot_LLL() {

    time_t t0 = time(nullptr);
    struct tm *lt = localtime(&t0);
    char time_str[20];
    sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
            lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
            lt->tm_hour, lt->tm_min, lt->tm_sec
    );
    printf("====================[ TEST | LLL | %s ]==================================\n", time_str);
    cout.flush();

    index d = 0, l = 0, num_trial = 200;
    scalar t_qr[4][200][20][2] = {}, t_aspl[4][200][20][2] = {}, t_total[4][200][20][2] = {}, run_time;
    cils::CILS<scalar, index> cils;
    cils::CILS_Reduction<scalar, index> reduction(cils);
    cils.is_local = 1;

    for (int t = 0; t < num_trial; t++) {
        d = 0;
        run_time = omp_get_wtime();
        for (int n = 50; n <= 200; n += 50) {
            printf("+++++++++++ Dimension %d ++++++++++++++++++++\n", n);
            for (int k = 0; k <= 1; k++) {
                printf("+++++++++++ Case %d ++++++++++++++++++++\n", k + 1);
                l = 0;
                cils::init_LLL(cils, n, k);
                cout.flush();

                cils::returnType<scalar, index> reT;
                reduction.reset(cils);
                reT = reduction.plll();
                t_qr[d][t][0][k] = reT.run_time;
                t_aspl[d][t][0][k] = reT.info;
                t_total[d][t][0][k] = t_qr[d][t][0][k] + t_aspl[d][t][0][k];
                printf("PLLL: QR: %8.4f, LLL: %8.4f, TOTAL:%8.4f\n",
                       reT.run_time, reT.info, reT.info + reT.run_time);

                l++;
                reduction.reset(cils);
                reT = reduction.aspl();
                t_qr[d][t][l][k] = reT.run_time;
                t_aspl[d][t][l][k] = reT.info;
                t_total[d][t][l][k] = t_qr[d][t][l][k] + t_aspl[d][t][l][k];
                printf("ASPL: QR: %8.4f, LLL: %8.4f, TOTAL:%8.4f\n",
                       reT.run_time, reT.info, reT.info + reT.run_time);


                for (index n_proc = 5; n_proc <= 30; n_proc += 5) {
                    l++;
                    reduction.reset(cils);
                    index n_c = n_proc;
                    reT = reduction.paspl(n_c);
                    t_qr[d][t][l][k] = reT.run_time;
                    t_aspl[d][t][l][k] = reT.info;
                    t_total[d][t][l][k] = t_qr[d][t][l][k] + t_aspl[d][t][l][k];
                    printf("PASPL: CORE: %3d, QR: %8.4f, LLL: %8.4f, TOTAL:%8.4f, "
                           "SPUQR: %8.4f, SPULLL: %8.4f, SPUTOTAL:%8.4f,"
                           "SPUPL: %8.4f, SPUTOTAL2:%8.4f\n",
                           n_c, reT.run_time, reT.info, reT.info + reT.run_time,
                           t_qr[d][t][0][k] / reT.run_time, t_aspl[d][t][1][k] / reT.info,
                           t_total[d][t][1][k] / t_total[d][t][l][k],
                           t_aspl[d][t][0][k] / reT.info,
                           t_total[d][t][0][k] / t_total[d][t][l][k]
                    );
                }

            }
            d++;
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
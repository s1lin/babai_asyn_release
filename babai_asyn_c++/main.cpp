//#include "src/example/cils_LLL_test.cpp"
//#include "src/example/cils_PBNP_test.cpp"
#include "src/example/cils_PBOB_test.cpp"
// #include "src/example/cils_PBSIC_test.cpp"

using namespace std;
using namespace cils;

void functiona(int i, int n_threads) {
    cout << i << "," << omp_get_thread_num() << endl;
//#pragma omp parallel num_threads(n_threads)
//    for (int t = 0; t < 2; t++)
//#pragma omp for nowait
//            for (int j = 0; j < 4; j++)
//                printf("Task %d: thread %d of the %d children of %d: handling iter %d\n",
//                       i, omp_get_thread_num(), omp_get_team_size(2),
//                       omp_get_ancestor_thread_num(1), j);
}

int main(int argc, char *argv[]) {


    printf("\n====================[ Run | cils | Release ]==================================\n");
    double t = omp_get_wtime();

    int size_n = stoi(argv[1]);
    int is_local = stoi(argv[2]);
    int info = stoi(argv[3]);
    int sec = stoi(argv[4]);
    int blk = stoi(argv[5]);

   test_PBOB<double, int>(size_n, nob, c, T, is_local);


    t = omp_get_wtime() - t;

    printf("====================[TOTAL TIME | %2.2fs, %2.2fm, %2.2fh]==================================\n",
           t, t / 60, t / 3600);


    return 0;
}

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

long long  int count_pi (long long int toss, int rank){
    long long int cnt = 0;
    unsigned int local_seed = rank * time(NULL);
    double rand_max = (double)RAND_MAX + 1;

    for(int i=0; i<toss; i++){
        double x = rand_r(&local_seed) /  rand_max * 2.0 - 1.0;
        double y = rand_r(&local_seed) /  rand_max * 2.0 - 1.0;
        if((x*x + y*y) < 1){
            cnt++; 
        }
    }
    return cnt;
}



int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---
    long long int all_sum;

    // TODO: MPI init
    MPI_Comm_size( MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank( MPI_COMM_WORLD, &world_rank);

    if (world_rank > 0)
    {
        // TODO: MPI workers
        long long int cnt;
        cnt = count_pi(tosses/world_size, world_rank);
        MPI_Send(&cnt, 1, MPI_LONG_LONG, 0, world_rank, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        all_sum = count_pi(tosses/world_size, world_rank);

        MPI_Request requests[world_size-1];
        long long int cnts[world_size];
        for (int i=1; i<world_size; i++){
            MPI_Irecv(&cnts[i], 1, MPI_LONG_LONG, i, i, MPI_COMM_WORLD, &requests[i-1]);
        }
        MPI_Waitall(world_size-1, requests, MPI_STATUS_IGNORE);

        for(int i=1; i<world_size; i++){
            all_sum += cnts[i];
        }
    }

    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = (double) 4 * all_sum / tosses ;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}

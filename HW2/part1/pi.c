# include <stdio.h>
# include <stdlib.h>
# include <pthread.h>



pthread_mutex_t mutex_sum;

typedef struct{
  long long int toss;
  // double *circle_cnt;
  long long int  sum;
}Arg;



void *count_pi (void *arg){
  Arg *data = (Arg *)arg;

  long long int toss = data->toss;
  // double *circle_cnt = data->circle_cnt;
  

  unsigned int local_seed = 1;
  long int cnt = 0;
  
  double rand_max = (double)RAND_MAX + 1;

  for (long long int i=0; i<toss; i++){
    double x = rand_r(&local_seed) /  rand_max * 2.0 - 1.0;
    double y = rand_r(&local_seed) /  rand_max * 2.0 - 1.0;
    if((x*x + y*y) < 1){
      cnt++;
    }
  }
  // pthread_mutex_lock(&mutex_sum);
  // *circle_cnt += cnt;
  data->sum = cnt;
  // pthread_mutex_unlock(&mutex_sum);
  pthread_exit((void *)0);
}

int main (int argc, char *argv[]){
  int thread_num = atoi(argv[1]);
  long long int toss = atoll(argv[2]);
  

  pthread_t callThd[thread_num];
  

  pthread_mutex_init(&mutex_sum, NULL);
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  // double *circle_cnt = malloc(sizeof(*circle_cnt));
  // *circle_cnt = 0;
  long long int part_loss = toss/thread_num;
  Arg arg[thread_num];


  // clock_t start, end;
  // start = clock();

  for (int i=0; i<thread_num; i++){
    arg[i].toss = part_loss + i;
    // arg[i].circle_cnt = circle_cnt;
    arg[i].sum = 0;

    // if (i==0){
    //   arg[i].toss += (toss%thread_num);
    // }

    pthread_create(&callThd[i], &attr, count_pi, (void *)&arg[i]);
  }

  pthread_attr_destroy(&attr);
  long long int all_cnt = 0;
  void *status;
  for(int i=0; i<thread_num; i++){
    pthread_join(callThd[i], &status);
    all_cnt += arg[i].sum;
  }
  // end = clock();
  // double diff = end - start; 
  // printf(" %f  ms" , diff);
  // printf(" %f  sec", diff / CLOCKS_PER_SEC );
 
  // printf("%f\n", (double) 4 * (*circle_cnt) / toss);
  printf("%f\n", (double) 4 * all_cnt / toss);

  pthread_mutex_destroy(&mutex_sum);
  pthread_exit(NULL);

  return 0;
}

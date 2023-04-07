#include "MonteCarlo.hpp"
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudrand_kernel.h>

int round_cnt = 0;
std::vector<double> time_vec;
// vector<int> cuda_scores;
// vector<int> cuda_move;
#define N 32
#define BLOCK_SIZE 16


/**
 * From a given game state, chooses random moves until the game is completed.
 * @return (int FIRST_MOVE, int FINAL_SCORE)
*/
__global__ void simulateOneRun(Game game,  int * cuda_scores, int * cuda_move) {

    int first_move = -1;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = j * N + i;

    while(game.canContinue()) {
        board before_state = game.state;
        std::vector<int> move_bank = {UP, DOWN, LEFT, RIGHT};

        while (before_state == game.state) {

            // int random_pos = DISTS[move_bank.size() - 1](rng);
            // int chosen_move = move_bank[random_pos];

            switch(chosen_move) {
                case UP:
                    game.up(false);
                    break;
                case DOWN:
                    game.down(false);
                    break;
                case LEFT:
                    game.left(false);
                    break;
                case RIGHT:
                    game.right(false);
                    break;
            }
            if (first_move == -1) {
                first_move = chosen_move;
            }
            move_bank.erase(move_bank.begin() + random_pos);
        }
    }

    // std::pair<int, int> ret(first_move, game.score);
    cuda_scores[idx] = game.score;
    cuda_move[idx] = first_move; 
    // return ret;
}

/**
 * Creates a new game and then completes the game from its current state with 
 * completely random moves until RUNS-many completions. The looks at the scores
 * from those random runs to decide the best move for the current state and 
 * executes that move. Continues this process until game completion.
 * @return (int HIGHEST_TILE, int FINAL_SCORE)
 */
std::pair<int, int> monteCarloSimulateGame(int runs, int display_level, Game game) {
    std::cout << "Attempting to solve a new game with Monte Carlo... " << std::flush;
    // cuda_scores.resize(runs);
    // cuda_move.resize(runs);

    while (game.canContinue()) {
        
        if (display_level >= 2) {
            std::cout << std::endl << game;
        }
        int scores[4] = {0, 0, 0, 0};
        int counter[4] = {0, 0, 0, 0};

        int *host_move, *host_scores, *cuda_move, *cuda_scores;

        host_scores = (int *)malloc( N * N * sizeof(int));
        host_move = (int *)malloc( N * N * sizeof(int));

        cudaMalloc((void **)&cuda_scores, N * N * sizeof(int));
        cudaMalloc((void **)&cuda_move, N * N * sizeof(int));

        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 numBlock(N / BLOCK_SIZE, N / BLOCK_SIZE);

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        // for (int i = 0; i < runs; ++i) {
        //     std::pair<int, int> res = simulateOneRun(game);
        //     scores[res.first] += res.second;
        //     counter[res.first] += 1;
        // }
        simulateOneRun<<<numBlock, blockSize>>>(game, cuda_scores, cuda_move);
        cudaDeviceSynchronize();

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> time_span = t2 - t1;
        // std::cout<<"time measure in one roune: "<<time_span.count()<<"us"<<"\n";
        time_vec.emplace_back(time_span.count());
        
        cudaMemcpy(host_move, cuda_move, N * N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_scores, cuda_scores, N * N * sizeof(int), cudaMemcpyDeviceToHost);

        for(int i=0; i<runs; i++){
            if(host_move[i] == 0){
                scores[0]+=host_scores[i];
                counter[0]++;
            }
            else if(host_move[i] == 1){
                scores[1]+=host_scores[i];
                counter[1]++;
            }
            else if(host_move[i] == 2){
                scores[2]+=host_scores[i];
                counter[2]++;
            }
            else {
                scores[3]+=host_scores[i];
                counter[3]++;
            }
            
        }
        free(host_move);
        free(host_scores);
        cudaFree(cuda_move);
        cudaFree(cuda_scores);

        int best_avg_score = 0;
        int best_move = UP;
        for (int i = 0; i < 4; ++i) {
            int move = i;
            if (counter[move] != 0 && ((float) scores[move] / counter[move]) > best_avg_score) {
                best_avg_score = (float) scores[move] / counter[move];
                best_move = move;
            }
        }

        switch(best_move) {
            case UP:
                game.up(false);
                break;
            case DOWN:
                game.down(false);
                break;
            case LEFT:
                game.left(false);
                break;
            case RIGHT:
                game.right(false);
                break;
        }
        round_cnt++;
        if(game.getHighestTile()>=2048)
            break;
    }
    if (display_level <= 1) {
        std::cout << "Done!" << (display_level == 0 ? "\n" : "");
    }
    if (display_level >= 1) {
        std::cout << std::endl << game << std::endl;
    }
    
    std::pair<int, int> ret(game.getHighestTile(), game.score);
    return ret;
}

/**
 * Creates and completes n-many games using the MonteCarlo simulation function.
 * Tabulates data from each game in order to display results at completion.
 */
int monteCarloSolve(int n, int runs, int display_level,
    std::vector<int> &scores, std::vector<int> &highest_tiles) {

    int successes = 0;
 
    for (int i = 0; i < n; ++i) {
        std::pair<int, int> result = monteCarloSimulateGame(runs, display_level, Game());
        if (result.first >= WIN) {
            successes++;
        }
        highest_tiles.push_back(result.first);
        scores.push_back(result.second);
    }
    return successes;
}

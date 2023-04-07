#include "MonteCarlo.hpp"
#include <chrono>
#include <future>
#include <thread>
#include <unistd.h>
#include <fstream>

std::ofstream ofs1, ofs2;

int round_cnt = 0;
std::vector<double> time_vec;

/**
 * From a given game state, chooses random moves until the game is completed.
 * @return (int FIRST_MOVE, int FINAL_SCORE)
*/


std::vector<std::pair<int, int>> simulateOneRun(Game game) {
    std::vector<std::pair<int, int>> ret;
    ret.emplace_back(std::pair<int, int>(0, 0));
    ret.emplace_back(std::pair<int, int>(0, 0));
    ret.emplace_back(std::pair<int, int>(0, 0));
    ret.emplace_back(std::pair<int, int>(0, 0));
    // std::cout<<std::this_thread::get_id()<<"\n";
    for(int i=0; i<1024/4; i++){
        Game game_cpy = game;
        int first_move = -1;
        

        while(game_cpy.canContinue()) {
            board before_state = game_cpy.state;
            std::vector<int> move_bank = {UP, DOWN, LEFT, RIGHT};

            while (before_state == game_cpy.state) {
                int random_pos = DISTS[move_bank.size() - 1](rng);
                int chosen_move = move_bank[random_pos];

                switch(chosen_move) {
                    case UP:
                        game_cpy.up(false);
                        break;
                    case DOWN:
                        game_cpy.down(false);
                        break;
                    case LEFT:
                        game_cpy.left(false);
                        break;
                    case RIGHT:
                        game_cpy.right(false);
                        break;
                }
                if (first_move == -1) {
                    first_move = chosen_move;
                }
                move_bank.erase(move_bank.begin() + random_pos);
            }
        }
        ret[first_move].first++;
        ret[first_move].second+=game_cpy.score;
    }
    
    // std::pair<int, int> ret(first_move, game_cpy.score);
    return ret;
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
    while (game.canContinue()) {
        if (display_level >= 2) {
            std::cout << std::endl << game;
        }
        int scores[4] = {0, 0, 0, 0};
        int counter[4] = {0, 0, 0, 0};

        // std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        // for (int i = 0; i < runs/4; ++i) {
            
           
        // }
        std::future<std::vector<std::pair<int, int> > > res1 = std::async(std::launch::async,simulateOneRun,game);
        std::future<std::vector<std::pair<int, int> > > res2 = std::async(std::launch::async,simulateOneRun,game);
        std::future<std::vector<std::pair<int, int> > > res3 = std::async(std::launch::async,simulateOneRun,game);
        std::future<std::vector<std::pair<int, int> > > res4 = std::async(std::launch::async,simulateOneRun,game);



        std::vector<std::pair<int, int>> val1 = res1.get();
        std::vector<std::pair<int, int>> val2 = res2.get();
        std::vector<std::pair<int, int>> val3 = res3.get();
        std::vector<std::pair<int, int>> val4 = res4.get();

        
        counter[0] = counter[0] + val1[0].first + val2[0].first + val3[0].first + val4[0].first;
        counter[1] = counter[1] + val1[1].first + val2[1].first + val3[1].first + val4[1].first;
        counter[2] = counter[2] + val1[2].first + val2[2].first + val3[2].first + val4[2].first;
        counter[3] = counter[3] + val1[3].first + val2[3].first + val3[3].first + val4[3].first;

        scores[0] = scores[0] + val1[0].second + val2[0].second + val3[0].second + val4[0].second;
        scores[1] = scores[1] + val1[1].second + val2[1].second + val3[1].second + val4[1].second;
        scores[2] = scores[2] + val1[2].second + val2[2].second + val3[2].second + val4[2].second;
        scores[3] = scores[3] + val1[3].second + val2[3].second + val3[3].second + val4[3].second;


        // std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        // std::cout<<"time measure in one roune: "<<time_span.count()<<"us"<<"\n";
        // ofs1<<time_span.count()<<"\n";
        // time_vec.emplace_back(time_span.count());
        

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
    ofs2.open("game.txt");

    for (int i = 0; i < n; ++i) {
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        std::pair<int, int> result = monteCarloSimulateGame(runs, display_level, Game());
        if (result.first >= WIN) {
            successes++;
        }
        highest_tiles.push_back(result.first);
        scores.push_back(result.second);
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        
        ofs2<<time_span.count()<<"\n";
    }
    ofs2.close();
    
    return successes;
    
}

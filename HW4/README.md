# PP-f22_HW4
###### tags: PP-f22

[TOC]

## Q1
1. 在 `mpirun` 指令加入 `-N <num_of_process_on_each_node>` 來控制
2. `MPI_Comm_size()` retrieve the total number of precesses, `MPI_Comm_rank()` retrieve the rank of MPI process
## Q2
1. 這兩個 function 執行後會暫停整個 process 直到收到回傳值才會繼續往下執行，所以是 blocking
2. 
![](https://i.imgur.com/EZkozDM.png)

## Q3
1. 
![](https://i.imgur.com/hTffQA3.png)
2.以時間結果來看，兩者是差不多速度，會影響到整體速度的就是 "向前傳遞的次數" ，linear 的傳輸次數與 process 數量關係為 O(N) (N 為 process 數量)，而 tree 的也是為O(N) (N 為 process 數量)，在2~16這個範圍理所當然不會有顯著的差距
3. 對 tree 來說需要多一道計算往前傳遞給誰的算法，而 linear 不需要計算，直接傳給 rank 0即可，所以當process 數量增加時， linear 執行會更有效率

## Q4
1.
![](https://i.imgur.com/kwT9BBz.png)
2.以我的 code 來說，`MPI_Irecv()` 就是 non-blocking communication ，process 並不會等到回傳值才繼續，一呼叫完這個函式後會繼續往下執行
3.以時間來看沒有顯著的差距，推測是各個 process 執行的很快，因為用 non-blocking 省下來的時間並不明顯，反而因為 function 回傳而產生而外的 overhead 所以稍微慢一點點

## Q5
1.![](https://i.imgur.com/FH8scXV.png)

## Q6
1.
![](https://i.imgur.com/qQYW71W.png)





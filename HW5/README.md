# PP-f22_HW5
###### tags: PP-f22

[TOC]

## Q1
首先，論這三個方法的優劣
- kernel1 的缺點是多分配一塊 host 端的 memory ，而且沒有參與實際計算，只有中途承接 memory 再複製到另一個 host 端的區塊，優點就在於撰寫容易，分配好整塊 memory，kernel function 的 index 照 x, y 座標抓位置即可
- kernel2 的優點是透過 cuda 抓取可以讓 memory 對齊的量來分配 thread，使用者不必自己計算對齊後的量值就可以直接拿 pitch 來做使用，缺點就是會因為要對齊而使用更多的 memory ，有額外的 overhead
- kernel3 優點是可以額外切工作，當一個 thread　完成更多工作的話，可以減少在 grid 和 block 中的移動，但缺點是裁切會依據不同切法會有差很多的效果，例如切到無法對齊反而會花更多時間來存取 memory

在未看結果之前會猜測速度 : kernel2 > kernel1 > kernel3 ，kernel3 容易因無法對齊而花更多 overhead，而kernel2 最快則是認為以工作繁重度來說，存取次數大於存取的量

## Q2
![](https://i.imgur.com/KAcYUd0.png)
![](https://i.imgur.com/r1fLOIh.png)

## Q3
結果上，kernel2 不如我的預期，猜測是大部分情況都沒有對齊，所以每次在存取的時候多存取很多記憶體，所以效果比 kernel1 還差，kernel2 和 kernel3 有試過其他種拆解 block 的數量，但最後選擇最好的也沒有超過 kernel1

## Q4
這邊的方法就類似 kernel1，因為這是實測下來最快的方式，少 copy 一次 memory 從 host 到 host 端就加速很多了，沒有試其他的方式

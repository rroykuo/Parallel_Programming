# PP-f22_HW2
###### tags: PP-f22

[TOC]

## Q1

VIEW 1
>![](https://i.imgur.com/wqrmsDw.png)
>
VIEW 2
>![](https://i.imgur.com/AyEKFsd.png)

在VIEW 1 中，只有thread 2的加速效果接近線性成長(接近2倍)，但是在thread3、4中效果卻很差，尤其是thread 3效果特別差，以VIEW 1繪製的圖形來看，大約是上下對稱，所以推測效能如此差是因為不同的 thread 所分到的工作量不同，造成在 join 的時候有 overhead 在等待工作量大的thread

:::info
整體圖片
>![](https://i.imgur.com/o4spoY5.png)

拆分 2 threads，上下各兩半，可以看出兩個thread的工作量大致相同
>![](https://i.imgur.com/4ohTeXF.png)
>
>![](https://i.imgur.com/l8aFmUa.png)

拆分 3 threads，中間這塊白色分佈特別多，猜測這部分的工作量較大，造成overhead
>![](https://i.imgur.com/MwOUsRj.png)
>
>![](https://i.imgur.com/k4R6AlA.png)
>
>![](https://i.imgur.com/KY8axRV.png)
:::

而在 VIEW 2 卻又大致呈現線性成長(效果還是很差)，推測是因為 VIEW 2的圖形分佈由上到下是工作量差不多的，所以不像 VIEW 1 在3 threads 有這麼差的效果
 
## Q2

為了確認假設，計算各執行緒執行`workerThreadStart()`所花費的時間
>![](https://i.imgur.com/2TLSeRP.png)

以下執行在 VIEW 1
* 2 threads
    * ![](https://i.imgur.com/ZyHiTjY.png)
* 3 threads
    * ![](https://i.imgur.com/Hwmr5au.png)
* 4 threads
    * ![](https://i.imgur.com/hD9xpLF.png)

由上面結果可證實推測，不同執行緒所分配到的工作量的確不同，尤其是 3 threads，Thread 1 工作量特別大，可能和繪製的白色區塊有關聯

---
以下這個 function 決定各個 pixel 的量值(0~255)
>![](https://i.imgur.com/6m4uLqk.png)
for 迴圈執行越多次代表計算這個 pixel花越多時間，而越多次迴圈會讓回傳的 pixel 值越高，這邊可以應證白色區域的 pixel 在這 function 花費較大時間運算，越白的區域代表 output[inedex] 的值越高，因此在 3 threads 中，被分配到中間區塊的 thread 有較長的執行時間 

## Q3

由於原先區塊式的拆分工作，會使得個別執行緒分配到的工作量不均，所以採用循環式的分配工作，盡可能讓不同執行緒有差不多的工作量
>![](https://i.imgur.com/cYiZgaK.jpg)

以結果來說，不同數量執行緒獲得改善，尤其是 3 threads 因為平均分配，所以整體呈現線性成長
>![](https://i.imgur.com/X8BRGzY.png)

## Q4

8 threads 的加速效果反而比 4 threads 還差，因為最多只能分配 4 個 threads，所以推測一個 core 會需要負擔２個 thread 的工作量，但由於軟體定義了8個 threads，所以執行上在同一個 core 切換了兩個不同編號的 thread 來執行兩份工作，這過程中的 context switch 等等的 overhead 也許是造成效果更差的原因
>![](https://i.imgur.com/UUmhhbN.png)


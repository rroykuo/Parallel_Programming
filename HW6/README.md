# PP-f22_HW6
###### tags: PP-f22

[TOC]

## Q1



kernel function 的實作上和 serial 版本幾乎一樣，差別是 index 要用 `get_global_id(0)` 取得，和 cuda 的 kernel function 相反，這邊會直接拿到對應的 index 而不需要自己計算，所以 for 迴圈的 i, j 需要反推回去，透過 `id / imageWidth` 和 `id % imageWidth` 取得

`hostFE.c` 基本上就是照助教給的 reference 網站填入參數即可

---

![](https://i.imgur.com/MXivXyG.png)


影響 Performance 的關鍵點在於 kernel function 的參數類別
- `*output` 是所有 workgroup 要共同存取且要寫入，所以只能選擇 `__global`
- `*filter` 也是workgroup 要共同存取但不需要寫入，可以嘗試使用 `__constant`, 依據網路上查到的資料，如果只需要 read data 且資料量不大的話，通常 `__constant` 比 `__global` 速度來的快，我試過兩種寫法在這個參數，速度相差 x10
- *`image`，和 `*filter` 相同狀況但是 size 卻大很多，使用 __constant 會算出錯誤的答案，應該是 size 超過 gpu memory 可以 cache 的大小，所以只能選擇 `__global`
- `filterWidth`, `imageWidth`, `imageHeight` 這三個值是固定存取的常數，所以放在 local 和 private 的層級會很有效率，但是 local 只能存取指標，所以選擇 private，直接 pass by value




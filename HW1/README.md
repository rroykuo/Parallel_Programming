# PP-f22_HW1
###### tags: PP-f22


### Q1


#### vector width:2
>![](https://i.imgur.com/u8fprdz.png)

---

#### vector width:4
>![](https://i.imgur.com/NJRBXAZ.png)

---
#### vector width:8
>![](https://i.imgur.com/4FNWLYT.png)

---

#### vector width:16
>![](https://i.imgur.com/HCmdXyJ.png)

---
Q1-1: Does the vector utilization increase, decrease or stay the same as VECTOR_WIDTH changes? Why?

Ans:
可觀察到 VECTOR_WIDTH 增加時，vector utilization 下降
原因:
**Vector Utilization = 沒有被 mask 的管道數 / 總管道數**

當在我計算流程中，先完成計算的數字就會被遮罩，只要有至少一個數字沒完成就會繼續做乘法，vector size 高的情況會讓數字的指數更參差不齊，因此，相對低 vector size 的情況，後面幾次的乘法是在大多數位置被遮罩的情況下做，使得 vector utilization 變低

---

### Q2-1
此題跳過

---
### Q2-2

此部分我做10次後取平均

not vectorized
>![](https://i.imgur.com/ugKqC5U.png)

vectorized
>![](https://i.imgur.com/zyqwJRe.png)

可以觀察到有 vectorize 前後相差接近4倍的執行時間



Q:What can you infer about the bit width of the default vector registers on the PP machines?
Ans:
![](https://i.imgur.com/7Ll5ksT.png)
應該為16bytes(128bits)，movups 是從 memory load 資料到 register，也代表著 PP machines 的 register 寬度是 128bits


### Q2-3
Provide a theory for why the compiler is generating dramatically different assembly

compiler　對於相同邏輯但是撰寫順序不一樣會有不同的結果
![](https://i.imgur.com/cB9tGML.png)
>ｃ[j]在 if 的邏輯判斷前會無法提前執行　c[j] = b[j]　而卡著，所以此部分無法被平行化


![](https://i.imgur.com/uz6UxE3.png)
>對於這個 case，無論是否b[j] > a[j]，都可以先提前執行 c[j] = b[j] 和 else c[j] = a[j]，直到邏輯處理完再決定要使用哪個結果

以上撰寫邏輯的差異就會造成 compiler 跑出很不一樣的結果

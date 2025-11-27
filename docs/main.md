**一、論文題目（含 Zelpha 縮寫）**

> **Zelpha: Zonal Earth-observation Latent Potential via Hierarchical Aggregation for Ordinal Agricultural Suitability**

Zelpha = **Z**onal **E**arth-observation **L**atent **P**otential via **H**ierarchical **A**ggregation
（中文可譯作：**「Zelpha：利用分區階層聚合估計潛在農業適宜度的遙測序位模型」**）

核心 idea：
把 AgriPotential 的像素級「潛勢等級」當成**潛在連續「農業適宜度場」的粗量化觀測**，
再加上一個**非常簡單的分區（zone-wise）+ 階層式 regularization**，
在理論上證明：這種學習方式對「潛在連續適宜度」是**一致且可解釋**的。

---

## 1. 研究核心問題

> **如何在多光譜、多時間序列的遙測資料上，
> 用極簡單的模型與 loss，穩定且可解釋地學出「連續的農業潛在適宜度場」，
> 同時尊重序位標籤（very low ~ very high）與空間分區結構？**

AgriPotential 提供的是 5 級序位潛勢（very low, low, average, high, very high），
本質上代表的是一個連續、但被粗糙切成 5 個 bin 的「農業適宜度」。

你這篇 Zelpha 要問的就是：

* 我們能不能

  * 用一個很簡單的 **UNet + scalar head** 模型
  * 加上一點點**階層式（pixel ↔ zone）regularization** 和**序位 loss**
  * 在**理論上證明**：這其實是在學一個「連續潛在適宜度函數」？
* 並且在實驗上證明：這樣學出來的「潛在場」
  比單純做 pixel-wise classification / ordinal regression 更穩定、更有解釋力。

---

## 2. 研究目標

1. **建模目標**

   * 提出一個簡單的模型 Zelpha，把 pixel-wise 序位標籤
     解釋為潛在連續變數的區間標註（interval-labelled data）。
   * 透過 zone（分區）階層結構，學出空間上平滑、對序位一致的「潛在農業適宜度場」。

2. **理論目標**

   * 定義一個簡單的 **Zelpha loss**（ordinal loss + zone regularizer）。
   * 證明在 mild 假設下，Zelpha loss 的風險最小化解，
     在 pixel 與 zone 層級上對真實「潛在適宜度」是 **Fisher-consistent / Bayes-consistent**
     （或至少對 MAE-type 風險是一致的）。

3. **實驗目標**

   * 在 AgriPotential 上，比較：

     * 單純 cross-entropy
     * 既有 ordinal loss（比如 cumulative link / CORAL / EMD）
     * 加上 Zelpha zone regularizer 的版本
   * 檢驗：

     * MAE、±1 accuracy、區域尺度的一致性
     * 模型對 class imbalance 與缺標區域的穩健性。

---

## 3. 預期貢獻（Contribution）

1. **理論貢獻**

   * 提出一個**非常輕量**的階層式序位學習框架 Zelpha，
     並給出其對「潛在連續適宜度」的一致性分析與誤差上界。
2. **方法貢獻**

   * 一個**可直接 plug-in 到任意 encoder（UNet / Swin-UNet / CNN）** 的
     zone-aware ordinal loss（只多出一個簡單的 regularization 項）。
3. **應用貢獻**

   * 在 AgriPotential 上證明：
     這種簡單的階層式序位學習，
     相較於單純 pixel-wise ord. regression，
     能更好地捕捉區域尺度的農業潛勢，且對 class imbalance 更穩定。
4. **實務貢獻**

   * 提供區域尺度（zone-level）的潛勢評估與不確定度估計，
     對土地規劃／作物推薦更符合實際決策單位的尺度。

---

## 4. 創新點（Novelty）

1. **「序位 + 區域階層」的極簡框架**

   * 多數遙測 work 要嘛只做 pixel-wise classification，要嘛做 segmentation，
     很少把「序位標籤」和「區域（zone）結構」**同時**放進同一個損失與理論框架。
2. **將序位標籤視為「區間標註」的潛在變數模型**

   * 把等級 (1,2,\dots,K) 解釋成 ((\tau_{k-1}, \tau_k]) 的區間，
     並在 loss 中直接懲罰潛在值落在區間外的程度。
3. **對 zone-average 的顯式 regularization**

   * 把 pixel level 的 prediction 平均成 zone-level
     並對 zone-level 使用額外的 ordinal loss / smoothness，
     理論上可視為在 function space 上加入一個簡單的 quadratic penalty。

---

## 5. 理論洞見與數學架構（概要）

### 5.1 問題形式化

* (x_{i} \in \mathbb{R}^{T \times B})：第 (i) 個 pixel 的多時間、多頻段特徵
  （例如 11 個時間點 × 10 頻段，展平或用 3D conv 皆可）。
* (y_{i,c} \in {1,\dots,K})：對 crop type (c \in {1,2,3}) 的潛勢等級（K=5）。
* 假設存在潛在連續變數：
  [
  Z_{i,c} = f^*(x_i, c) \in \mathbb{R}
  ]
* 序位標籤由閾值產生：
  [
  y_{i,c} = k \quad \Longleftrightarrow \quad \tau_{k-1} < Z_{i,c} \le \tau_k
  ]
  其中 (\tau_0 < \tau_1 < \dots < \tau_K) 為未知閾值。

### 5.2 模型

* 用一個 encoder（例如 UNet）+ head 近似 (f^*(x_i, c))：
  [
  f_\theta(x_i, c) = z_{i,c} \in \mathbb{R}
  ]
* 閾值 ({\tau_k}) 可以：

  * 當成可學習參數（類似 ordinal regression 中的 cut-points），或
  * 固定成等距（簡化）。

### 5.3 Zelpha ordinal loss（像素層級）

對單一 pixel，給定 label (y)，定義「區間違反」的損失：

[
\ell_{\text{ord}}(z, y)
= \big( \tau_{y-1} - z \big)*+^2 + \big( z - \tau*{y} \big)_+^2
]

* 如果 (z) 落在正確區間 ((\tau_{y-1}, \tau_y])，損失為 0。
* 若偏離該區間，懲罰距離的平方。

**理論洞見 1（Fisher consistency 概念）**

在假設：

* (Z|X=x) 的條件分佈連續且有界
* 標籤產生真實遵守上述區間 model

可以證明：當樣本數趨近無窮，最小化期望風險
[
R(f) = \mathbb{E}\left[ \ell_{\text{ord}}(f(X,c), Y) \right]
]
的函數 (f^\dagger)，會把預測值 (f^\dagger(x,c)) 推向
對應區間中最「風險最小」的點，
而這個點與條件中位數/條件平均（看你設計）存在單調對應。

換句話說，你可以 show Zelpha ordinal loss 對一類「潛在連續適宜度」風險是**一致的 surrogate**。
（論文中可以 formalize 為定理＋證明）

### 5.4 Zone-level 階層 regularization

設每個 pixel (i) 屬於一個 zone (g(i) \in {1,\dots,G})：

[
\bar{z}*{g,c} = \frac{1}{|S_g|}\sum*{i \in S_g} f_\theta(x_i, c)
]

定義 zone-level loss（可用相同 ordinal loss 或 smoothness）：

[
L_{\text{zone}} = \sum_{g,c} w_g , \ell_{\text{ord}}(\bar{z}*{g,c}, \tilde{y}*{g,c})
]

* (\tilde{y}_{g,c})：zone 的潛勢標籤，可由 pixel majority 或官方區域標註（如果有）。
* 同時加入 pixel 與 zone 的 regularization：

[
L_{\text{Zelpha}}(\theta) =
\sum_{i,c} \ell_{\text{ord}}(f_\theta(x_i,c), y_{i,c})
;+; \lambda \sum_{g,c} w_g , \ell_{\text{ord}}(\bar{z}*{g,c}, \tilde{y}*{g,c})
;+; \mu \sum_{(i,j)\in \mathcal{N}} (f_\theta(x_i,c) - f_\theta(x_j,c))^2
]

(\mathcal{N}) 為鄰域 pixel（Laplacian 平滑）。

**理論洞見 2（等價於帶 Tikhonov regularization 的學習）**

在固定 encoder 表示、只在線性 head / function space 上優化的簡化情況下，可證明：

* Zelpha 的最小化問題等價於
  在 Hilbert space 上最小化
  「期望 ordinal 損失 + 一個 quadratic norm」，
  並導出一般化誤差上界（依賴 (\lambda,\mu)、Rademacher complexity、樣本量）。

這一段可以是比較數學味的 main theorem + sketch proof。

---

## 6. 方法論（實作面很簡單）

1. **Backbone**

   * 直接沿用 AgriPotential paper 的 2D UNet baseline，
     把 11 時間 × 10 頻段 stack 成 110 channels 輸入。
   * Head 改成：每 crop type 一個 scalar head，輸出 (z_{i,c})。

2. **Loss 組合**

   * Pixel-wise Zelpha ordinal loss（上面的 (\ell_{\text{ord}})）。
   * Zone-level ordinal loss + 鄰域 smoothness penalty。
   * Multi-task over 3 crop types，共享 encoder，head 分開。

3. **訓練細節**

   * 與原論文相同的 train/val/test split（80/10/10），
     保持實驗可比性。
   * Adam / AdamW，學習率 schedule，train 幾十 epoch 即可。

---

## 7. 預計使用的 Dataset

**主要：AgriPotential**

* **影像來源**：Sentinel-2 多光譜、多時間序列
* **時間維度**：2019 年約 11 個月份時間點
* **頻段**：10 個 spectral bands（B2–B4, B5–B8A, B11, B12）
* **空間解析度**：5 m / pixel（super-resolved）
* **patch**：128×128 pixels，total 8890 patches（train 7095, val 914, test 881）
* **標籤**：

  * 3 種作物：viticulture、market gardening、field crops
  * 5 級序位農業潛勢：very low, low, average, high, very high
  * pixel-level annotation，部分像素無標籤（ignore）。

**可選延伸（optional）**

* 若時間與篇幅允許，可在 BigEarthNet / EuroSAT 上
  人工構造簡單的「序位標籤」（例如 NDVI-based quantile），
  驗證 Zelpha 框架的可遷移性。

---

## 8. 與現有研究之區別

1. **相對於 AgriPotential 原論文**

   * 原論文重點在於：

     * 資料集構建與描述
     * 提供 U-Net baseline（classification / regression / ordinal regression），
       沒有做**明確的潛在變數理論建模**，也沒有分區階層 loss。
   * Zelpha 則是：

     * 把這個 dataset 當成「序位區間標註的潛在連續適宜度問題」，
       提出理論框架與 loss。
     * 加上「zone-aware regularization」與一致性分析。

2. **相對於一般 ordinal regression / segmentation 工作**

   * 大多只在 pixel level 使用 ordinal loss，
     沒有整合空間分區與 zone-level 約束。
   * Zelpha 把

     * 序位性（ordinal）
     * 空間平滑（local smoothness）
     * 區域階層結構（zone-level）
       放在同一個統一的損失與理論裡，
       但模型仍然**非常簡單**（UNet + scalar head + regularizer）。

---

## 9. Experiment 設計（概要）

### 9.1 任務設定

* **Task 1：pixel-wise ordinal potential prediction**

  * 輸出 5 級等級；評估：

    * MAE（等級差）
    * Accuracy
    * Accuracy within ±1 等級
    * class-wise F1 / macro-F1。
* **Task 2：zone-level potential prediction**

  * 將 pixel prediction 平均成 zone，和 zone 標籤比較（若可取得）
  * 評估：

    * zone-level MAE
    * zone-level ±1 accuracy。

### 9.2 比較方法（Baselines）

1. **CE-UNet**：

   * multi-class cross-entropy，視等級為 nominal class。
2. **Reg-UNet**：

   * 直接 regression 到 1–5，使用 L2 / L1 loss。
3. **Ord-UNet**：

   * 使用既有 ordinal loss（例如 cumulative link / CORAL / EMD），
     但沒有 zone regularization。
4. **Zelpha（Ours）**：

   * Pixel-wise ordinal + zone-level + smoothness regularizer。

### 9.3 Ablation Studies

* 移除不同項目：

  * no-zone（只保留 pixel ordinal）
  * no-smooth（移除 Laplacian 項）
  * no-shared encoder（每 crop 單獨訓練）
* 改變 (\lambda,\mu) 大小，觀察

  * zone-level vs pixel-level 的 performance trade-off
  * class imbalance 時是否更穩定。

### 9.4 分析與可視化

* **潛在適宜度場的連續 map**：

  * 直接畫出 (f_\theta(x,c)) 的連續值，而不是只看 5 級等級。
* **等級邊界 vs 閾值位置**：

  * 分析學到的閾值 (\tau_k) 是否與土壤／地形特徵有關聯。
* **錯誤案例分析**：

  * 看看模型是否主要在「相鄰等級」錯誤（符合實務可接受性）。

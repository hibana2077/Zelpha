# ZELPHA：Zoom-Equivariant Lipschitz Prototype Hypothesis for Aerial Land-Use Classification

> （ZELPHA：具縮放等變與 Lipschitz 正則之原型假說，用於衛星遙測土地利用分類）

縮寫對應：**Z**oom-**E**quivariant **L**ipschitz **P**rototype **H**ypothesis for **A**erial scenes → ZELPHA

核心 idea：
用一個**很小、結構簡單的 CNN + 原型表示 + Lipschitz 約束**，在像 UC Merced Land Use 這種**老但小又乾淨**的遙測資料集上，專門研究「**縮放／zoom 變化下的分類穩定性**」，給出**數學上的 robustness 條件與界**，而不是只做 engineering SOTA。

---

## 一、資料集設定

**主資料集：UC Merced Land Use**

* 21 類 land-use（agriculture, airplane, beach, residential...），
* 每類 100 張，總共 2100 張影像，解析度 256×256 RGB、約 0.3 m/pixel。([weegee.vision.ucmerced.edu][1])
  → 小、完整、經典，而且是標準的場景 classification 資料集，很適合做「理論導向」的實驗。

**可選擴充：EuroSAT-RGB（附錄或第二組實驗）**

* 基於 Sentinel-2 的土地利用／覆蓋分類，10 類、約 27,000 張影像，13 波段（可只用 RGB 版）。([GitHub][2])

**新任務設定（ZELPHA 任務）**
在 UC Merced 上定義新 protocol：

1. **Train scale**：訓練時只看「一個固定 scale」的影像（例如固定 256×256，限制 zoom augmentation 的範圍）。
2. **Test scale shift**：測試時對每張圖做一系列 zoom-in / zoom-out（例如 0.7×, 0.85×, 1.15×, 1.3×），再 resize 回 256×256。
3. 任務：**在 scale 變化下保持分類結果穩定、性能下降可被理論界估計。**

這樣你等於是**在舊資料集上定義一個「縮放穩健性」的新任務**。

---

## 二、研究核心問題（Research Core Question）

1. **能不能設計一個結構簡單、可分析的 CNN 模型，在遙測影像的縮放變化下仍具有理論上可證明的分類穩定性？**
2. **在小型遙測資料集（UC Merced）上，scale-equivariant + Lipschitz 約束 + prototype-based decision，是否能在「縮放分布移轉」下給出明確的 robustness 條件與實驗驗證？**

---

## 三、研究目標（Objectives）

1. 提出 **ZELPHA 模型**：一個

   * 具 **縮放等變（Zoom-Equivariant）** 的卷積特徵抽取器，
   * 加上 **Lipschitz 約束**，
   * 以 **原型（prototype）為核心的分類頭**。
2. 對於 ZELPHA，建立一個**數學上可證明的條件**：在何種 Lipschitz 常數與原型間距下，對一定範圍的縮放變化仍保持正確分類。
3. 在 UC Merced（以及選配的 EuroSAT）上驗證：

   * ZELPHA 在 scale shift 下的準確率與穩定性優於 baseline CNN / scale-equivariant CNN / 單純 Lipschitz CNN。
4. 定義並公開一個**Scale-Robust UC Merced Benchmark Protocol**（含程式與設定），讓後人可直接沿用。

---

## 四、方法論概觀（Methodology）

### 1. 模型結構（ZELPHA 模型）

**(a) Feature extractor：縮放等變 CNN**

* 採用 **scale-equivariant convolution / steerable filters** 的 idea，把尺度當作 group element 處理；像是 scale-equivariant CNN 或 Scale-Equivariant U-Net 這類方法中的 core 觀念。([arXiv][3])
* 實作上可以簡化為：

  * 對若干個 $scale ( s \in S = {s_1,\dots,s_k} )$ 做共享權重的卷積（group conv），
  * 然後做 **scale pooling**（取平均或最大），得到近似的 scale-invariant feature。

**(b) Lipschitz 約束**

* 使用 spectral normalization / weight normalization 限制每層線性轉換的 operator norm，使整個網路的 Lipschitz 常數 $L$ 有明確上界。([Broad Institute][4])
* 可加上 Lipschitz regularization term：
  $$
  \mathcal{L}*{\text{Lip}} = \lambda \sum_l \max(0, |W_l|*{\text{op}} - 1)^2
  $$
  讓整個網路近似 1-Lipschitz 或某固定 (L)。

**(c) Prototype-based classifier**

* 為每個類別 $c$ 存一個 prototype $\mu_c \in \mathbb{R}^d$。
* 給定輸入 $x$，feature extractor 得到 $z = f_\theta(x)$。
* 分類用 **距離到 prototype**：
  $$
  p(y=c\mid x) \propto \exp(-|z - \mu_c|^2)
  $$
* 加上 **原型分離 loss**，鼓勵不同類別的 prototype 彼此距離大、同類樣本靠近：
  $$
  \mathcal{L}*{\text{proto}} = \sum_i |f*\theta(x_i) - \mu_{y_i}|^2 * \alpha \sum_{c\neq c'} \max(0, m - |\mu_c - \mu_{c'}|)^2
$$

**總 loss：**
$$
\mathcal{L} = \mathcal{L}*{\text{CE}} + \beta \mathcal{L}*{\text{proto}} + \gamma \mathcal{L}_{\text{Lip}}
$$

---

### 2. 縮放等變與 robustness 設計

* 定義縮放操作 $T_s(x)$：對影像做縮放 factor $s$ 並重新 crop/resize 回 256×256。
* 訓練時：

  * 一部分 batch 使用單一 scale，
  * 一部分 batch 對每張圖抽一組 $s \in S$ 做隨機縮放，以強化等變性與 invariance。
* ZELPHA 的設計目標：

  * 讓 $f_\theta$ 在縮放下接近等變：
    $$
    f_\theta(T_s x) \approx \rho_s(f_\theta(x))
    $$
  * 再透過 scale pooling 取得近似 invariant representation。

---

## 五、數學理論推演與證明（方向）

這邊給你可以寫進「理論部分」的主幹，詳細 proof 可以自己鋪：

### 1. 問題建模

* 影像空間 $\mathcal{X}$，類別集合 $\mathcal{Y} = {1,\dots,C}$。
* 縮放群 $\mathcal{S} = [s_{\min}, s_{\max}]$，$T_s: \mathcal{X} \to \mathcal{X}$ 為對應的影像縮放操作。
* Feature extractor $f_\theta: \mathcal{X}\to\mathbb{R}^d$ 是 (L)-Lipschitz：
  $$
  |f_\theta(x) - f_\theta(x')| \le L\cdot d_{\mathcal{X}}(x,x')
  $$
* 原型集合 ${\mu_c}*{c=1}^C$，分類 rule：
  $$
  \hat{y}(x) = \arg\min_c |f*\theta(x) - \mu_c|
  $$

### 2. Robust margin 定義

對樣本 $(x,y)$，定義 margin：
$$
m(x) = \min_{c\ne y} \left(|f_\theta(x) - \mu_c| - |f_\theta(x) - \mu_y|\right)
$$

若 $m(x) > 0$ 則分類正確。

### 3. 命題：縮放擾動下的 margin 下界

**命題 1（Scale-Robust Margin）**
假設 $f_\theta$ 是 $L$-Lipschitz，且對每個縮放 $s$ 有
$$
d_{\mathcal{X}}(x, T_s x) \le \delta_s
$$
則對所有 $s$：
$$
|m(T_s x) - m(x)| \le 2L \delta_s
$$

**證明 sketch：**

* 對任意類別 $c$，
  $$
  \big||f_\theta(T_s x) - \mu_c| - |f_\theta(x) - \mu_c|\big|
  \le |f_\theta(T_s x) - f_\theta(x)|
  \le L\delta_s
  $$
* margin 是兩個「距離到 prototype」的差（y vs. 其他 c），
  → 兩項各變動最多 $L\delta_s$，
  → 差的變動最多 $2L\delta_s$。

**推論（Corollary）**
若存在 $\bar{s}$ 使得對所有 $|s-1|\le \bar{s}$，皆有 $2L\delta_s < m(x)$，
則所有這些縮放版本 $T_s x$ 仍維持正確分類。

→ **這給了「在多大 range 的縮放下仍穩定」的明確條件**：

* margin 越大（原型分得越開）、
* Lipschitz 常數越小、
* 影像縮放後與原圖距離越小，
  則模型越穩定。

### 4. 一般化界（可以寫成延伸）

基於 Lipschitz network 的一般化理論（Lipschitz regularity of DNNs、Sorting out Lipschitz function approximation 等）可以得到：([NeurIPS Papers][5])

* 函數類 $\mathcal{F}$ 的 Rademacher complexity 與 Lipschitz 常數成正比。
* 即在控制 $L$ 下，你可以給出樣本數 $n$ 的 generalization bound：
  $$
  \mathcal{R}(f) \le \hat{\mathcal{R}}(f) + O\Big(\frac{L \cdot B}{\sqrt{n}}\Big)
  $$
  其中 $B$ 與輸入空間直徑有關。

你可以在論文中給出一個 **「在縮放群下的 robust risk」**：
$$
\mathcal{R}*{\text{rob}}(f) = \mathbb{E}*{(x,y)}\Big[\sup_{s\in\mathcal{S}} \ell(f(T_s x),y)\Big]
$$
再用上面的 margin + Lipschitz 不等式把它 upper bound 成標準 risk 加上一個與 $L$ 和 $\sup_s \delta_s$ 有關的項。

---

## 六、預期貢獻與創新點

### 貢獻

1. **ZELPHA 模型**：一個結構簡單、可實作的小型 CNN，結合

   * scale-equivariant feature learning、
   * Lipschitz 約束、
   * prototype-based decision rule，
     專門針對遙測場景的縮放穩健性。
2. **理論結果**：給出

   * margin 在縮放擾動下的上下界，
   * 在 Lipschitz constant 與 prototype 間距條件下的 robustness 條件。
3. **新 benchmark protocol**：在 UC Merced/EuroSAT 上定義**縮放穩健性測試流程**，並公開 code。
4. 顯示在**小型實際遙測資料集**上，理論導向的設計（而不是超深網路）也能帶來明顯的 scale shift robustness 改善。

### 創新點（相對現有研究）

1. 相對於既有 UC Merced / EuroSAT 工作多半只做「一般分類」或深度 CNN benchmark，你的工作：

   * 把重點換成「縮放／zoom shift 下的穩健性」，
   * 並且有明確的數學分析，而不是純 empirical。([arXiv][6])
2. 與既有 scale-equivariant CNN 研究不同：這些工作多半在自然影像或 3D/segmentation 上展示效果，較少聚焦在小型遙測資料集與 Lipschitz+prototype 的組合。([arXiv][3])
3. 與 Lipschitz-constrained network 的工作不同：那一線文獻主要關於 adversarial robustness 或 Wasserstein distance estimation，你這邊把它具體套在「縮放群」上的幾何擾動，並給出實際的遙測應用示範。([Broad Institute][4])

---

## 七、與現有研究之區別（更明確講法）

寫在 Related Work 最後可以這樣對比：

1. **傳統遙測場景分類**：像各種在 UC Merced 與 EuroSAT 上的 CNN / ensemble 方法，多關注提升 accuracy；你關注的是**在未見過的縮放條件下的穩健性 + 理論分析**。([arXiv][6])
2. **Scale-equivariant networks**：雖然已有對 scale-equivariance 的一般理論與實作，但缺少「Lipschitz + prototype margin 分析」和「遙測縮放 benchmark」的結合。([arXiv][3])
3. **Lipschitz networks 理論**：現有工作多在 toy setting 或 adversarial norm ball perturbation，較少具體到「實務影像上的幾何變換（縮放）」與遙測場景。([NeurIPS Papers][5])

---

## 八、實驗設計（Experiment Design）

### 1. Dataset split 與 protocol

* **UC Merced**：

  * Train/Val/Test 60/20/20 split（或 50/20/30），確保每類都平均。
  * 對 test set 建立多個縮放版本：

    * $s \in {0.7, 0.85, 1.0, 1.15, 1.3}$。
    * 每個 $s$ 都形成一個 test subset。
* **EuroSAT（選配）**：

  * 使用標準 train/test split，類似方式建立縮放 test set。([arXiv][6])

### 2. Baselines

1. **Vanilla CNN**：ResNet-18 / ResNet-34，不做特殊 scale-equivariant 設計。
2. **Data-aug only CNN**：加強 random resize/crop as augmentation，但無 Lipschitz、無 prototype。
3. **Scale-equivariant CNN**：參考 scale-equivariant conv 網路設計一個 baseline。([arXiv][3])
4. **Lipschitz CNN**：只加 spectral norm + Lipschitz regularization，但 classifier 用標準 linear head。

→ ZELPHA 則是 **scale-equivariant + Lipschitz + prototype** 全部都有。

### 3. 評估指標

* **Accuracy vs. scale factor**：

  * 畫出每個模型在不同 $s$ 的 accuracy 曲線，看誰的 degradation 最小。
* **Robust accuracy**：

  * 對每張 test image 取所有縮放版本，若「對所有 $s$ 都分類正確」才算 1。
* **Margin 分析**：

  * 測量 ZELPHA 在 feature space 的 margin 分佈，對比 robustness。
* **Calibration / Uncertainty（選配）**：

  * 看在大縮放 shift 下，模型的不確定性有沒有變大；可算 ECE。

### 4. Ablation studies

* 移除 Lipschitz regularization → 看 robustness 掉多少。
* 不用 prototype-based classifier，改 linear head → 比較 margin 與 scale robustness。
* 不做 scale pooling，只做 data augmentation → 看哪一部分貢獻最大。

---

[1]: https://weegee.vision.ucmerced.edu/datasets/landuse.html?utm_source=chatgpt.com "UC Merced Land Use Dataset"
[2]: https://github.com/phelber/EuroSAT?utm_source=chatgpt.com "EuroSAT: Land Use and Land Cover Classification with ..."
[3]: https://arxiv.org/abs/1910.11093?utm_source=chatgpt.com "[1910.11093] Scale-Equivariant Steerable Networks"
[4]: https://www.broadinstitute.org/talks/efficient-lipschitz-constrained-neural-networks?utm_source=chatgpt.com "Efficient Lipschitz-constrained neural networks"
[5]: https://papers.neurips.cc/paper/7640-lipschitz-regularity-of-deep-neural-networks-analysis-and-efficient-estimation.pdf?utm_source=chatgpt.com "Lipschitz regularity of deep neural networks"
[6]: https://arxiv.org/abs/1709.00029?utm_source=chatgpt.com "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification"
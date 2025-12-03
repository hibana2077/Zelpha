# ZELPHA：Zoom-Equivariant Lipschitz Prototype Hypothesis for Aerial Land-Use Classification

> （ZELPHA：具縮放等變與 Lipschitz 正則之原型假說，用於衛星遙測土地利用分類）

縮寫對應：**Z**oom-**E**quivariant **L**ipschitz **P**rototype **H**ypothesis for **A**erial scenes → ZELPHA

核心 idea：  
把 **Zoom-Equivariant + Lipschitz + Prototype** 這一套設計，做成可以直接掛在各種現成視覺 backbone（如 ConvNeXt、DenseNet、RepVGG、RegNetZ、DeiT3 transformer）的 **plug-and-play 訓練 protocol / head**。在 UC Merced Land Use 這種小而乾淨的遙測資料集上，我們定義一個「縮放穩健性（scale-robust）」的 benchmark protocol，比較「原始模型」與「加上 ZELPHA 後的模型」在縮放變化下的 **robust accuracy**，並為這種穩健性提供可證明的數學條件，而不是只追求 engineering SOTA。

---

## 一、資料集設定

**主資料集：UC Merced Land Use**

* 21 類 land-use（agriculture, airplane, beach, residential...），
* 每類 100 張，總共 2100 張影像，解析度 256×256 RGB、約 0.3 m/pixel。([weegee.vision.ucmerced.edu][1])  
  → 小、完整、經典，而且是標準的場景 classification 資料集，很適合做「理論導向」的實驗與縮放穩健性分析。

**可選擴充：EuroSAT-RGB（附錄或第二組實驗）**

* 基於 Sentinel-2 的土地利用／覆蓋分類，10 類、約 27,000 張影像，13 波段（可只用 RGB 版）。([GitHub][2])

**新任務設定（ZELPHA 任務）**  
在 UC Merced 上定義新 protocol：

1. **Train scale**：訓練時主要在「一個固定 scale」的影像上學習（例如固定 256×256，僅做有限度的 RandomResizedCrop），避免直接把 test-time 的縮放條件「偷看進去」。
2. **Test scale shift**：測試時對每張圖做一系列 zoom-in / zoom-out（例如 0.7×, 0.85×, 1.15×, 1.3×），再 resize 回 256×256。
3. 任務：**在 scale 變化下保持分類結果穩定，性能下降可用理論上的 margin / Lipschitz 條件來解釋。**

這樣等於是在舊資料集上**定義一個「縮放穩健性」的新任務與新 benchmark protocol**。

---

## 二、研究核心問題（Research Core Question）

1. **能不能設計一個 backbone-agnostic、可 plug-and-play 的訓練 protocol / head，只要掛在現有的 CNN / Transformer backbone 上，就能在遙測影像的縮放變化下帶來可證明、可觀察的穩健性提升？**
2. **在小型遙測資料集（UC Merced）上，scale-aware 的 prototype head + Lipschitz 約束，是否能在「縮放分布移轉」下，同時在 CNN 與 Vision Transformer backbone 上給出明確的 robustness 條件與實驗驗證？**

---

## 三、研究目標（Objectives）

1. 提出 **ZELPHA protocol**：一個

   * 以各種現成的視覺 backbone（ConvNeXt、DenseNet、RepVGG、RegNetZ、DeiT3 等）作為特徵抽取器，
   * 在其上方接上 **Lipschitz 控制的原型（prototype）分類頭**，
   * 並搭配針對縮放擾動設計的訓練與評估 protocol。

2. 對於 ZELPHA，建立一個**數學上可證明的條件**：在何種 Lipschitz 常數與原型間距下，對一定範圍的縮放變化仍保持正確分類。

3. 在 UC Merced（以及選配的 EuroSAT）上驗證：

   * 對多種 backbone 家族（包含 CNN 與 Vision Transformer），比較「原始 linear classifier」與「套用 ZELPHA head」在 scale shift 下的 robust accuracy；
   * 檢驗這些差異是否在統計上顯著（例如 paired t-test），並與 margin / Lipschitz 理論對應。

4. 定義並公開一個**Scale-Robust UC Merced Benchmark Protocol**（含資料切分、縮放設定、評估指標與程式），讓後人可以在任何 backbone 上重複這組實驗。

---

## 四、方法論概觀（Methodology）

### 1. 模型結構（ZELPHA 模型）

整體可以拆成「**backbone**」與「**ZELPHA head**」兩層：

**(a) Feature extractor：主流 backbone（CNN / Transformer）**

* 使用 ImageNet 預訓練的 backbone 作為特徵抽取器，例如：
  * ConvNeXt-Atto 代表 modern CNN，
  * DenseNet121 類型的密集連接 CNN，
  * RepVGG（a0 / a1 / a2 / b0 / b1 / b2）代表 re-parameterizable CNN 家族，
  * RegNetZ-C16 代表輕量級卷積網路，
  * DeiT3-Small Patch16 代表 Vision Transformer。
* 我們將這些 backbone 視為同一個抽象函數
  $$z = f_{\text{backbone}}(x) \in \mathbb{R}^d,$$
  並盡量只改動最後一層 classifier，使 ZELPHA 能直接 plug-and-play。

**(b) ZELPHA head：Lipschitz + prototype-based classifier**

在 backbone 輸出 $z$ 之上，我們定義一組原型 $\{\mu_c\}_{c=1}^C$ 與 Lipschitz 控制的映射：

* 先用一個小型 MLP 或線性層 $g_\phi$ 做最後的特徵投影（對其權重施加 spectral / weight normalization），得到
  $$h(x) = g_\phi(f_{\text{backbone}}(x)) \in \mathbb{R}^m.$$
* 為每個類別 $c$ 存一個 prototype $\mu_c \in \mathbb{R}^m$。
* 分類時，以距離到 prototype 決定 logits：
  $$
  \ell_c(x) = -\|h(x) - \mu_c\|^2,\quad
  p(y=c\mid x) \propto \exp(\ell_c(x)).
  $$

為了讓原型在特徵空間中更具有幾何結構，我們加入 **原型分離 loss**：
$$
\mathcal{L}_{\text{proto}} =
\sum_i \|h(x_i) - \mu_{y_i}\|^2
+ \alpha \sum_{c\neq c'} \max(0, m - \|\mu_c - \mu_{c'}\|)^2,
$$
鼓勵同類樣本靠近各自的 prototype、不同類別的 prototype 彼此分開。

**(c) Lipschitz 約束**

* 對 $g_\phi$ 中的線性 / 卷積層加上 spectral normalization 或 weight normalization，使其 operator norm 有明確上界。
* 若需要，也可以對 backbone 的最後幾層一起施加 Lipschitz regularization，使整個 $f_\theta = g_\phi \circ f_{\text{backbone}}$ 的 Lipschitz 常數 $L$ 可被估計與控制。
* 額外加入 regularization term：
  $$
  \mathcal{L}_{\text{Lip}} =
  \lambda \sum_l \max(0, \|W_l\|_{\text{op}} - L_0)^2,
  $$
  將每層的 operator norm 壓在預定的門檻 $L_0$ 附近。

**(d) 總 loss**

訓練時的目標函數為：
$$
\mathcal{L} = \mathcal{L}_{\text{CE}}
+ \beta \mathcal{L}_{\text{proto}}
+ \gamma \mathcal{L}_{\text{Lip}},
$$
其中 $\mathcal{L}_{\text{CE}}$ 為標準 cross-entropy loss。  
這個定義完全與 backbone 無關，因此可以在不同模型上共用。

### 2. 縮放等變與 robustness 設計

* 定義縮放操作 $T_s(x)$：對影像做比例因子為 $s$ 的縮放，並 crop / resize 回 256×256。
* **訓練階段**：
  * 以某個「標準 scale」（例如 1.0，搭配少量 RandomResizedCrop）為主，確保模型沒有直接看過 test-time 的極端縮放條件；
  * 在部分 mini-batch 中加入有限範圍的 scale augmentation，以避免模型過度依賴單一尺度細節。
* **測試階段（scale shift protocol）**：
  * 對每張 test image 產生多個版本 $\{T_s x\}_{s\in S}$；
  * 在每個 scale 上分別進行預測，並計算
    * 單一 scale 的 accuracy，
    * 以及在所有 $s\in S$ 都預測正確才算成功的 **robust accuracy**。
* ZELPHA 的設計目標是：
  * 透過 Lipschitz 約束，讓 $h(x)$ 在縮放擾動下不會劇烈變動；
  * 透過 prototype margin，讓不同類別在特徵空間的距離夠大；
  * 兩者合起來，保證在一定範圍的縮放下，決策結果不會改變，這部分在下一節以 margin bound 形式給出。

---

## 五、數學理論推演與證明（方向）

這裡整理可以寫進「理論部分」的主幹，重點是：**推導與 backbone 無關，只要求整個 $f_\theta$ 是 Lipschitz 且使用 prototype-based decision rule**。

### 1. 問題建模

* 影像空間 $\mathcal{X}$，類別集合 $\mathcal{Y} = \{1,\dots,C\}$。
* 縮放群 $\mathcal{S} = [s_{\min}, s_{\max}]$，$T_s: \mathcal{X} \to \mathcal{X}$ 為對應的影像縮放操作。
* Feature extractor $f_\theta: \mathcal{X}\to\mathbb{R}^d$ 是 $L$-Lipschitz：
  $$
  \|f_\theta(x) - f_\theta(x')\| \le L\cdot d_{\mathcal{X}}(x,x').
  $$
* 原型集合 $\{\mu_c\}_{c=1}^C$，分類 rule：
  $$
  \hat{y}(x) = \arg\min_c \|f_\theta(x) - \mu_c\|.
  $$

### 2. Robust margin 定義

對樣本 $(x,y)$，定義 margin：
$$
m(x) = \min_{c\ne y} \Big(\|f_\theta(x) - \mu_c\|
- \|f_\theta(x) - \mu_y\|\Big).
$$

若 $m(x) > 0$ 則分類正確，且 margin 越大，對擾動越穩健。

### 3. 命題：縮放擾動下的 margin 下界

**命題 1（Scale-Robust Margin）**  
假設 $f_\theta$ 是 $L$-Lipschitz，且對每個縮放 $s$ 有
$$
d_{\mathcal{X}}(x, T_s x) \le \delta_s,
$$
則對所有 $s$：
$$
|m(T_s x) - m(x)| \le 2L \delta_s.
$$

*證明 sketch*：

* 對任意類別 $c$，
  $$
  \big|\|f_\theta(T_s x) - \mu_c\| - \|f_\theta(x) - \mu_c\|\big|
  \le \|f_\theta(T_s x) - f_\theta(x)\|
  \le L\delta_s.
  $$
* margin 是「到錯誤類別原型的距離」減去「到正確原型的距離」，兩項各自最多變動 $L\delta_s$，相減後最多變動 $2L\delta_s$。

**推論（Corollary）**  
若存在 $\bar{s}$ 使得對所有 $|s-1|\le \bar{s}$，皆有
$$
2L\delta_s < m(x),
$$
則所有這些縮放版本 $T_s x$ 仍維持正確分類。  
→ margin 越大、Lipschitz 常數越小、縮放後影像與原圖距離越小，模型對縮放越穩健。

### 4. 一般化界（可以寫成延伸）

可引用 Lipschitz network 一般化理論（例如 Lipschitz regularity of DNNs 等），說明：

* 函數類 $\mathcal{F}$ 的 Rademacher complexity 與 Lipschitz 常數成正比；
* 在控制 $L$ 下，可以得到樣本數 $n$ 的 generalization bound：
  $$
  \mathcal{R}(f) \le \hat{\mathcal{R}}(f)
  + O\Big(\frac{L \cdot B}{\sqrt{n}}\Big),
  $$
  其中 $B$ 與輸入空間直徑有關。

再進一步定義在縮放群下的 **robust risk**：
$$
\mathcal{R}_{\text{rob}}(f)
= \mathbb{E}_{(x,y)}\Big[\sup_{s\in\mathcal{S}} \ell(f(T_s x),y)\Big],
$$
並用上面的 margin + Lipschitz 不等式，將其 upper bound 成標準 risk 加上一個與 $L$ 和 $\sup_s \delta_s$ 有關的項，連結到實驗中觀察到的 robust accuracy 變化。

---

## 六、預期貢獻與創新點

### 貢獻

1. **ZELPHA：backbone-agnostic 的 scale-robust head / protocol**  
   提出一個可以直接掛在各種主流 backbone（CNN 與 Vision Transformer）上的輕量 head，結合
   * scale-aware 的特徵表示與訓練 protocol、
   * Lipschitz 約束、
   * prototype-based decision rule，
   專門針對遙測場景的縮放穩健性。

2. **理論結果**：給出
   * margin 在縮放擾動下的上下界，
   * 在 Lipschitz constant 與 prototype 間距條件下的 robustness 條件，
   並說明這些條件如何對應到實際的 robust accuracy。

3. **新 benchmark protocol**：在 UC Merced / EuroSAT 上定義**縮放穩健性測試流程**，包括資料切分、縮放設定與評估指標，並公開實作程式，讓其他人可以在任何 backbone 上重現與比較。

4. **跨架構的一致實驗證據**：在多個主流 backbone 家族與多個隨機 seed 上，系統性比較「原始模型」與「加上 ZELPHA」的表現，顯示在絕大多數情況下，ZELPHA 都能提升在縮放 shift 下的 robust accuracy，少數失敗案例則可作為後續分析與改進的出發點。

### 創新點（相對現有研究）

1. 相對於既有 UC Merced / EuroSAT 工作多半只做「一般分類」或深度 CNN benchmark，本研究：
   * 把重點換成「縮放／zoom shift 下的穩健性」，
   * 並在多種 backbone 上同時給出數學分析與實驗驗證，而不是只在單一模型上做 case study。([arXiv][6])

2. 與既有 scale-equivariant networks 不同：過去多數工作設計特定的 scale-equivariant 架構並在自然影像或 segmentation 上驗證；本研究則偏向「**protocol / head** 的設計」，強調只要遵守 Lipschitz + prototype 的條件，即使 backbone 本身不是嚴格的 group-equivariant，也能在縮放 benchmark 上得到可觀的穩健性效果，並額外結合 Lipschitz + prototype margin 的分析。([arXiv][3])

3. 與 Lipschitz-constrained network 的文獻不同：既有工作多關注 adversarial norm ball 擾動或 Wasserstein 距離估計，本研究則把 Lipschitz 條件具體套在「幾何縮放」這類實際遙測影像上的變換，並與 prototype margin 與 benchmark protocol 連結起來，給出完整的理論＋實驗故事。([NeurIPS Papers][5])

---

## 七、與現有研究之區別（更明確講法）

寫在 Related Work 最後可以這樣對比：

1. **傳統遙測場景分類**：像各種在 UC Merced 與 EuroSAT 上的 CNN / ensemble 方法，多關注提升 accuracy；本研究則關注**在未見過的縮放條件下的穩健性 + 理論分析**，並且驗證這種穩健性在多種 backbone 上都能成立。([arXiv][6])

2. **Scale-equivariant networks**：雖然已有對 scale-equivariance 的一般理論與實作架構，但多半需要專門的卷積設計；本研究則展示一個「只改 head 和訓練 protocol」的做法，也能在實際 benchmark 上得到類似的縮放穩健性效果，並額外結合 Lipschitz + prototype margin 的分析。([arXiv][3])

3. **Lipschitz networks 理論**：現有工作多在 toy setting 或 adversarial norm ball perturbation 上分析 Lipschitz DNN，本研究則以遙測場景中的「幾何縮放」為主軸，將 Lipschitz 約束、prototype margin、robust risk 與實際的 scale-shift benchmark 串起來。([NeurIPS Papers][5])

---

## 八、實驗設計（Experiment Design）

### 1. Dataset split 與 protocol

* **UC Merced**：
  * Train/Val/Test 60/20/20 split（或 50/20/30），確保每類都平均。
  * 對 test set 建立多個縮放版本：
    * $s \in \{0.7, 0.85, 1.0, 1.15, 1.3\}$。
    * 每個 $s$ 都形成一個獨立的 test subset。
* **EuroSAT（選配）**：
  * 使用標準 train/test split，類似方式建立縮放 test set。([arXiv][6])

### 2. Baselines

對每一個 backbone（ConvNeXt、DenseNet、RepVGG 系列、RegNetZ、DeiT3），我們都訓練兩種版本：

1. **Vanilla**：保留原本的 global pooling + linear classifier，使用標準 cross-entropy，在 Train scale 上訓練，並在多個 Test scale 上評估。
2. **ZELPHA**：將原本 linear classifier 換成 ZELPHA head（prototype-based + Lipschitz regularization），其餘訓練設定與 Vanilla 版本保持一致（learning rate, scheduler, augmentation 等）。

如此可以形成 **成對的實驗**：同一個 backbone + 同一個 seed 下，單純比較是否套用 ZELPHA head 對 robust accuracy 的影響。

### 3. 評估指標

* **Accuracy vs. scale factor**  
  * 分別計算每個 backbone / 每種訓練版本在各個 $s$ 上的 accuracy，畫出 accuracy–scale 曲線，比較誰的 degradation 最小。
* **Robust accuracy（主指標）**  
  * 對每張 test image，同時考慮所有 $s \in S$ 的預測結果：只要有任一個 scale 分錯，就視為失敗；只有在所有縮放版本皆正確時才算 1。  
  * 將這個指標對不同 backbone、不同 seed 取平均，可以得到整體的 robust accuracy 表現。
* **Seed-level pairwise comparison**  
  * 對於每個 backbone，對應的 Vanilla 與 ZELPHA 版本使用相同的 random seed，構成一對實驗；
  * 對所有 backbone × seed 的成對結果做差，並可進一步執行 paired t-test，檢驗 ZELPHA 是否在統計上顯著優於 baseline。
* **（選配）Calibration / Uncertainty**  
  * 在大的縮放 shift 下，衡量模型的校準程度（例如 ECE），觀察 ZELPHA 是否在信心分佈上也更合理。

### 4. Ablation studies

* **移除 Lipschitz regularization** → 只保留 prototype head，觀察 robust accuracy 與 margin 分佈的變化。
* **改回 linear classifier** → 保留 Lipschitz regularization，但不用 prototype-based classifier，改為標準 linear head，比較 margin 與 scale robustness。
* **只用 data augmentation、不加 ZELPHA head** → 僅使用更 aggressive 的 random resize / crop，檢查單純靠資料增強能否達到類似的縮放穩健性。
* **backbone family 分組分析** → 分別在 CNN 與 Vision Transformer 子集合中做 ablation，檢查 ZELPHA 在不同架構上的效果是否一致。

---

[1]: https://weegee.vision.ucmerced.edu/datasets/landuse.html?utm_source=chatgpt.com "UC Merced Land Use Dataset"  
[2]: https://github.com/phelber/EuroSAT?utm_source=chatgpt.com "EuroSAT: Land Use and Land Cover Classification with ..."  
[3]: https://arxiv.org/abs/1910.11093?utm_source=chatgpt.com "[1910.11093] Scale-Equivariant Steerable Networks"
[5]: https://papers.neurips.cc/paper/7640-lipschitz-regularity-of-deep-neural-networks-analysis-and-efficient-estimation.pdf?utm_source=chatgpt.com "Lipschitz regularity of deep neural networks"  
[6]: https://arxiv.org/abs/1709.00029?utm_source=chatgpt.com "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification"
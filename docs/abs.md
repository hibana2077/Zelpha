新增兩個圖邊權重的baseline供比較：

- RBF/Gaussian 親和力：`build_rbf_knn_graph(X, k, gamma=None)`，若未提供 gamma，使用鄰居距離的 median heuristic 自動估計。
- 共享近鄰 (SNN) 相似度：`build_snn_graph(X, k, sim="jaccard")`，可選 `sim="count"`；可透過 `include_mutual_only=True` 僅保留 mutual kNN。

這兩個介面已整合到 `training.build_graphs_for_point_cloud`，在 moons/circles 實驗中會自動加到報表中（鍵名：`rbf`、`snn_jaccard`）。

以下是一個「簡單但有深度」且縮寫為 **Zelpha** 的研究題目與完整大綱，貼合你給的方向（Graph construction／representation learning／representation graph theory）與想法（提出一種比 cosine 更表達力強的 edge 權重，圖建好後餵 GNN/GCN）：

# 題目（含縮寫）

**Zelpha：Path-Aware Hilbertian Affinity for Feature-Space Graph Construction**
（Zelpha：結合流形路徑感知的 Hilbert 空間親和力，用於特徵空間圖構建）

---

## 研究核心問題

在僅有節點特徵（或初步鄰近關係）的情況下，**如何定義一個比 cosine similarity 更有表達力、同時具理論保障（正定性、可收斂性）的 edge 權重函數**，使得後續以 GCN 等 GNN 進行學習時，能更忠實反映資料的流形幾何與多跳關係？（cosine 僅反映角度、對幅度不敏感，且非度量，對多跳關係無能為力。) ([nlp.stanford.edu][1])

---

## 研究目標

1. **提出 Zelpha 親和力**：將「特徵層級的 Hilbert 空間相似度」與「圖上熱核（heat kernel）的多跳擴散關係」結合為單一正定親和函數，用以取代傳統 cosine 權重。([alex.smola.org][2])
2. **理論證明** Zelpha 的**正定性**、對 cosine 與純擴散的**可退化性**（作為特例）、以及在迭代精煉時的**收斂條件**（非擴張映射假設）。([alex.smola.org][2])
3. **實證驗證**：在合成資料（Swiss roll／moons／circles）與 UCI ML datasets 上，檢證 Zelpha-GCN 相較 cosine-GCN 的穩定優勢。([scikit-learn.org][3])

---

## 核心貢獻（3）

1. **一體化邊權設計**：首次以**乘法結合**「RKHS 親和」與「熱核路徑感知」構成**單一正定（PD）**的 edge 權重，保留特徵相似與多跳可達性；並證明其 PD 性質。([alex.smola.org][2])
2. **理論到實作的可伸縮路線**：給出以 **Chebyshev/Krylov 近似**與 **Nyström／RFF** 的高效近似計算策略，使 Zelpha 能落地於中大型資料。([papers.nips.cc][4])
3. **可退化與可互操作**：Zelpha 在參數極限下**退化為 cosine 或純熱核**；同時與現有 diffusion-based 預處理（如 GDC）與 GNN 架構可相容整合。([nlp.stanford.edu][1])

---

## 創新

* **乘法融合（Hilbert × Heat）**：相較於傳統以 cosine（線性角度）或單獨擴散（PPR/heat）的做法，Zelpha 以**RKHS 的「核化 cosine」** × **圖熱核**的乘法結合，天然保持 PD，利於後續光譜方法／核方法分析。([CommLab][5])
* **理論上可證的正定與特例涵蓋**：Zelpha 在 α=1 時化為線性核之 cosine；在 α=0 時化為 heat-kernel 權重，統一多種常見相似度。([nlp.stanford.edu][1])

---

## 理論洞見

* **正定性（PD）閉包性**：PD 核的**乘法與加法**仍為 PD；而對角縮放（D^{-1/2} K D^{-1/2}）保持半正定，故以核矩陣做「相關係數化」的 RKHS-cosine 仍為 PD。([alex.smola.org][2])
* **熱核即擴散核**：exp(−tL) 為圖上的 diffusion kernel（t 為時間尺度），可攬多跳路徑，連結 manifold 幾何（與 diffusion maps 理念呼應）。([people.cs.uchicago.edu][6])
* **cosine 的不足**：cosine 僅看角度、對尺度不敏感，且常用的「1−cosine」非真正度量，限制其在索引／幾何保持上的理論工具；以 Zelpha 融入 heat-kernel 可彌補多跳與幾何面。([nlp.stanford.edu][1])

---

## 數學理論推演與（可納入論文的）證明綱要

令節點特徵 $x_i\in\mathbb{R}^d$，選一個特徵核 $\kappa(x_i,x_j)$（如 RBF/Laplacian 等），定義 **RKHS-cosine**

$$
h_{ij}=\frac{\kappa(x_i,x_j)}{\sqrt{\kappa(x_i,x_i)\kappa(x_j,x_j)}}\, .
$$

令初始圖的拉普拉斯 $L$（可由 $k$-NN 建*），其 **熱核** $K^{\text{heat}}_t=\exp(-tL)$。定義 **Zelpha 親和力**

$$
w_{ij}=\big(h_{ij}\big)^{\alpha}\cdot \big(K^{\text{heat}}_t\big)_{ij}^{\,1-\alpha},\qquad \alpha\in[0,1].
$$

* **命題1（PD）**：若 $\kappa$ 為 PD，則 $H=[h_{ij}]=D^{-1/2}KD^{-1/2}$ 為 PD；而 $K^{\text{heat}}_t$ 為 PD（diffusion kernel）。PD 核的乘積仍 PD，故 $W=[w_{ij}]$ PD。*證明要點*：核閉包性與相似變換保持半正定。([alex.smola.org][2])
* **命題2（特例）**：取線性核 $\kappa(x,y)=x^\top y$ 且 $\alpha=1$ 時，$w_{ij}$ 化為傳統 cosine；取 $\alpha=0$ 時，$w_{ij}=(K^{\text{heat}}_t)_{ij}$。([nlp.stanford.edu][1])
* **命題3（幾何保持）**：在隨機幾何圖與流形假設下，$K^{\text{heat}}_t$ 近似 geodesic-aware 的擴散距離，而 $H$ 捕捉高頻特徵相似；故 Zelpha 在多尺度 t、α 下兼顧局部特徵與多跳幾何。*論據*：diffusion maps 理論。([科學直接][7])
* **（選做）迭代精煉收斂**：定義 $W^{(s+1)}=(1-\mu)W^{(s)}+\mu\big(H\big)^{\alpha}\!\odot\!\big(\exp(-tL(W^{(s)}))\big)^{1-\alpha}$。在 $\mu$ 足夠小且 $\exp(-tL)$ 對 $W$ 的變動具 Lipschitz 界的條件下，此映射對 Frobenius 範數為非擴張，存在唯一不動點。（正文給出充分條件與證明綱要；附錄含技術細節與界。）參考 diffusion 方程與其近似解的光譜／Krylov 理論。([papers.nips.cc][4])

* 初始圖可用歐氏距／RBF／cosine 任一簡單準則產生疏圖，以利計算 $L$ 與 $K^{\text{heat}}_t$。

---

## 方法論（Zelpha-GCN 流程）

1. **特徵核與正規化**：選 $\kappa$（RBF/Laplacian/線性），形成 $H=D^{-1/2}KD^{-1/2}$。([CommLab][5])
2. **預圖與熱核**：由 $H$ 的 top-k 近鄰建初始圖求 $L$，計算 $K^{\text{heat}}_t=\exp(-tL)$（以 Chebyshev 展開或 Krylov 近似避免特徵分解）。([papers.nips.cc][4])
3. **Zelpha 權重**：$W=(H)^{\alpha}\odot (K^{\text{heat}}_t)^{1-\alpha}$；可再做 sparsify（每列保留 top-k）。
4. **下游學習**：以 $W$ 作為鄰接（或其歸一化）餵 **GCN**／其他 GNN；與 cosine-kNN 圖作對照。([arXiv][8])
5. **可伸縮性**：若 $n$ 大，對 $K$ 用 **Nyström** 或 **Random Fourier Features (RFF)** 近似；對 $\exp(-tL)$ 用多項式或局部解器。([stat.berkeley.edu][9])
6. **與 GDC 的關係**：Zelpha 的熱核項與 **Graph Diffusion Convolution**（使用 heat kernel 或 PPR）理念一致，但 Zelpha 同時保留 RKHS-cosine 成分並保證整體 PD。([arXiv][10])

---

## 預計使用資料集

* **合成**：Swiss roll（檢驗流形保持）、two moons、concentric circles（檢驗非線性可分）。([scikit-learn.org][3])
* **真實**：UCI ML datasets

---

## 與現有研究之區別

* **對比 cosine 圖**：cosine 僅角度、非度量，難表達多跳幾何；Zelpha 兼顧特徵相似與擴散幾何，且在 α 控制下可調兩者比重。([nlp.stanford.edu][1])
* **對比 diffusion-only（如 GDC／PPR／heat）**：Zelpha 不是單獨擴散，而是「**Hilbert × Diffusion**」的**正定融合**，對高頻特徵相似與多跳關係同時敏感。([arXiv][10])
* **對比學習圖結構（LDS 等）**：LDS 透過雙層最適化直接學圖，代價較高；Zelpha 為**閉式可近似**的權重設計，成本低、理論性質清楚，亦可作為 LDS 的初始化。([Proceedings of Machine Learning Research][12])

---

## Toy-experiment 設計

1. **Swiss roll 最近鄰圖質量**

   * 任取 $n=2{,}000$ 點，對比 **cosine-kNN** 與 **Zelpha-kNN** 的鄰居純度（近鄰落在同一流形折帶的比例）、圖切割能量與平均 geodesic-distortion。預期 Zelpha 在「不同捲層間的錯連」顯著降低。([scikit-learn.org][3])
2. **two moons／circles 節點分類**

   * 同一 GCN 架構，僅替換圖：cosine vs Zelpha。量測 Accuracy／Macro-F1 與對噪聲的穩健度（增加 0.1～0.3 高斯噪聲）。
3. **UCI ML datasets 分類**

   * 任務：ML classification。固定 GCN 超參數網格，掃描 $t\in\{0.1,0.5,1,2\},\alpha\in\{0.25,0.5,0.75\}$；回報平均±標準差與統計顯著性。
   * 資料集：sklearn.datasets.load_digits、sklearn.datasets.load_wine、sklearn.datasets.load_breast_cancer、Ionosphere。
4. **消融與可視化**

   * 僅用 $H$（RKHS-cosine）、僅用 $K^{\text{heat}}_t$、Zelpha 合成三組；再視覺化鄰接差分與特徵訊息的擴散範圍。
5. **時間／空間成本**

   * 報告 Krylov/多項式近似與 Nyström/RFF 的加速比與精度折衝。

---

## 形式化定義（放論文方法段首）

**Zelpha 親和力**：對任意兩點 $i,j$，

$$
\boxed{\;w_{ij}=\left(\frac{\kappa(x_i,x_j)}{\sqrt{\kappa(x_i,x_i)\kappa(x_j,x_j)}}\right)^{\alpha}\cdot \left[\exp(-tL)\right]_{ij}^{\,1-\alpha}\;}
$$

其中 $\kappa$ 為 PD 核（RBF 等）、$L$ 為初始疏圖的圖拉普拉斯、$t>0$ 為擴散時間、$\alpha\in[0,1]$。
**特例**：$\alpha=1$ 且 $\kappa$ 線性 ⇒ cosine；$\alpha=0$ ⇒ 純 heat-kernel 權重。([nlp.stanford.edu][1])

---

## 與 GNN 的接合

* 將 $W$ 正規化為 $\hat A$（含自環），照 **GCN** 的一階近似做前向傳播；或與 GAT、GraphSAGE 等替換 plug-in。([arXiv][8])
* 亦可與 **GDC** 思想結合：先以 Zelpha 得基底圖，再做個性化 PageRank/heat 的稀疏化修剪。([arXiv][10])

---

## 參考依據（關鍵論點的文獻支撐）

* **GCN 基礎**：Kipf & Welling（2016/2017）。([arXiv][14])
* **Diffusion/Heat kernel**：Kondor & Lafferty（2002）；Diffusion Maps（Coifman & Lafon, 2006）。([people.cs.uchicago.edu][6])
* **核方法與 PD 閉包性**：Schölkopf & Smola（2002）；Hofmann/Schölkopf/Smola（2008）。([CommLab][5])
* **cosine 侷限**：IIR 教科書；cosine 非度量（triangle inequality 不成立）之分析。([nlp.stanford.edu][1])
* **GDC 與擴散改良圖學習**：Gasteiger et al.（NeurIPS 2019）。([NeurIPS Papers][15])
* **資料集**：Cora/Citeseer/PubMed（Sen et al. 摘載於多篇回顧）；OGB（Hu et al., 2020）。([PMC][11])
* **近似計算**：Nyström（Williams & Seeger 等後續工作）、RFF（Rahimi & Recht）；快速圖擴散解器。([stat.berkeley.edu][9])

---

如果你要把這個題目真的「寫成論文」，我可以直接幫你展開**摘要、相關研究、方法、理論證明草稿、實驗設定表格與圖例模板**，或先幫你做 **Zelpha-GCN** 的實驗計畫與結果表格骨架。你想先從哪一塊動手？

[1]: https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf "Introduction to Information Retrieval - Stanford NLP"
[2]: https://alex.smola.org/papers/2008/HofSchSmo08.pdf "Kernel methods in machine learning - Alex Smola"
[3]: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_swiss_roll.html "make_swiss_roll"
[4]: https://papers.nips.cc/paper_files/paper/2024/file/0506ad3d1bcc8398a920db9340f27fe4-Paper-Conference.pdf "Faster Local Solvers for Graph Diffusion Equations"
[5]: https://mcube.lab.nycu.edu.tw/~cfung/docs/books/scholkopf2002learning_with_kernels.pdf "Learning with Kernels"
[6]: https://people.cs.uchicago.edu/~risi/papers/diffusion-kernels.pdf "Diffusion Kernels on Graphs and Other Discrete Structures"
[7]: https://www.sciencedirect.com/science/article/pii/S1063520306000546/pdf?md5=3a36a8a3d85b69535dde24c2f397f835&pid=1-s2.0-S1063520306000546-main.pdf&utm_source=chatgpt.com "Diffusion maps"
[8]: https://arxiv.org/pdf/1609.02907 "Semi-Supervised Classification with Graph Convolutional ..."
[9]: https://www.stat.berkeley.edu/~mmahoney/pubs/nystrom-jmlr16.pdf "Revisiting the Nystr m Method for Improved Large-scale ..."
[10]: https://arxiv.org/pdf/1911.05485 "[PDF] Diffusion Improves Graph Learning - arXiv"
[11]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6989613/ "Multi-Task Network Representation Learning - PMC"
[12]: https://proceedings.mlr.press/v97/franceschi19a/franceschi19a.pdf "Learning Discrete Structures for Graph Neural Networks"
[13]: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html "make_moons — scikit-learn 1.7.2 documentation"
[14]: https://arxiv.org/abs/1609.02907 "Semi-Supervised Classification with Graph Convolutional Networks"
[15]: https://papers.neurips.cc/paper/9490-diffusion-improves-graph-learning.pdf "[PDF] Diffusion Improves Graph Learning - NIPS"

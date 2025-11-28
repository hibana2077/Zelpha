下面這份可以當成你實作 **ZELPHA** 的「開發規格 + 理論對照表」，照著做就不太容易歪掉。
我會用「**必做檢查點**」標出跟理論一致性相關的地方。

---

## 0. 文件目標與整體架構

**目標**：給一份從「資料 → 模型 → Loss → 訓練 → 評估」的實作說明，並在每個關鍵步驟標出它對應到的理論假設（scale-equivariant、Lipschitz、prototype margin），讓你在：

* 實作時知道 **哪行 code 是為了哪個理論條件**。
* 實驗時知道 **該檢查哪些統計量來驗證理論**。

**ZELPHA 高階流程**

1. 載入 UC Merced（選配：EuroSAT）資料集。([weegee.vision.ucmerced.edu][1])
2. 建立 train / val / test split，並為 test 產生不同縮放版本的影像。
3. 模型：

   * Shared CNN backbone（淺層、易分析）
   * 多尺度輸入 + scale pooling（近似 scale-equivariant）([arXiv][2])
   * 對所有 conv / linear 層做 spectral normalization（控制 Lipschitz 常數）([atom-101.github.io][3])
   * Prototype-based classifier。
4. 損失：Cross-entropy + prototype 聚合 / 分離正則 + Lipschitz 正則。
5. 訓練 loop：warm-up → full loss → ablation。
6. 評估：accuracy vs. scale、robust accuracy、margin 統計。

---

## 1. 資料與前處理

### 1.1 Dataset 選擇

**主要：UC Merced Land Use**

* 21 類 land-use 場景，每類 100 張，共 2100 張。
* 多數影像為 256×256 RGB，空間解析度約 0.3 m/pixel。([weegee.vision.ucmerced.edu][1])

**選配：EuroSAT-RGB**

* 基於 Sentinel-2，10 類土地利用／覆蓋，約 27k 張影像。
* 有完整 13 band 版本與 RGB 子集（你可以先用 RGB）。([GitHub][4])

### 1.2 資料夾 / Loader 設計

建議檔案結構：

```text
data/
  uc_merced/
    train/
    val/
    test/
      s1.00/    # 原始 scale
      s0.70/
      s0.85/
      s1.15/
      s1.30/
  eurosat_rgb/ (optional)
```

**步驟：**

1. 下載 UC Merced 原始檔，依類別放資料夾（或直接用 TFDS / TorchGeo loader）。([TensorFlow][5])
2. 隨機 per-class 打散，做 60%/20%/20%（或 50/20/30）split，確保每類分布均衡。
3. 對 test set 的每張圖 (x)：

   * 產生縮放 factor 集合 (S = {0.70, 0.85, 1.00, 1.15, 1.30})。
   * 對每個 (s)：

     * 先以 factor (s) 做中心縮放（scale + center crop），
     * 再 resize 回 256×256。
   * 存成不同子資料夾（上面範例結構）。

**必做檢查點 A（縮放正確性）**

對同一張影像，畫出幾個縮放版本疊在一起（例如使用透明度）或直接肉眼看，確認：

* 中心物件有被放大/縮小，但大致仍在畫面中央。
* 大縮放時（1.3）沒有切掉關鍵物件；小縮放時（0.7）不會出現太多 padding 或背景。

---

## 2. 資料增強與縮放 pipeline

### 2.1 Train Transform

為了維持理論分析乾淨度，建議：

* **基本增強**：Random horizontal flip、輕微 color jitter。
* **Scale augmentation**（可選）：

  * 小幅 random resize crop（例如 scale ∈ [0.9, 1.1]），不要跟 test scale 重疊太多，避免「完全 seen scales」。
* Normalize：以 dataset 的 mean / std（可自行計算）。

### 2.2 Test Transform

* 除了各 scale 子資料夾的縮放本身，不再做隨機增強。
* 同樣做 normalize（與 train 一致）。

---

## 3. 模型設計：ZELPHA 架構

### 3.1 High-level API 設計

```python
class ZelphaModel(nn.Module):
    def __init__(self, num_classes, feature_dim, scales):
        self.backbone = ScalePooledBackbone(feature_dim, scales)
        self.classifier = PrototypeHead(num_classes, feature_dim)
```

* `scales`: 訓練時使用的內部多尺度（例如 `{0.8, 1.0, 1.2}`）。
* `feature_dim`: 最終 embedding 維度（比如 128 或 256）。

---

### 3.2 ScalePooledBackbone：近似 scale-equivariant feature extractor

你可以選擇「簡化版」實作，理論上是對 scale 群做一個有限子群近似：

1. 定義一個 **shared CNN** (g_\theta)：

   * 幾層 conv + BN + ReLU + pooling，最後 global average pooling → feature 向量 (z \in \mathbb{R}^d)。
   * 所有 conv / linear 之後都會加 spectral norm（之後講）。([atom-101.github.io][3])

2. 對於輸入影像 (x)，在 forward 時做：

```python
def forward(self, x):  # x: [B, C, H, W]
    multi_scale_feats = []
    for s in self.scales:
        x_s = resize_and_center_crop(x, scale=s)  # differentiable
        z_s = g_theta(x_s)  # shared CNN
        multi_scale_feats.append(z_s)  # [B, D]
    # scale pooling
    Z = torch.stack(multi_scale_feats, dim=1)  # [B, S, D]
    z_pooled = Z.mean(dim=1)  # or max-pooling over scale
    return z_pooled  # [B, D]
```

這個做法對 scale 的處理等價於：**把 scale 當成一個離散 group 元素集合 S，並對它做 group pooling**，跟 scale-equivariant CNN 的核心精神一致，只是少了 kernel resizing 的細緻部分。([arXiv][2])

**必做檢查點 B（scale pooling 是否有用）**

訓練一個 **不做 scale pooling、只用 s=1 的 backbone** 做為對照，檢查：

* 在正常 test set（s=1.0）兩者 accuracy 類似。
* 在 scale shift test set 中，ZELPHA 的 accuracy vs. s 曲線下降較慢。

---

## 4. Lipschitz 約束實作：Spectral Normalization

### 4.1 理論回顧（對應）

Spectral normalization 把每個 weight matrix (W) 的最大奇異值 (\sigma_{\max}(W)) 限制在某個值以下，從而控制每一層的 Lipschitz 常數；多層組合後，可以得到整個網路 Lipschitz 常數的上界。([atom-101.github.io][3])

### 4.2 PyTorch 實作規則

對所有 conv / linear 層都做：

```python
from torch.nn.utils import spectral_norm

self.conv1 = spectral_norm(nn.Conv2d(...))
self.fc = spectral_norm(nn.Linear(...))
```

**注意事項：**

1. Spectral norm 實作一般是用 power iteration 近似；要確保：

   * 設定 `n_power_iterations` 不太小（預設 1，有時會不夠穩定）。
2. 避免使用會改變 Lipschitz 常數的非線性（例如某些強放大 scaling 的 op）；ReLU / LeakyReLU 皆為 1-Lipschitz 沒問題。
3. Pooling（max/avg）在適當定義的 norm 下也維持 Lipschitz 常數 ≤ 1。

**必做檢查點 C（估計實際 Lipschitz）**

在實作完成後，對一些 batch 做「有限差分」估計：

* 取 (x)，加上小擾動 (\Delta x)（例如高斯噪音或小幅縮放 s=0.98/1.02）。
* 計算 (|f(x+\Delta x) - f(x)| / |\Delta x|)。
* 把這個值在多個樣本上取最大，作為實際的粗估 Lipschitz 常數。
* 檢查有沒有遠大於你設定的理論上界（例如希望 < 5 或 < 10）。

---

## 5. Prototype-based Classifier 設計

### 5.1 參數形式

```python
class PrototypeHead(nn.Module):
    def __init__(self, num_classes, feature_dim):
        self.mu = nn.Parameter(torch.randn(num_classes, feature_dim))
```

* (\mu_c) 是類別 (c) 的 prototype。

### 5.2 前向傳遞與 logit 計算

給定輸入 feature (z = f_\theta(x))：

1. 計算距離矩陣：
   [
   D_{i,c} = |z_i - \mu_c|^2
   ]
2. 轉成 logit：
   [
   \text{logit}*{i,c} = - D*{i,c}
   ]
3. Softmax → 預測分佈 (p(y\mid x))。

### 5.3 Prototype Loss 與 Margin 正則

Loss 組成：

1. **分類 cross-entropy**：

   * 使用 logit。

2. **類內聚合**（pull-to-prototype）：
   [
   \mathcal{L}*{\text{intra}} = \frac{1}{N}\sum_i |z_i - \mu*{y_i}|^2
   ]

3. **類間分離**（push-apart）：
   設 margin 超參數 (m > 0)，
   [
   \mathcal{L}*{\text{inter}} = \sum*{c\neq c'} \max(0, m - |\mu_c - \mu_{c'}|)^2
   ]

總 prototype 正則：

[
\mathcal{L}*{\text{proto}} = \lambda*{\text{intra}}\mathcal{L}*{\text{intra}} + \lambda*{\text{inter}}\mathcal{L}_{\text{inter}}
]

**必做檢查點 D（margin 分布）**

定期在 validation set 上計算每個樣本的 margin：

[
m(x) = \min_{c\ne y} \big(|z - \mu_c| - |z - \mu_y|\big)
]

* 畫出 margin histogram。
* 問題：若大多數 margin 接近 0 或負值，代表 prototype 約束太弱或 Lipschitz 太大。

---

## 6. 總損失與理論對應

總 loss：

[
\mathcal{L} = \mathcal{L}_{\text{CE}}

* \beta, \mathcal{L}_{\text{proto}}
* \gamma, \mathcal{L}_{\text{Lip}}
  ]

其中：

* (\mathcal{L}_{\text{CE}})：跨 entropy。
* (\mathcal{L}_{\text{proto}})：上一節定義。
* (\mathcal{L}_{\text{Lip}})：可以使用簡單 proxy，例如：

  * 對每層 spectral norm 估計值 (\hat{\sigma}*l) 的 penalty：
    [
    \mathcal{L}*{\text{Lip}} = \sum_l \max(0, \hat{\sigma}_l - K)^2
    ]
    其中 (K) 是你希望的 per-layer 上界。

**對應理論：**

* margin 在縮放擾動下的變化量 (\le 2L\delta_s)。

  * 大 margin + 小 Lipschitz ⇒ 對一定範圍的縮放仍維持正確分類。
* 你在實作時透過 `\mathcal{L}_{\text{proto}}` 放大 margin、透過 `\mathcal{L}_{\text{Lip}}` 壓低 Lipschitz，對應到這個命題的條件。

---

## 7. 訓練流程 Step-by-Step

### 7.1 初始化

1. 建立 `ZelphaModel(num_classes=21, feature_dim=128, scales={0.8,1.0,1.2})`。
2. 用 Kaiming init 初始化 CNN；prototype 可用小 Gaussian，或在 warm-up 之後重新設為 class mean。
3. Optimizer：AdamW / SGD + momentum。
4. Scheduler：Cosine decay 或 StepLR 均可。

### 7.2 Warm-up 階段（建議）

**Epoch 0~5：**

* 只用 (\mathcal{L}_{\text{CE}})，暫時關掉 `\mathcal{L}_{\text{proto}}` 與 `\mathcal{L}_{\text{Lip}}`（或給很小係數）。
* 每 N 個 iteration，計算每類 feature 平均，更新 prototype：

  ```python
  with torch.no_grad():
      mu_c.data = running_mean_of_class_c_features
  ```
* 目的：先讓 backbone 學一個合理 embedding，再鎖定 prototype 初值。

### 7.3 Full Training 階段

**Epoch > 5：**

1. 啟用 full loss：

   * (\beta) 由 0 緩慢升到 target（例如 0.1~0.5）。
   * (\gamma) 小但不為 0（例如 0.001~0.01），避免 Lipschitz 約束太強導致學不動。
2. 每個 iteration：

   * 取 batch (x)（只用 train scale，或少量 random scale augmentation）。
   * 前向：`z = backbone(x)` → `logits, proto_loss, lip_loss`。
   * 組合 loss，backprop。
3. 每隔若干 step：

   * 記錄、plot：train loss、val accuracy、average margin。
   * 粗估 Lipschitz 常數（有限差分）。

**必做檢查點 E（訓練穩定性）**

* 如果加入 spectral norm + Lipschitz penalty 後，loss 不降或 accuracy 卡住：

  * 先把 (\gamma) 設回 0，只用 spectral norm（不加 extra penalty）。
  * 或調整 power iteration 次數、learning rate。

---

## 8. 評估與實驗設計

### 8.1 Accuracy vs. Scale

對 test set：

1. 分別在 s = 0.70, 0.85, 1.00, 1.15, 1.30 的 test subset 上跑 inference。
2. 計算每個 scale 的 accuracy。
3. 畫出曲線（x 軸 scale、y 軸 accuracy）。

**對照：**

* Vanilla CNN baseline
* ZELPHA without spectral norm
* ZELPHA without prototype loss
* Full ZELPHA

理想情況：full ZELPHA 在 scale 偏離 1.0 時 accuracy 下降較慢。

### 8.2 Robust Accuracy（per-image robustness）

對於每張原始 test image (x)：

* 如果在所有 scale (s \in S) 上都分類正確，計為 robust=1，否則 0。
* robust accuracy = robust 樣本比例。

這個指標更接近「理論上的對所有縮放皆穩定」。

### 8.3 Margin vs. Scale

對部分樣本（或全部）：

1. 對每個 scale 的 feature (z_s)，計算 margin (m(x_s))。
2. 看 margin 如何隨 scale 變化。

如果你實作的 Lipschitz 約束有效，**margin 應該在小幅縮放下不會劇烈變化**，與理論 bound 相符合。

---

## 9. 常見錯誤與 Debug 建議

1. **Spectral norm 沒有真的套用到所有層**

   * 檢查 `model.named_modules()`，確認所有 conv / linear 都經過 `spectral_norm` 包裝。
2. **Scale pooling 維度搞錯**

   * 確認 `multi_scale_feats` 中每個元素 shape 都是 `[B, D]`，最後 stack → `[B, S, D]` 再沿 scale 維度 pooling。
3. **Prototype 沒有好好更新**

   * 如果發現某些 class 的 prototype 幾乎不動，可以在 warm-up 期間直接用 class mean 重設。
4. **Margin 全部接近 0 或負值**

   * 增加 (\lambda_{\text{inter}}) 讓 prototypes 分散。
   * 降低 Lipschitz penalty 或 backbone 深度，避免 features 被壓得太扁。
5. **Scale test set 產生錯誤**

   * 確認縮放是「先 scale、再中心 crop / pad、最後 resize 回 256」，不是直接 random crop，否則 scale 變化不純粹。

---

## 10. 小結：對齊理論的簡短 Checklist

你在實作時可以直接拿這段當「完成度檢查表」：

1. ✅ UC Merced / EuroSAT 資料載入正確，train/val/test split 每類均衡。([weegee.vision.ucmerced.edu][1])
2. ✅ Test set 建立多個 scale 版本，縮放操作一致且可視覺確認。
3. ✅ Backbone 為淺層 CNN + multi-scale shared forward + scale pooling。
4. ✅ 所有 conv / linear 層都有 spectral normalization。([atom-101.github.io][3])
5. ✅ Prototype head 用距離作為 logit，並有 intra / inter 兩種正則。
6. ✅ 訓練過程有 warm-up → full loss，並記錄 margin 分佈與 Lipschitz 粗估。
7. ✅ 實驗報告包含：

   * Accuracy vs. scale（多個模型對照）
   * Robust accuracy
   * Margin vs. scale 分析

如果你願意，下一步我可以幫你把 **實際的 PyTorch 專案 skeleton** 寫出來（例如 `models.py`, `datasets.py`, `train.py`, `eval.py` 的架構與關鍵函式簽名），讓你直接在上面補完細節。

[1]: https://weegee.vision.ucmerced.edu/datasets/landuse.html?utm_source=chatgpt.com "UC Merced Land Use Dataset"
[2]: https://arxiv.org/abs/1910.11093?utm_source=chatgpt.com "[1910.11093] Scale-Equivariant Steerable Networks"
[3]: https://atom-101.github.io/blog/posts/2022-03-18-spectral-norm.html?utm_source=chatgpt.com "Spectral Normalization - Atmadeep Banerjee"
[4]: https://github.com/phelber/EuroSAT?utm_source=chatgpt.com "EuroSAT: Land Use and Land Cover Classification with ..."
[5]: https://www.tensorflow.org/datasets/catalog/uc_merced?utm_source=chatgpt.com "uc_merced | TensorFlow Datasets"

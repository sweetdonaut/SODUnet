# SODUnet Training & Testing SOP

## 1. 環境需求

- Python 3.10+, 使用 `uv` 管理環境
- GPU: NVIDIA (>=16GB VRAM for bs=16, patch_size=224)
- 所有指令在專案根目錄下執行: `cd SODUnet/src_core`

---

## 2. 資料集格式

### 2.1 目錄結構

```
data_root/
├── train/
│   ├── images/          # 訓練影像
│   │   ├── train_0000.png
│   │   ├── train_0001.png
│   │   └── ...
│   └── labels/          # YOLO 分割標註
│       ├── train_0000.txt
│       ├── train_0001.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    ├── labels/
    └── test_defects.csv   # 測試集 defect metadata (FROC 分析用)
```

### 2.2 影像格式

- 灰階 (grayscale) PNG, uint8
- 所有影像需為**相同尺寸** (e.g. 1536x1536)
- 無缺陷 (normal) 影像直接放入 images/, 對應 label 為空檔案

### 2.3 標註格式 (YOLO Segmentation)

每個 `.txt` 與同名影像對應, 每行一個缺陷:

```
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

- `class_id`: 固定為 `0`
- 座標為 **normalized** (0~1), 相對於影像寬高
- 至少 3 個頂點 (6 個數值)
- 無缺陷影像: label 檔案為空 (0 bytes)

範例 (12 頂點多邊形):
```
0 0.8687 0.1602 0.8681 0.1626 0.8663 0.1643 0.8639 0.1650 0.8615 0.1643 0.8598 0.1626 0.8591 0.1602 0.8598 0.1578 0.8615 0.1560 0.8639 0.1554 0.8663 0.1560 0.8681 0.1578
```

### 2.4 測試集 CSV (可選, FROC/Review Efficiency 分析用)

`test_defects.csv` 欄位:

| 欄位 | 說明 |
|------|------|
| filename | 影像檔名 (e.g. `test_0000.png`) |
| rawx | 缺陷中心 X (pixel) |
| rawy | 缺陷中心 Y (pixel) |
| dsnr | 缺陷 DSNR (signal-to-noise ratio) |

範例:
```csv
filename,rawx,rawy,dsnr
test_0000.png,1318,550,6.454
test_0001.png,350,924,2.482
```

- 只包含有缺陷的影像, normal 影像不列入
- 未提供 CSV 時, inference 仍可執行但不會產生 FROC 曲線

---

## 3. 訓練

### 3.1 基本指令

```bash
cd src_core

uv run python trainer.py \
  --task_name my_experiment \
  --data_root /path/to/data_root \
  --model_file model_v3 \
  --encoder_attention \
  --patch_size 224 \
  --bs 16 \
  --lr 0.001 \
  --epochs 100 \
  --eval_interval 5 \
  --seed 42
```

### 3.2 關鍵參數說明

| 參數 | 預設 | 說明 |
|------|------|------|
| `--task_name` | (必填) | 實驗名稱, 輸出存到 `../outputs/<task_name>/` |
| `--data_root` | (必填) | 資料集根目錄 (含 train/, valid/) |
| `--model_file` | model_v1 | 模型: `model_v1`, `model_v2`, `model_v3`, `model_v4` |
| `--encoder_attention` | off | 啟用 CoordAttention (v3 建議開啟) |
| `--patch_size` | 128 | 切 patch 大小, 建議 224 |
| `--bs` | (必填) | Batch size, 16 for v3+224 |
| `--lr` | (必填) | Learning rate, 建議 0.001 |
| `--epochs` | (必填) | 訓練 epoch 數 |
| `--eval_interval` | 1 | 每 N epochs 做一次 validation |
| `--bg_ratio` | -1 | 背景 patch 比例. -1=全部, 2=每個 defect patch 搭配 2 個背景 |
| `--seed` | None | 隨機種子 |
| `--gamma_start` | 1.0 | Focal loss gamma 起始值 |
| `--gamma_end` | 3.0 | Focal loss gamma 結束值 (cosine schedule) |

### 3.3 推薦配置 (v3)

```bash
uv run python trainer.py \
  --task_name v3_production \
  --data_root /path/to/data_root \
  --model_file model_v3 \
  --encoder_attention \
  --patch_size 224 \
  --bs 16 \
  --lr 0.001 \
  --epochs 100 \
  --eval_interval 5 \
  --seed 42
```

### 3.4 輸出結構

```
outputs/<task_name>/
├── weights/
│   └── best.pth          # 最佳模型 checkpoint
├── results.csv            # 每次 eval 的指標記錄
└── training_log.txt       # 完整訓練設定記錄
```

`results.csv` 欄位: `epoch, focal_loss, wass_loss, lr, gamma, iAUROC, pAUROC, R@1FP, R@5FP, LocErr@1, LocErr@5`

---

## 4. 測試 (Inference)

Inference 支援三種模式, 依據提供的參數自動切換:

| 模式 | labels_dir | csv_path | 輸出 |
|------|-----------|----------|------|
| **生產模式** | - | - | heatmaps + spots.csv + image_scores.csv |
| **座標評估模式** | - | 提供 | 以上 + Image AUROC + FROC + Review Efficiency |
| **Mask 評估模式** | 提供 | - | 以上 + Image AUROC + Pixel AUROC |
| **完整模式** | 提供 | 提供 | 全部指標 |

### 4.1 生產模式 (無 GT)

只有影像, 無標註:

```bash
cd src_core

uv run python inference.py \
  --task_name production_run \
  --model_path ../outputs/my_experiment/weights/best.pth \
  --model_file model_v3 \
  --images_dir /path/to/test/images
```

### 4.2 座標評估模式 (有 defect CSV, 無 pixel mask)

最常見的生產環境評估場景 — 只需提供缺陷座標 CSV:

```bash
uv run python inference.py \
  --task_name coord_eval \
  --model_path ../outputs/my_experiment/weights/best.pth \
  --model_file model_v3 \
  --images_dir /path/to/test/images \
  --csv_path /path/to/test/test_defects.csv
```

輸出: Image AUROC + FROC + Review Efficiency (無 Pixel AUROC)

### 4.3 Mask 評估模式 (有 YOLO labels)

提供 YOLO polygon labels, 計算 pixel-level AUROC:

```bash
uv run python inference.py \
  --task_name mask_eval \
  --model_path ../outputs/my_experiment/weights/best.pth \
  --model_file model_v3 \
  --images_dir /path/to/test/images \
  --labels_dir /path/to/test/labels
```

### 4.4 完整模式 (YOLO labels + defect CSV)

所有指標:

```bash
uv run python inference.py \
  --task_name full_eval \
  --model_path ../outputs/my_experiment/weights/best.pth \
  --model_file model_v3 \
  --images_dir /path/to/test/images \
  --labels_dir /path/to/test/labels \
  --csv_path /path/to/test/test_defects.csv
```

### 4.4 參數說明

| 參數 | 預設 | 說明 |
|------|------|------|
| `--task_name` | (必填) | 輸出目錄名稱 |
| `--model_path` | auto | 模型權重路徑 |
| `--model_file` | model_v1 | 需與訓練一致 |
| `--base_channels` | 64 | 需與訓練一致 |
| `--images_dir` | (必填) | 測試影像目錄 |
| `--labels_dir` | (可選) | 標註目錄, 提供則計算 AUROC |
| `--csv_path` | (可選) | defect CSV, 提供則產生 FROC + Review Efficiency |
| `--threshold` | 0.1 | Heatmap 閾值 (spot 提取用) |

### 4.5 輸出結構

```
outputs/<task_name>/
├── spots.csv                       # [必出] 所有偵測點位 (filename, x, y, score)
├── image_scores.csv                # [必出] 每張影像分數 (filename, max_score, num_spots)
├── heatmaps/                       # [必出] 每張影像的 heatmap 視覺化
│   ├── test_0000.png
│   └── ...
├── froc_all.png                    # [需 csv_path] FROC 曲線
├── froc_dsnr_low.png               # [需 csv_path] FROC (DSNR < 3.5)
├── froc_dsnr_high.png              # [需 csv_path] FROC (DSNR >= 3.5)
├── review_efficiency_all.png       # [需 csv_path] Review efficiency
├── review_efficiency_dsnr_low.png
└── review_efficiency_dsnr_high.png
```

### 4.6 輸出指標

終端輸出範例:
```
Spots: 452 total -> spots.csv
Image scores: 205 images -> image_scores.csv

Image-level AUROC: 0.9740              # 需 labels_dir
Pixel-level AUROC: 0.9973              # 需 labels_dir + 有缺陷影像
FROC Curve (All): Recall@1FP=0.9600    # 需 csv_path
```

- **Image-level AUROC**: 以影像為單位的缺陷偵測能力 (需 labels)
- **Pixel-level AUROC**: 像素級分割品質 (需 labels, 且只計算有缺陷的影像)
- **Recall@1FP**: 在平均每張影像 1 個 false positive 時的召回率 (需 CSV)
- **Spots**: 總偵測點數 (越少表示 false positive 越少)
- **image_scores.csv**: 可用來自行設定閾值判斷影像是否有缺陷

---

## 5. 快速驗證範例

用專案內建合成資料集進行完整流程測試:

```bash
cd src_core

# 訓練 (約 30 分鐘, 100 epochs)
uv run python trainer.py \
  --task_name quicktest \
  --data_root ../synthetic_data/png \
  --model_file model_v3 \
  --encoder_attention \
  --patch_size 224 \
  --bs 16 --lr 0.001 --epochs 100 \
  --eval_interval 5 --seed 42

# 測試
uv run python inference.py \
  --task_name quicktest_eval \
  --model_path ../outputs/quicktest/weights/best.pth \
  --model_file model_v3 \
  --images_dir ../synthetic_data/png/test/images \
  --labels_dir ../synthetic_data/png/test/labels \
  --csv_path ../synthetic_data/png/test/test_defects.csv
```

預期結果: Image AUROC > 0.95, Pixel AUROC > 0.99, R@1FP > 0.93

---

## 6. 生產環境注意事項

1. **model_file 和 encoder_attention 必須訓練/測試一致**, 否則模型載入會失敗
2. **影像尺寸必須一致** — 訓練集所有影像需為同一尺寸, 測試時也建議相同
3. **patch_size 影響偵測精度** — 224 為目前最佳, 較小的 patch 會降低感受野
4. **threshold 調整** — 預設 0.1, 調高可減少 false positive 但降低召回率
5. **bg_ratio** — 資料量大時可用 `--bg_ratio 2` 加速訓練 (約 15x), 性能損失 < 2%
6. **GPU 記憶體不足時** — 降 bs 到 8, 或降 patch_size 到 128

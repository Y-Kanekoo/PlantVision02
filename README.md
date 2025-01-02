pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install xformers --index-url https://download.pytorch.org/whl/cu124
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu124

transformers         4.46.3
wandb                0.19.1
bitsandbytes            0.45.0
numpy                   1.26.3
scikit-learn            1.6.0
pandas                  2.2.3

### 設計書

---

#### **目的**

Llama3.2 11B を使用し、PlantDoc データセットを活用して農業分野における画像分類タスクの精度向上を目指します。このプロジェクトは卒業論文に使用されるため、実験手法や結果を再現可能で明確に記述する必要があります。

---

#### **使用するリソースと技術**

1. **モデル**

   - [Llama3.2 11B](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) を基盤モデルとして利用。
   - **量子化**: 4bit量子化を採用することで、メモリ使用量を削減し、NVIDIA 4070 Ti のメモリ制約に対応する。
     - 利用ライブラリ: `bitsandbytes`
     - 量子化後のモデルサイズ: 理論上1/8 (約5.5GB)

2. **データセット**

   - PlantDoc データセット ([GitHubリンク](https://github.com/pratikkayal/PlantDoc-Dataset.git))。

   - 元データの詳細については[関連論文](https://arxiv.org/pdf/1911.10317)を参照。

   - **データセット概要**:

     - 総画像数: 2,598枚。
     - クラス数: 27クラス（17の病害クラス、10の健康状態）。
     - 対象植物種: 13種類。
     - クラス分布の偏りがあるため、分布を確認し必要に応じて対応する。
     - 自然環境下で撮影された多様な背景や照明条件を含む。

   - **分割方法**:

     - 学習用: 70%
     - 検証用: 20%
     - テスト用: 10%

   - **前処理必要事項**:

     - 画像のリサイズ（モデル入力サイズに適合）。
     - データの正規化（平均0、標準偏差1など）。
     - 必要に応じてノイズ除去やラベルの確認。

3. **開発環境**

   - **GPU**: NVIDIA 4070 Ti
   - **ランタイム環境**: ローカル環境（Windows）
   - **CUDA バージョン**: 12.4
   - **Python**: >= 3.9
   - **乱数シード**: 2025（再現性確保のため固定）
   - **ライブラリおよびフレームワーク**:
     - PyTorch (CUDA 対応)
       ```bash
       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
       ```
     - Transformers
     - Weights & Biases (W&B)
     - `bitsandbytes`（量子化用ライブラリ）
     - `xformers`
       ```bash
       pip install xformers --index-url https://download.pytorch.org/whl/cu124
       ```
     - その他必要ライブラリはrequirements.txtで管理。

4. **公式リソース**

   - ファインチューニングの参考として以下のGoogle Colabを活用。 [公式Google Colab](https://colab.research.google.com/drive/1j0N4XTY1zXXy7mPAhOC1_gMYZ2F2EBlk?usp=sharing#scrollTo=HpUdg2c7eajr)

5. **バージョン管理**

   - GitHub を利用してコードや設計書を共有。

---

#### **手順詳細**

##### **1. 環境構築**

- 必要なソフトウェアをインストール。
  ```bash
  pip install --upgrade pip
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  pip install xformers --index-url https://download.pytorch.org/whl/cu124
  pip install transformers unsloth wandb bitsandbytes
  ```
- CUDA 12.4 のインストール。
  - NVIDIA 4070 Ti に適したドライババージョンを確認。

##### **2. データ準備**

1. GitHub リポジトリからデータセットをクローン。
   ```bash
   git clone https://github.com/pratikkayal/PlantDoc-Dataset.git
   ```

2. 無効なファイル名の修正（Linux環境で実施）。
   ```bash
   sudo apt update
   sudo apt install rename
   find . -name '*\?*' -exec rename 's/\?/_/g' {} +
   find . -name '*&*' -exec rename 's/&/_/g' {} +
   ```

3. 必要なディレクトリに移動。
   ```bash
   mv PlantDoc-Dataset/* data/raw/
   ```

4. データセットの分割。
   - データ数が足りない場合はデータ拡張を検討（初期段階では拡張しない）。
   - Python スクリプトで以下の割合に分割:
     - 学習用: 70%
     - 検証用: 20%
     - テスト用: 10%
   ```python
   import os
   import random
   from shutil import copy2

   def split_dataset(dataset_dir, output_dir, train_ratio=0.7, val_ratio=0.2, seed=2025):
       random.seed(seed)
       classes = os.listdir(dataset_dir)

       for cls in classes:
           cls_path = os.path.join(dataset_dir, cls)
           images = os.listdir(cls_path)
           random.shuffle(images)

           train_split = int(len(images) * train_ratio)
           val_split = int(len(images) * (train_ratio + val_ratio))

           train_images = images[:train_split]
           val_images = images[train_split:val_split]
           test_images = images[val_split:]

           for split, split_images in zip(["train", "val", "test"], [train_images, val_images, test_images]):
               split_dir = os.path.join(output_dir, split, cls)
               os.makedirs(split_dir, exist_ok=True)
               for img in split_images:
                   copy2(os.path.join(cls_path, img), split_dir)

   split_dataset("data/raw", "data/processed")
   ```

##### **3. モデルの準備**

1. Hugging Face Hub から Llama3.2 11B モデルをダウンロード。
   ```python
   from transformers import AutoModel, AutoTokenizer

   model = AutoModel.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct", load_in_4bit=True, device_map="auto")
   tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
   ```
2. unsloth を用いた高速化設定。
   ```python
   from unsloth import optimize_model

   model = optimize_model(model)
   ```
3. W&B の初期設定。
   ```python
   import wandb

   wandb.init(project="llama3.2-plantdoc-finetuning")
   ```

##### **4. ファインチューニング**

1. データローダーの作成。
   ```python
   from torch.utils.data import DataLoader
   from datasets import load_dataset

   dataset = load_dataset("path/to/Split-Dataset")
   train_loader = DataLoader(dataset['train'], batch_size=16, shuffle=True)
   valid_loader = DataLoader(dataset['validation'], batch_size=16)
   ```
2. モデルのトレーニング設定。
   ```python
   from transformers import Trainer, TrainingArguments

   training_args = TrainingArguments(
       output_dir="./results",
       num_train_epochs=5,
       per_device_train_batch_size=16,
       per_device_eval_batch_size=16,
       warmup_steps=500,
       weight_decay=0.01,
       logging_dir="./logs",
       logging_steps=10,
       evaluation_strategy="steps",
   )

   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=dataset['train'],
       eval_dataset=dataset['validation']
   )

   trainer.train()
   ```
3. 学習過程の可視化。
   ```python
   wandb.watch(model, log="all")
   ```

##### **5. 結果の評価**

1. テストデータを用いてモデルの精度を確認。
   ```python
   results = trainer.evaluate()
   print(results)
   ```

##### **6. モデルの保存と共有**

1. モデルを Hugging Face Hub にアップロード。
   ```python
   model.push_to_hub("your-username/Llama3.2-PlantDoc")
   ```
2. GitHub リポジトリにコードやドキュメントをプッシュ。
   ```bash
   git add .
   git commit -m "Add fine-tuning scripts and results"
   git push origin main
   ```

---

#### **ファイル構成**

```plaintext
project-root/
├── data/
│   ├── raw/                  # オリジナルデータセット
│   ├── processed/            # 分割後のデータセット
│   ├── PlantDoc-Dataset/     # GitHubからクローンしたデータセット
├── scripts/
│   ├── split_dataset.py      # データセット分割スクリプト
│   ├── train.py              # ファインチューニングスクリプト
│   ├── evaluate.py           # 評価スクリプト
├── models/
│   ├── llama3.2-11b/         # 学習済みモデル
├── results/
│   ├── logs/             　# W&B用ログ
│   ├── metrics/              # 学習結果指標
├── notebooks/
│   ├── eda.ipynb             # データ探索用ノートブック
├── requirements.txt          # 必要ライブラリ一覧
├── README.md                 # プロジェクト概要
├── .gitignore                # Git用/

#### **スケジュール**
1. **環境構築**: 1日
2. **データ準備**: 2日
3. **モデル準備**: 1日
4. **ファインチューニング**: 5日
5. **評価と結果の分析**: 2日
6. **ドキュメント作成と論文執筆**: 3日

---

#### **注意点**
1. GPU メモリの使用量に注意し、バッチサイズを調整。
2. 学習中のログを適切に保存し、トラブル発生時に対処可能な状態を維持。
3. データセットの偏りやモデルのバイアスに留意し、適切な評価を行う。

```


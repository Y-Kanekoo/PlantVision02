from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
    Trainer,
    TrainingArguments
)
from datasets import load_from_disk
import wandb
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from multiprocess import freeze_support

# メイン処理を関数化


def main():
    # W&Bの初期化
    wandb.init(project="vit-plantdoc-finetuning")

    # モデルと特徴抽出器のロード
    model_name = "google/vit-base-patch16-224-in21k"
    classes = [
        "Apple leaf", "Apple rust leaf", "Apple Scab Leaf", "Bell_pepper leaf",
        "Bell_pepper leaf spot", "Blueberry leaf", "Cherry leaf", "Corn Gray leaf spot",
        "Corn leaf blight", "Corn rust leaf", "grape leaf", "grape leaf black rot",
        "Peach leaf", "Potato leaf early blight", "Potato leaf late blight", "Raspberry leaf",
        "Soyabean leaf", "Squash Powdery mildew leaf", "Strawberry leaf", "Tomato Early blight leaf",
        "Tomato leaf", "Tomato leaf bacterial spot", "Tomato leaf late blight", "Tomato leaf mosaic virus",
        "Tomato leaf yellow virus", "Tomato mold leaf", "Tomato Septoria leaf spot",
        "Tomato two spotted spider mites leaf"
    ]

    label2id = {label: i for i, label in enumerate(classes)}
    id2label = {i: label for i, label in enumerate(classes)}

    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=len(classes),
        id2label=id2label,
        label2id=label2id
    )

    image_processor = AutoImageProcessor.from_pretrained(model_name)

    # データセットをロード
    dataset_path = "data/processed"
    datasets = load_from_disk(dataset_path)

    # 入力データの前処理（画像とラベルの処理）

    def preprocess_images(example):
        try:
            # 画像の読み込みと変換
            image = Image.open(example["image"]).convert("RGB")

            # 画像の前処理を強化
            encoding = image_processor(
                images=image,
                return_tensors="pt",
                do_resize=True,
                size={"height": 224, "width": 224},
                do_normalize=True
            )

            # ラベルの処理
            if example["label"] not in label2id:
                print(f"Warning: Unknown label '{example['label']}' found")
                return {"pixel_values": None, "labels": None}

            encoding = {
                "pixel_values": encoding["pixel_values"][0],
                "labels": label2id[example["label"]]
            }
            return encoding

        except Exception as e:
            print(f"Error processing image {example['image']}: {str(e)}")
            return {"pixel_values": None, "labels": None}

    # データセット処理時の進捗表示を改善
    print("Processing datasets...")
    processed_datasets = datasets.map(
        preprocess_images,
        batched=False,
        remove_columns=datasets["train"].column_names,
        desc="Processing images",
        num_proc=2 if __name__ == '__main__' else None  # Windows環境での並列処理対応
    ).filter(lambda x: x["labels"] is not None)

    # データセットの統計情報を表示
    print("\nProcessed dataset sizes:")
    for split in processed_datasets:
        print(f"{split}: {len(processed_datasets[split])} examples")

    # クラスの分布を確認
    print("\nClass distribution in training set:")
    label_counts = {}
    for example in processed_datasets["train"]:
        label = id2label[example["labels"]]
        label_counts[label] = label_counts.get(label, 0) + 1

    for label, count in sorted(label_counts.items()):
        print(f"{label}: {count} images")

    # トレーニング引数の設定
    training_args = TrainingArguments(
        # 基本設定
        output_dir="results",          # モデルの出力ディレクトリ
        num_train_epochs=30,            # 学習エポック数

        # バッチサイズ設定
        per_device_train_batch_size=16,  # 訓練時の1デバイスあたりのバッチサイズ
        per_device_eval_batch_size=16,  # 評価時の1デバイスあたりのバッチサイズ
        gradient_accumulation_steps=2,  # 勾配を蓄積するステップ数

        # 評価と保存の戦略
        eval_strategy="steps",         # ステップ数ベースで評価を実行
        evaluation_strategy="steps",   # 評価戦略
        eval_steps=500,               # 何ステップごとに評価を行うか
        save_strategy="steps",        # モデル保存の戦略
        save_steps=500,              # 何ステップごとにモデルを保存するか
        save_total_limit=30,          # 保存するチェックポイントの最大数

        # モデル選択設定
        load_best_model_at_end=True,   # 学習終了時に最良モデルをロード
        metric_for_best_model="eval_accuracy",  # モデル選択の評価指標
        greater_is_better=True,        # 評価指標は大きいほど良い

        # ログ設定
        logging_dir="logs",           # ログの出力ディレクトリ
        logging_steps=30,             # 何ステップごとにログを記録するか

        # 最適化設定
        learning_rate=2e-5,           # 学習率
        weight_decay=0.01,            # Weight decay（L2正則化係数）

        # 高速化・効率化設定
        fp16=True,                    # 16ビット精度での学習を有効化
        dataloader_num_workers=4,     # Windows環境での安定性を考慮して調整
        dataloader_pin_memory=True,   # GPUへのデータ転送を高速化
    )

    def compute_metrics(eval_pred):
        """評価指標を計算する関数

        Args:
            eval_pred: モデルの予測結果と正解ラベルのタプル

        Returns:
            dict: 各種評価指標を含む辞書
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)  # 確率分布から最も確率の高いクラスを選択

        # 各種評価指標の計算
        accuracy = accuracy_score(labels, predictions)  # 正解率
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions,
            average='weighted',  # クラスの重みづけ平均を使用
            zero_division=1      # ゼロ除算時は1を返す
        )

        return {
            'accuracy': accuracy,    # 正解率（正しく分類された割合）
            'precision': precision,  # 適合率（陽性と予測したものの中で実際に陽性だった割合）
            'recall': recall,       # 再現率（実際の陽性の中で陽性と予測できた割合）
            'f1': f1                # F1スコア（適合率と再現率の調和平均）
        }

    # トレーナーの設定
    trainer = Trainer(
        model=model,                          # 学習対象のモデル
        args=training_args,                   # 学習設定
        train_dataset=processed_datasets["train"],  # 訓練データ
        eval_dataset=processed_datasets["val"],     # 検証データ
        processing_class=image_processor,           # 画像の前処理クラス
        compute_metrics=compute_metrics             # 評価指標の計算関数
    )

    # トレーニングの実行
    if __name__ == "__main__":
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--dry-run", action="store_true",
                            help="実際の学習を行わずに設定のテストのみを実行")
        args = parser.parse_args()

        if not args.dry_run:
            # トレーニングの実行
            trainer.train()

            # モデルと設定の保存
            final_model_path = "results/vit-finetuned-final"
            trainer.save_model(final_model_path)          # モデルの保存
            image_processor.save_pretrained(final_model_path)  # 前処理設定の保存

            # 最終評価の実行
            val_results = trainer.evaluate(
                eval_dataset=processed_datasets["val"])   # 検証データでの評価
            test_results = trainer.evaluate(
                eval_dataset=processed_datasets["test"])  # テストデータでの評価

            print("\nValidation Results:")
            print(val_results)
            print("\nTest Results:")
            print(test_results)

            # 結果をWandBに記録（実験管理）
            wandb.log({
                "final_validation": val_results,
                "final_test": test_results
            })

            print(f"\nTraining completed. Model saved to {final_model_path}")


if __name__ == "__main__":
    freeze_support()  # Windows環境でのマルチプロセス対応
    main()  # メイン処理の実行

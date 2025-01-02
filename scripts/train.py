import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import wandb
from unsloth import optimize_model

# WandBプロジェクトの初期化
wandb.init(project="llama3_2_plantdoc")

# モデルとトークナイザの準備
print("Loading the model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision-Instruct",  # モデル名
    load_in_4bit=True,  # 4bit量子化を適用
    device_map="auto"   # デバイスを自動設定（GPUが利用可能ならGPUへ）
)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision-Instruct")

# モデルの最適化（Unslothを利用）
print("Optimizing the model...")
model = optimize_model(model)

# データセットのロード
print("Loading the dataset...")
data_dir = "data/processed"  # 処理済みデータのディレクトリ
dataset = load_dataset("imagefolder", data_dir=data_dir)

# トークナイザを使用してデータセットを前処理


def preprocess_function(examples):
    # テキストデータのトークナイズ
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)


print("Tokenizing the dataset...")
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# トレーニングと評価用データセットの分割
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["val"]

# トレーニング引数の設定
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./results",                  # 結果保存先
    overwrite_output_dir=True,              # 出力ディレクトリを上書き
    evaluation_strategy="steps",            # ステップごとに評価
    eval_steps=100,                         # 評価間隔
    logging_dir="./results/logs",           # ログの保存先
    logging_steps=50,                       # ログ出力間隔
    save_steps=500,                         # モデルを保存する間隔
    save_total_limit=2,                     # 保存するモデルの最大数
    per_device_train_batch_size=8,          # トレーニング時のバッチサイズ
    per_device_eval_batch_size=8,           # 評価時のバッチサイズ
    num_train_epochs=5,                     # エポック数
    learning_rate=5e-5,                     # 学習率
    warmup_steps=500,                       # ウォームアップステップ数
    weight_decay=0.01,                      # 重み減衰率
    fp16=True,                              # 半精度（FP16）トレーニング
    report_to="wandb",                      # WandBにログを送信
)

# Trainerの初期化
print("Initializing the Trainer...")
trainer = Trainer(
    model=model,                            # トレーニング対象モデル
    args=training_args,                     # トレーニング引数
    train_dataset=train_dataset,            # トレーニングデータ
    eval_dataset=eval_dataset,              # 検証データ
)

# トレーニングの実行
print("Starting training...")
trainer.train()

# モデルの保存
print("Saving the model...")
model.save_pretrained("./results/model")
tokenizer.save_pretrained("./results/model")

# トレーニング完了のログを記録
print("Training complete! Model saved in ./results/model.")
wandb.finish()

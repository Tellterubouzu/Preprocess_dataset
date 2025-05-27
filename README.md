# Preprocess Dataset

タイ語コーパスデータを前処理し、OpenAI apiを使用してテキストを生成・変換するツールです。

## 機能

- Hugging Faceのデータセット（pythainlp/thaigov-v2-corpus-31032024）からデータを読み込み
- OpenAI APIを使用したテキスト生成
- マルチスレッド処理による効率的なAPI呼び出し
- キャッシュ機能による重複処理の防止
- トークン使用量の追跡とコスト計算

## 必要条件

- Python 3.x
- OpenAI APIキー
- 必要なPythonパッケージ（requirements.txtに記載）

## インストール

```bash
git clone https://github.com/Tellterubouzu/preprocess_dataset.git
cd preprocess_dataset
pip install -r requirements.txt
```

## 使用方法

1. `.env`ファイルを作成し、OpenAI APIキーを設定します：
```
OPENAI_API_KEY=your_api_key_here
```

2. スクリプトを実行します：
```bash
python src/main.py --output_file thai_text_processed.txt --num_lines 10000
```

### 主なオプション

- `--output_file`: 出力ファイル名（デフォルト: thai_text_processed.txt）
- `--model`: 使用するOpenAIモデル（デフォルト: gpt-4o-mini）
- `--num_lines`: 生成する行数（デフォルト: 10000）
- `--batch_size`: バッチサイズ（デフォルト: 100）
- `--workers`: 同時API呼び出し数（デフォルト: 100）
- `--limit`: トークン使用量の制限（デフォルト: 100）

## ライセンス

このプロジェクトは [MIT License](LICENSE) の下で公開しています。

Copyright (c) 2024 Tellterubouzu

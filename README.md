# filter-person

Keep only single-person, upper-body photos in a folder — removes images with no face, multiple faces/people, face close-ups, or tiny subjects.

> **Note:** This tool's console output, prompts, and folder picker dialog are in Japanese.

## Features

- Filters `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, and `.tiff` images in a folder.
- Supports folder and file paths containing non-ASCII (e.g., Japanese) characters.
- Displays a breakdown of deletion reasons and sample file paths, then asks for confirmation (`y`/`N`) before deleting anything.
- Images that fail to load or infer are kept and never deleted.

## How it works

Each image is evaluated using two bundled detection models:

1. **Face detection**: YuNet via OpenCV `FaceDetectorYN` (`face_detection_yunet_2026may.onnx`, default confidence `0.8`).
2. **Person detection**: Ultralytics YOLOv8n (`yolov8n.pt`, COCO `person` class, confidence `0.25` on CPU).

An image is **kept** only if all of the following rules hold:

- **Exactly 1 face** is detected.
- **Exactly 1 person** is detected, and that person bounding box contains the face center.
- **Minimum person size**: the person bounding box height is at least `35%` of the image height by default (`--min-person-height 0.35`).
- **Upper-body framing**: the person bounding box extends below the bottom of the face by at least `1.0` face height.

Images that fail any rule are queued for deletion under one of the reported reasons (no face, multiple faces, no person detected, multiple people, face close-up / no chest area, or person too small). Images that fail to load or raise an inference error are always kept.

## Requirements

- Windows (primarily, via `run.bat`)
- Python 3.10+
- `numpy>=1.26`
- `opencv-python>=5.0.0`
- `ultralytics>=8.4,<9` (installs PyTorch)

## Installation

Open a command prompt in the repository folder and create a virtual environment:

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install --cache-dir .pip-cache -r requirements.txt
```

Because `ultralytics` installs PyTorch, the initial setup requires time and disk space.

## Usage

### Windows (`run.bat`)

- **Drag & drop** a folder onto `run.bat`, or
- **Double-click** `run.bat` to open a folder picker dialog.

`run.bat` automatically uses `.venv\Scripts\python.exe` if `.venv` exists (otherwise `python`).

### Command line

```bat
python fp.py <folder> [--recursive] [--confidence 0.8] [--min-person-height 0.35]
```

| Option | Default | Description |
| --- | --- | --- |
| `folder` | *(folder picker)* | Target folder containing images. Opens a folder selection dialog if omitted. |
| `--recursive` | `False` | Process subfolders recursively. Without this flag, only images directly inside `folder` are processed. |
| `--confidence` | `0.8` | Minimum face detection confidence (`0.0`–`1.0`). |
| `--min-person-height` | `0.35` | Minimum person bounding box height as a ratio of image height (`0.0`–`1.0`). |

> [!WARNING]
> **Permanent deletion:** Confirmed deletions (`y` or `yes`) permanently remove files from disk (`unlink`) and do **not** send them to the Recycle Bin. Always back up your folder before running.

## Limitations

- Filtering is based on estimates from face and general-purpose person detection models rather than direct upper-body/chest classification; false positives and false negatives are possible.
- Small subjects or heavily occluded people may be missed or misclassified.

## Bundled models

- `face_detection_yunet_2026may.onnx` — YuNet from OpenCV Zoo (MIT License; see `YUNET_LICENSE.txt`).
- `yolov8n.pt` — Ultralytics YOLOv8n (AGPL-3.0).

## License

This project is licensed under AGPL-3.0 — see LICENSE. It depends on Ultralytics YOLOv8 (AGPL-3.0). YuNet model: MIT (YUNET_LICENSE.txt).

---

## 日本語

フォルダ内の画像から「顔が1人・人物が1人・人物の高さが画像の35%以上・顔の下に上半身が続いている」画像だけを残し、顔なし／複数の顔／複数人／顔アップ／人物が小さすぎる画像を削除候補として抽出するツールです（コンソール出力・確認プロンプト・フォルダ選択画面は日本語です）。

### 初回準備

Python 3.10以降が必要です。リポジトリのフォルダでコマンドプロンプトを開き、仮想環境へ依存関係（OpenCV 5以降・Ultralytics・PyTorch）をインストールします（初回は時間とディスク容量が必要です）。

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install --cache-dir .pip-cache -r requirements.txt
```

### 使い方

1. 画像フォルダを `run.bat` にドラッグ＆ドロップするか、`run.bat` をダブルクリックしてフォルダを選択します。
2. 削除予定数と理由別の件数・例を確認します。
3. 問題なければ `y` を入力して削除します（`n` または Enter でキャンセル）。

コマンドラインから実行する場合（`.venv\Scripts\python.exe` でも実行可能）：

```bat
python fp.py "画像フォルダのパス" [--recursive] [--confidence 0.8] [--min-person-height 0.35]
```

- 通常は選択フォルダ直下の画像だけを処理します。サブフォルダも処理する場合は `--recursive` を指定してください。
- `--confidence`（既定値 `0.8`）：顔検出の最低信頼度（`0.0`～`1.0`）。
- `--min-person-height`（既定値 `0.35`）：画像高さに対する人物検出枠の最低割合（`0.0`～`1.0`）。値を上げるほど大きく写った人物だけ残ります。

> [!WARNING]
> `y` で実行した削除はごみ箱を経由せず**完全に削除**されます。実行前に必ずバックアップを取ってください。

### 判定と制限事項

- YuNet（顔検出）と YOLOv8n（COCO `person` クラス）の検出結果から推定しており、胸部そのものを直接識別しているわけではありません。誤判定や、小さな人物・大きな遮蔽の見逃しが起こり得ます。
- 読み込み・推論エラーが発生した画像は削除せず残します。

### 同梱モデル・依存関係・ライセンス

- YuNet（OpenCV Zoo）: `face_detection_yunet_2026may.onnx`（MIT、`YUNET_LICENSE.txt`）
- YOLOv8n: `yolov8n.pt`（AGPL-3.0）
- Python 依存関係: `requirements.txt`
- 本プロジェクトのライセンスは AGPL-3.0 です（`LICENSE` 参照）。

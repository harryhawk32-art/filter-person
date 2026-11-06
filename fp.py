from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "face_detection_yunet_2026may.onnx"
PERSON_MODEL_PATH = APP_DIR / "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.8
PERSON_CONFIDENCE = 0.25
MIN_BODY_BELOW_FACE_HEIGHTS = 1.0
DEFAULT_MIN_PERSON_HEIGHT = 0.35


def select_folder() -> str:
    """引数なしで起動した場合にフォルダ選択画面を表示する。"""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        return filedialog.askdirectory(title="処理する画像フォルダを選択してください")
    finally:
        root.destroy()


def load_image(path: Path) -> np.ndarray | None:
    """日本語を含むパスから画像を読み込む。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def detect_faces(detector: cv2.FaceDetectorYN, image: np.ndarray) -> np.ndarray | None:
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    return faces


def load_person_detector():
    """Ultralyticsの設定と重みキャッシュをプロジェクト内に限定して読み込む。"""
    os.environ.setdefault("YOLO_CONFIG_DIR", str(APP_DIR / ".ultralytics"))
    os.environ.setdefault("TORCH_HOME", str(APP_DIR / ".torch"))
    from ultralytics import YOLO

    return YOLO(str(PERSON_MODEL_PATH))


def detect_people(detector, image: np.ndarray) -> np.ndarray:
    results = detector.predict(
        source=image,
        classes=[0],
        conf=PERSON_CONFIDENCE,
        imgsz=640,
        device="cpu",
        verbose=False,
    )
    boxes = results[0].boxes
    return np.empty((0, 4), dtype=np.float32) if boxes is None else boxes.xyxy.cpu().numpy()


def person_frame_issue(
    face: np.ndarray,
    people: np.ndarray,
    image_shape: tuple[int, ...],
    min_person_height: float,
) -> str | None:
    """人物サイズと顔下の検出範囲を見て、削除理由を返す。"""
    image_height = image_shape[0]
    x, y, width, height = face[:4]
    center_x = x + width / 2
    center_y = y + height / 2
    face_bottom = y + height

    for person_x1, person_y1, person_x2, person_y2 in people:
        contains_face = (
            person_x1 <= center_x <= person_x2
            and person_y1 <= center_y <= person_y2
        )
        if not contains_face:
            continue
        person_height_ratio = (person_y2 - person_y1) / image_height
        if person_height_ratio < min_person_height:
            return "人物が小さすぎる"
        body_below_face = (person_y2 - face_bottom) >= height * MIN_BODY_BELOW_FACE_HEIGHTS
        if body_below_face:
            return None
        return "顔アップ／胸元なし"
    return "人物検出なし"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YuNetで顔の人数を判定し、顔が0人または2人以上の画像を削除します。"
    )
    parser.add_argument(
        "folder", nargs="?", help="処理する画像フォルダ（run.batへドラッグ&ドロップ可）"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="選択フォルダ内のサブフォルダも再帰的に処理する",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"顔として数える最低信頼度 0.0～1.0（既定値: {DEFAULT_CONFIDENCE}）",
    )
    parser.add_argument(
        "--min-person-height",
        type=float,
        default=DEFAULT_MIN_PERSON_HEIGHT,
        help=f"画像高さに対する人物検出枠の最低割合（既定値: {DEFAULT_MIN_PERSON_HEIGHT}）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        print("エラー: --confidence は0.0～1.0で指定してください。")
        return 2
    if not 0.0 <= args.min_person_height <= 1.0:
        print("エラー: --min-person-height は0.0～1.0で指定してください。")
        return 2

    folder = args.folder or select_folder()
    if not folder:
        print("フォルダが選択されませんでした。終了します。")
        return 0

    image_dir = Path(folder).expanduser().resolve()
    if not image_dir.is_dir():
        print(f"エラー: フォルダが見つかりません: {image_dir}")
        return 2
    for model_path in (MODEL_PATH, PERSON_MODEL_PATH):
        if not model_path.is_file():
            print(f"エラー: 検出モデルが見つかりません: {model_path}")
            return 2

    image_paths = image_dir.rglob("*") if args.recursive else image_dir.iterdir()
    images = sorted(
        (path for path in image_paths if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: str(path.relative_to(image_dir)).casefold(),
    )
    if not images:
        print(f"画像がありません: {image_dir}")
        return 0

    try:
        face_detector = cv2.FaceDetectorYN_create(
            str(MODEL_PATH), "", (320, 320), args.confidence, 0.3, 5000
        )
        person_detector = load_person_detector()
    except Exception as exc:
        print(f"検出モデルを初期化できませんでした: {exc}")
        print("requirements.txt の依存関係をプロジェクト内の .venv にインストールしてください。")
        return 2

    to_delete: list[tuple[Path, str]] = []
    errors: list[tuple[Path, str]] = []
    reason_counts = {
        "顔なし": 0,
        "複数の顔": 0,
        "人物検出なし": 0,
        "複数人": 0,
        "顔アップ／胸元なし": 0,
        "人物が小さすぎる": 0,
    }
    print(f"対象: {image_dir}")
    print(
        f"画像 {len(images)} 枚を確認中...（顔1人・人物1人・人物高が画像の{args.min_person_height:.0%}以上）"
    )

    for index, path in enumerate(images, start=1):
        try:
            image = load_image(path)
            if image is None:
                errors.append((path, "画像を読み込めません"))
                continue
            faces = detect_faces(face_detector, image)
            people = detect_people(person_detector, image)
            face_count = 0 if faces is None else len(faces)
            reason = None
            if face_count == 0:
                reason = "顔なし"
            elif face_count > 1:
                reason = "複数の顔"
            elif len(people) == 0:
                reason = "人物検出なし"
            elif len(people) > 1:
                reason = "複数人"
            else:
                reason = person_frame_issue(
                    faces[0], people, image.shape, args.min_person_height
                )
            if reason is not None:
                to_delete.append((path, reason))
                reason_counts[reason] += 1
        except Exception as exc:
            errors.append((path, str(exc)))
        if index % 100 == 0 or index == len(images):
            print(f"  判定済み: {index}/{len(images)}")

    print("\n判定結果:")
    print(f"  残す（顔1人・胸元まで検出・人物1人）: {len(images) - len(to_delete) - len(errors)} 枚")
    print(f"  削除対象合計: {len(to_delete)} 枚")
    for reason, count in reason_counts.items():
        print(f"    {reason}: {count} 枚")
    print(f"  エラー（削除せず残す）: {len(errors)} 枚")
    if errors:
        print("\n読み込み・判定エラー（安全のため削除しません）:")
        for path, message in errors[:20]:
            print(f"  [エラー] {path.relative_to(image_dir)}: {message}")
        if len(errors) > 20:
            print(f"  ...ほか {len(errors) - 20} 件")

    if not to_delete:
        print("削除対象はありません。")
        return 0

    print("\n削除対象の例:")
    for path, reason in to_delete[:10]:
        print(f"  [{reason}] {path.relative_to(image_dir)}")
    if len(to_delete) > 10:
        print(f"  ...ほか {len(to_delete) - 10} 枚")

    answer = input(f"\n削除対象 {len(to_delete)} 枚を完全に削除します。続行しますか？ (y/N): ").strip().lower()
    if answer not in {"y", "yes"}:
        print("キャンセルしました。画像は削除していません。")
        return 0

    deleted = 0
    for path, reason in to_delete:
        try:
            path.unlink()
            deleted += 1
            print(f"[削除: {reason}] {path.relative_to(image_dir)}")
        except OSError as exc:
            errors.append((path, f"削除できません: {exc}"))
            print(f"[削除エラー] {path.relative_to(image_dir)}: {exc}")

    print(f"\n完了: {deleted} 枚を削除しました。エラー: {len(errors)} 枚")
    return 0


if __name__ == "__main__":
    sys.exit(main())

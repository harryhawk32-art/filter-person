import os
import cv2
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

def select_folder():
    """GUI（ファイル選択ウィンドウ）でフォルダを選択する"""
    root = tk.Tk()
    root.withdraw() # 余分なメインウィンドウを隠す
    root.attributes('-topmost', True) # ウィンドウを最前面に表示
    
    # ダイアログを開いてユーザーにフォルダを選択させる
    folder_path = filedialog.askdirectory(title="画像が保存されているフォルダを選択してください")
    return folder_path

def main():
    print("========================================")
    print(" 人物の【顔】が写っていない画像を削除するツール ")
    print("========================================\n")
    
    # 1. フォルダの選択（ダイアログを表示）
    print("フォルダ選択画面を開いています...（別ウィンドウを探してください）")
    folder_path = select_folder()
    
    if not folder_path:
        print("フォルダが選択されなかったため、終了します。")
        return
        
    img_dir = Path(folder_path)
    print(f"選択されたフォルダ: {img_dir}\n")
    
    # 2. 実行の最終確認
    print("※ 注意: このプログラムは「顔」が検出されなかった画像を【直接削除】します。元に戻せません。")
    ans = input("実行してもよろしいですか？ (y/Enterで実行 / nでキャンセル): ").strip().lower()
    if ans == 'n':
        print("キャンセルしました。")
        return

    # 3. OpenCV 標準の顔検出AIの初期化（追加インストール不要で動くものに切り替え）
    print("\n顔検出システム（OpenCV標準）を準備しています...")
    # 正面顔用の検出器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # 横顔用の検出器
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]

    if not images:
        print(f"画像が見つかりませんでした: {img_dir}")
        return

    print(f"\n合計 {len(images)} 枚の画像から「顔」を探し、無いものを削除します...")
    
    deleted_count = 0
    error_count = 0

    for img_path in images:
        try:
            # OpenCVの日本語ファイルパス対応の読み込み
            img_array = np.fromfile(str(img_path), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                continue
                
            # メモリ効率と精度のためグレースケール（白黒）に変換
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 顔検出を実行 (正面顔、横顔の両方を探す)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            
            # もし「真正面の顔」も「横顔」も一つも検出されなかったら
            if len(faces) == 0 and len(profiles) == 0:
                print(f"[削除] 顔が見つかりませんでした: {img_path.name}")
                img_path.unlink()  # ファイルを削除
                deleted_count += 1
                
        except Exception as e:
            print(f"エラー発生 ({img_path.name}): {e}")
            error_count += 1

    print(f"\n■ 完了: {deleted_count} 枚の画像を削除しました！")
    if error_count > 0:
        print(f"   (エラーで処理できなかった画像: {error_count}枚)")

if __name__ == "__main__":
    main()

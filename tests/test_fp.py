from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fp


def make_face(x: float, y: float, w: float, h: float) -> np.ndarray:
    """OpenCV FaceDetectorYN 出力形式 (15列) の顔配列を1件作る。"""
    face = np.zeros(15, dtype=np.float32)
    face[:4] = [x, y, w, h]
    return face


def make_faces(*boxes: tuple[float, float, float, float]) -> np.ndarray:
    """OpenCV FaceDetectorYN 出力形式 (N, 15) の配列を作る。"""
    if not boxes:
        return np.empty((0, 15), dtype=np.float32)
    return np.stack([make_face(*box) for box in boxes], axis=0)


def make_people(*boxes: tuple[float, float, float, float]) -> np.ndarray:
    """xyxy 形式 (N, 4) の人物配列を作る。"""
    if not boxes:
        return np.empty((0, 4), dtype=np.float32)
    return np.array(boxes, dtype=np.float32)


class TestPersonFrameIssue:
    IMAGE_SHAPE = (1000, 800, 3)

    def test_kept_valid_upper_body(self) -> None:
        face = make_face(350, 100, 100, 100)
        people = make_people((250, 80, 550, 580))  # height=500 (0.5 >= 0.35), below_face=380 >= 100
        assert fp.person_frame_issue(face, people, self.IMAGE_SHAPE, 0.35) is None

    def test_person_too_small(self) -> None:
        face = make_face(380, 100, 40, 40)
        people = make_people((350, 90, 450, 340))  # height=250 (0.25 < 0.35)
        assert fp.person_frame_issue(face, people, self.IMAGE_SHAPE, 0.35) == "人物が小さすぎる"

    def test_face_closeup_no_chest(self) -> None:
        face = make_face(300, 100, 200, 200)
        people = make_people((200, 50, 600, 450))  # height=400 (0.4 >= 0.35), below_face=150 < 200
        assert fp.person_frame_issue(face, people, self.IMAGE_SHAPE, 0.35) == "顔アップ／胸元なし"

    def test_no_person_contains_face_center(self) -> None:
        face = make_face(100, 100, 100, 100)  # center=(150, 150)
        people = make_people((300, 100, 600, 600))
        assert fp.person_frame_issue(face, people, self.IMAGE_SHAPE, 0.35) == "人物検出なし"


class TestClassifyImage:
    IMAGE_SHAPE = (1000, 800, 3)

    def test_no_faces_empty_array(self) -> None:
        faces = make_faces()
        people = make_people()
        assert fp.classify_image(faces, people, self.IMAGE_SHAPE, 0.35) == "顔なし"

    def test_no_faces_none(self) -> None:
        people = make_people()
        assert fp.classify_image(None, people, self.IMAGE_SHAPE, 0.35) == "顔なし"

    def test_multiple_faces(self) -> None:
        faces = make_faces((100, 100, 80, 80), (400, 100, 80, 80))
        people = make_people()
        assert fp.classify_image(faces, people, self.IMAGE_SHAPE, 0.35) == "複数の顔"

    def test_one_face_zero_people(self) -> None:
        faces = make_faces((350, 100, 100, 100))
        people = make_people()
        assert fp.classify_image(faces, people, self.IMAGE_SHAPE, 0.35) == "人物検出なし"

    def test_one_face_two_people(self) -> None:
        faces = make_faces((350, 100, 100, 100))
        people = make_people((250, 80, 550, 580), (50, 100, 200, 600))
        assert fp.classify_image(faces, people, self.IMAGE_SHAPE, 0.35) == "複数人"

    def test_valid_single_person(self) -> None:
        faces = make_faces((350, 100, 100, 100))
        people = make_people((250, 80, 550, 580))
        assert fp.classify_image(faces, people, self.IMAGE_SHAPE, 0.35) is None


class TestMoveDestination:
    def test_preserves_relative_path(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "src"
        move_to = tmp_path / "rejected"
        src_file = image_dir / "sub" / "nested" / "photo.jpg"
        src_file.parent.mkdir(parents=True)
        src_file.write_bytes(b"img")

        dest = fp.move_destination(src_file, image_dir, move_to)
        assert dest == move_to / "sub" / "nested" / "photo.jpg"

    def test_collision_appends_suffix(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "src"
        move_to = tmp_path / "rejected"
        src_file = image_dir / "sub" / "photo.jpg"
        src_file.parent.mkdir(parents=True)
        src_file.write_bytes(b"img")

        existing = move_to / "sub" / "photo.jpg"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"old")

        dest_1 = fp.move_destination(src_file, image_dir, move_to)
        assert dest_1 == move_to / "sub" / "photo_1.jpg"

        dest_1.write_bytes(b"old1")
        dest_2 = fp.move_destination(src_file, image_dir, move_to)
        assert dest_2 == move_to / "sub" / "photo_2.jpg"

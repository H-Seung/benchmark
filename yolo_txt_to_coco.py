"""
YOLO 라벨(txt) → COCO GT
"""
import argparse, json
from pathlib import Path
import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def read_yolo(label_path: Path):
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c, cx, cy, w, h = line.split()[:5]
        out.append((int(c), float(cx), float(cy), float(w), float(h)))
    return out

def img_to_label_path(img_path: Path, images_root: Path, labels_root: Path) -> Path:
    # images_root 기준 상대경로를 유지하면서 labels_root로 매핑
    rel = img_path.relative_to(images_root)
    return (labels_root / rel).with_suffix(".txt")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_txt", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--categories", default="cessna", help="comma separated")

    # tiles/raw 공통: data/... 구조를 그대로 쓸 수 있게 기본값 제공
    ap.add_argument("--images_root", default="data/tiles_2x4/images")
    ap.add_argument("--labels_root", default="data/tiles_2x4/labels")
    ap.add_argument("--file_name_mode", choices=["basename", "relative"], default="basename",
                    help="basename: file_name에 파일명만 저장(추천). relative: images_root 기준 상대경로 저장.")
    args = ap.parse_args()

    images_root = Path(args.images_root)
    labels_root = Path(args.labels_root)

    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    categories = [{"id": i, "name": name} for i, name in enumerate(cats)]

    img_paths = [Path(x.strip()) for x in Path(args.split_txt).read_text(encoding="utf-8").splitlines() if x.strip()]

    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    for p in img_paths:
        if p.suffix.lower() not in IMG_EXTS:
            continue
        if not p.exists():
            print(f"[WARN] missing image: {p}")
            continue

        img = cv2.imread(str(p))
        if img is None:
            print(f"[WARN] cv2.imread failed: {p}")
            continue
        H, W = img.shape[:2]

        # ✅ file_name 저장 규칙
        if args.file_name_mode == "basename":
            file_name = p.name
        else:
            file_name = str(p.relative_to(images_root))

        images.append({
            "id": img_id,
            "file_name": file_name,
            "width": W,
            "height": H
        })

        # 라벨 경로는 images_root/labels_root 기준으로 안전하게 매핑
        label_path = img_to_label_path(p, images_root=images_root, labels_root=labels_root)
        labs = read_yolo(label_path)

        for (cls, cx, cy, w, h) in labs:
            cx *= W; cy *= H; w *= W; h *= H
            x = cx - w/2
            y = cy - h/2
            x = max(0, x); y = max(0, y)
            w = max(0, min(W - x, w))
            h = max(0, min(H - y, h))
            if w <= 1e-3 or h <= 1e-3:
                continue

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls,
                "bbox": [x, y, w, h],
                "area": float(w*h),
                "iscrowd": 0
            })
            ann_id += 1

        img_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {args.out_json} (images={len(images)}, anns={len(annotations)})")

if __name__ == "__main__":
    main()


# # 실행 예시
# python scripts/yolo_txt_to_coco.py --split_txt data/splits/tiles_4x2_train.txt --out_json data/coco/tiles_4x2/annotations/instances_train.json --images_root data/tiles_4x2/images --labels_root data/tiles_4x2/labels --file_name_mode basename
# python scripts/yolo_txt_to_coco.py --split_txt data/splits/tiles_4x2_val.txt   --out_json data/coco/tiles_4x2/annotations/instances_val.json   --images_root data/tiles_4x2/images --labels_root data/tiles_4x2/labels --file_name_mode basename
# python scripts/yolo_txt_to_coco.py --split_txt data/splits/tiles_4x2_test.txt  --out_json data/coco/tiles_4x2/annotations/instances_test.json  --images_root data/tiles_4x2/images --labels_root data/tiles_4x2/labels --file_name_mode basename
#
# python scripts/yolo_txt_to_coco.py --split_txt data/splits/raw_train.txt --out_json data/coco/raw/annotations/instances_train.json --images_root /home/rs02/NAS/Dataset/Cessna_220429/images --labels_root /home/rs02/NAS/Dataset/Cessna_220429/labels --file_name_mode basename
# python scripts/yolo_txt_to_coco.py --split_txt data/splits/raw_val.txt   --out_json data/coco/raw/annotations/instances_val.json   --images_root /home/rs02/NAS/Dataset/Cessna_220429/images --labels_root /home/rs02/NAS/Dataset/Cessna_220429/labels --file_name_mode basename
# python scripts/yolo_txt_to_coco.py --split_txt data/splits/raw_test.txt  --out_json data/coco/raw/annotations/instances_test.json  --images_root /home/rs02/NAS/Dataset/Cessna_220429/images --labels_root /home/rs02/NAS/Dataset/Cessna_220429/labels --file_name_mode basename

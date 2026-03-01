"""Train YOLOv11n-p2-seg on synthetic defect data."""
from ultralytics import YOLO
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolo11-p2-seg.yaml')
    parser.add_argument('--data', type=str, default='data.yaml')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=1536)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--name', type=str, default='yolo11_p2_seg')
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project='../outputs',
        name=args.name,
        seed=42,
        workers=4,
        verbose=True,
    )


if __name__ == '__main__':
    main()

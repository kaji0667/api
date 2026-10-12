"""Prepare the teammate's single-scene VRSBench ImageFolder dataset."""

import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


class VRSBenchSceneDatasetBuilder:
    """Select images with exactly one scene label and archive them by that label."""

    OBJECT_CLASSES = {"vehicle", "ship", "airplane", "helicopter", "container-crane"}
    SCENE_CLASSES = {
        "airport",
        "baseball-diamond",
        "basketball-court",
        "bridge",
        "chimney",
        "dam",
        "expressway-service-area",
        "expressway-toll-station",
        "golffield",
        "ground-track-field",
        "harbor",
        "helipad",
        "overpass",
        "roundabout",
        "soccer-ball-field",
        "stadium",
        "storage-tank",
        "swimming-pool",
        "tennis-court",
        "trainstation",
        "windmill",
    }

    def __init__(self, voc_root, output_dir, link_mode="hardlink", max_per_class=0):
        self.voc_root = Path(voc_root).expanduser().resolve()
        self.images_dir = self.voc_root / "JPEGImages"
        self.annotations_dir = self.voc_root / "Annotations"
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.link_mode = str(link_mode)
        self.max_per_class = int(max_per_class)
        if self.link_mode not in {"hardlink", "copy"}:
            raise ValueError("link_mode must be 'hardlink' or 'copy'.")

    @staticmethod
    def read_labels(xml_path):
        root = ET.parse(xml_path).getroot()
        return [node.text.strip() for node in root.iter("name") if node.text and node.text.strip()]

    @classmethod
    def select_scene_label(cls, labels):
        scene_labels = sorted(set(labels) & cls.SCENE_CLASSES)
        return scene_labels[0] if len(scene_labels) == 1 else None

    def _materialize(self, source, destination):
        if destination.exists():
            return
        if self.link_mode == "hardlink":
            try:
                os.link(source, destination)
                return
            except OSError:
                pass
        shutil.copy2(source, destination)

    def build(self):
        if not self.images_dir.is_dir() or not self.annotations_dir.is_dir():
            raise FileNotFoundError(f"VOC images/annotations not found under {self.voc_root}")
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"Refusing to replace non-empty scene dataset: {self.output_dir}")

        selected = defaultdict(list)
        dropped_multi_scene = 0
        dropped_pure_object = 0
        for xml_path in sorted(self.annotations_dir.glob("*.xml")):
            labels = self.read_labels(xml_path)
            scene_labels = sorted(set(labels) & self.SCENE_CLASSES)
            if len(scene_labels) == 1:
                image_path = self.images_dir / f"{xml_path.stem}.jpg"
                if image_path.is_file():
                    selected[scene_labels[0]].append(image_path)
            elif len(scene_labels) >= 2:
                dropped_multi_scene += 1
            else:
                dropped_pure_object += 1

        self.output_dir.mkdir(parents=True, exist_ok=True)
        class_counts = {}
        for class_name, images in sorted(selected.items()):
            if self.max_per_class > 0:
                images = images[: self.max_per_class]
            class_dir = self.output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for image_path in images:
                self._materialize(image_path, class_dir / image_path.name)
            class_counts[class_name] = len(images)
        report = {
            "source": str(self.voc_root),
            "output": str(self.output_dir),
            "selection_rule": "exactly_one_scene_class",
            "link_mode": self.link_mode,
            "classes": class_counts,
            "total_images": sum(class_counts.values()),
            "dropped_multi_scene": dropped_multi_scene,
            "dropped_pure_object": dropped_pure_object,
        }
        (self.output_dir / "scene_dataset_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report


def build_parser():
    parser = argparse.ArgumentParser(
        description="Prepare VRSBench scene-classification ImageFolder data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--voc-root", default="data/VRSBench")
    parser.add_argument("--output-dir", default="data/VRSBench_scene")
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--max-per-class", type=int, default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = VRSBenchSceneDatasetBuilder(
        args.voc_root,
        args.output_dir,
        link_mode=args.link_mode,
        max_per_class=args.max_per_class,
    ).build()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

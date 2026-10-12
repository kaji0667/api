"""VOC-style VRSBench samples used by the counting evaluator."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterator
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class CountingSample:
    """One image and its positive ground-truth class counts."""

    image_id: str
    image_path: Path
    annotation_path: Path
    counts: Dict[str, int]


class VRSCountingDataset:
    """Iterate the teammate's VOC-style positive-query counting protocol."""

    def __init__(self, voc_root: str | Path, split_file: str = "test.txt") -> None:
        self.voc_root = Path(voc_root).expanduser().resolve()
        self.split_file = str(split_file)
        if not self.voc_root.is_dir():
            raise FileNotFoundError(f"VOC dataset root not found: {self.voc_root}")
        self.split_path = self._resolve_split_path()
        self.image_ids = self._load_image_ids()

    def _resolve_split_path(self) -> Path:
        candidates = (
            self.voc_root / self.split_file,
            self.voc_root / "ImageSets" / "Main" / self.split_file,
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"Split file '{self.split_file}' was not found under {self.voc_root} "
            "or ImageSets/Main."
        )

    def _load_image_ids(self) -> list[str]:
        with open(self.split_path, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    def _resolve_image_path(self, image_id: str) -> Path | None:
        for suffix in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            candidate = self.voc_root / "JPEGImages" / f"{image_id}{suffix}"
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def parse_counts(annotation_path: Path) -> Dict[str, int]:
        """Parse object counts grouped by the original VOC class name."""
        counts: DefaultDict[str, int] = defaultdict(int)
        root = ET.parse(annotation_path).getroot()
        for obj in root.findall("object"):
            name_node = obj.find("name")
            if name_node is None or not name_node.text:
                continue
            counts[name_node.text.strip().lower()] += 1
        return dict(counts)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __iter__(self) -> Iterator[CountingSample]:
        for image_id in self.image_ids:
            annotation_path = self.voc_root / "Annotations" / f"{image_id}.xml"
            image_path = self._resolve_image_path(image_id)
            if not annotation_path.is_file() or image_path is None:
                continue
            counts = self.parse_counts(annotation_path)
            if not counts:
                continue
            yield CountingSample(
                image_id=image_id,
                image_path=image_path,
                annotation_path=annotation_path,
                counts=counts,
            )

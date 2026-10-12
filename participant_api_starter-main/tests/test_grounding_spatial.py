"""Public synthetic descriptions exercise safe absolute-position attachment."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grounding_spatial import AbsolutePosition, parse_absolute_position


class AbsolutePositionTests(unittest.TestCase):
    def test_nine_english_positions_and_common_hyphens(self):
        cases = {
            "The warehouse in the upper-left corner of the image": (-1, -1),
            "The warehouse at the top middle of this picture": (0, -1),
            "The warehouse in the top-right corner": (1, -1),
            "The warehouse on the middle left side": (-1, 0),
            "The warehouse in the centre of the photo": (0, 0),
            "The warehouse in the center right area": (1, 0),
            "The lower–left warehouse": (-1, 1),
            "The warehouse at the bottom center": (0, 1),
            "A lower-right warehouse": (1, 1),
        }
        for description, axes in cases.items():
            with self.subTest(description=description):
                self.assertEqual(parse_absolute_position(description), AbsolutePosition(*axes))

    def test_nine_chinese_positions_and_target_first_locations(self):
        cases = {
            "图像左上角的仓库": (-1, -1), "顶部中央的仓库": (0, -1),
            "图片右上角的红色仓库": (1, -1), "左侧中部的仓库": (-1, 0),
            "仓库位于画面正中央": (0, 0), "右侧中部的仓库": (1, 0),
            "红色仓库位于左下方": (-1, 1), "底部中央的仓库": (0, 1),
            "仓库在这张图片的右下角": (1, 1),
        }
        for description, axes in cases.items():
            with self.subTest(description=description):
                self.assertEqual(parse_absolute_position(description), AbsolutePosition(*axes))

    def test_single_axis_does_not_invent_a_center_axis(self):
        cases = {
            "The warehouse at the left": (-1, None),
            "The warehouse on the right side of the image": (1, None),
            "The warehouse located at the top": (None, -1),
            "The warehouse in the lower part": (None, 1),
            "左侧的仓库": (-1, None), "仓库在右边": (1, None),
            "正上方的仓库": (None, -1), "仓库位于图中下方": (None, 1),
            "图中左侧的仓库": (-1, None), "图中右边的仓库": (1, None),
        }
        for description, axes in cases.items():
            with self.subTest(description=description):
                self.assertEqual(parse_absolute_position(description), AbsolutePosition(*axes))

    def test_finite_request_prefixes_leave_target_position_attachment_intact(self):
        cases = {
            "请框出图像左下角的梯形草坪": (-1, 1),
            "定位左下角的草坪": (-1, 1),
            "请定位位于图片右上角的草坪": (1, -1),
            "Locate the bottom-left lawn": (-1, 1),
            "Please find the lawn in the top right corner of the image": (1, -1),
            "Identify the warehouse at the center": (0, 0),
        }
        for description, axes in cases.items():
            with self.subTest(description=description):
                self.assertEqual(parse_absolute_position(description), AbsolutePosition(*axes))
        for description in ("定位右上角仓库旁的汽车", "Locate a car near the bottom-left warehouse"):
            with self.subTest(description=description):
                self.assertIsNone(parse_absolute_position(description))

    def test_target_names_do_not_turn_into_negation(self):
        for description in ("左下角的无人机", "无人机在图像左下角", "左下角的不规则草坪"):
            with self.subTest(description=description):
                self.assertEqual(parse_absolute_position(description), AbsolutePosition(-1, 1))

    def test_only_first_target_clause_contributes(self):
        cases = {
            "The red warehouse at the lower left. A road is above it.": (-1, 1),
            "The warehouse in the top right, a road lies to its left": (1, -1),
            "The warehouse on the left; a car is at the bottom": (-1, None),
            "红色仓库位于左下方，右侧有道路。": (-1, 1),
            "右上角的仓库；下方是一条道路": (1, -1),
            "The warehouse\nThe road is in the upper left": None,
            "The warehouse. A road is in the upper left": None,
            "仓库，右上角有道路": None,
        }
        for description, axes in cases.items():
            with self.subTest(description=description):
                expected = AbsolutePosition(*axes) if axes is not None else None
                self.assertEqual(parse_absolute_position(description), expected)

    def test_reference_objects_do_not_become_absolute_target_positions(self):
        descriptions = (
            "The warehouse to the left of the bridge",
            "The warehouse on the left side of the bridge",
            "The warehouse lower left of the road",
            "warehouse near bridge in top right of image",
            "The warehouse with a road in the upper-left corner of the image",
            "The car beside the top-right warehouse",
            "The warehouse that has a bridge in the bottom left",
            "The warehouse has a red door on the left",
            "The warehouse's car in the upper left",
            "The left wing of the airplane",
            "右上角仓库旁的汽车", "靠近右上角桥的仓库",
            "图片右上角附近的仓库", "右上角仓库的汽车",
            "道路左侧的仓库", "仓库在道路左边",
            "仓库的右侧有道路", "仓库有一条道路在右侧",
            "位于右上角的仓库旁边的汽车",
        )
        for description in descriptions:
            with self.subTest(description=description):
                self.assertIsNone(parse_absolute_position(description))

    def test_negation_conflicts_and_multiple_positions_have_no_hint(self):
        descriptions = (
            "The warehouse not in the lower left",
            "The warehouse isn't in the upper right",
            "Exclude the warehouse in the upper left",
            "The warehouse in the lower left or upper right",
            "The warehouse on the left and at the bottom",
            "The upper left warehouse at the top",
            "不是右上角的仓库", "排除左下方的仓库",
            "左下角而非右上角的仓库", "仓库在左边或右边",
            "左侧和右侧的仓库", "非右上角的仓库", "无人机不在右上角",
        )
        for description in descriptions:
            with self.subTest(description=description):
                self.assertIsNone(parse_absolute_position(description))

    def test_motion_cardinal_geography_and_nonspatial_words_have_no_hint(self):
        descriptions = (
            "The ship heading left", "The ship facing right",
            "The warehouse in the northeast", "东北角的仓库",
            "The right angle building", "The upper deck",
            "The middle school", "The middle-sized warehouse",
            "The upper floor", "The lower deck",
            "The copyright building", "The tabletop warehouse",
            "向左行驶的车辆", "左转的车辆", "右手形状的建筑",
            "上海的仓库", "The warehouse", "仓库", "", "  ",
        )
        for description in descriptions:
            with self.subTest(description=description):
                self.assertIsNone(parse_absolute_position(description))

    def test_empty_or_unsupported_input_returns_no_hint(self):
        self.assertIsNone(parse_absolute_position(None))
        self.assertIsNone(parse_absolute_position(17))

    def test_axes_are_valid_and_immutable(self):
        for axes in ((2, None), (None, -2), (True, None), (0.0, None), (None, None)):
            with self.subTest(axes=axes), self.assertRaises(ValueError):
                AbsolutePosition(*axes)
        position = AbsolutePosition(-1, 1)
        with self.assertRaises(AttributeError):
            position.horizontal = 1


if __name__ == "__main__":
    unittest.main()

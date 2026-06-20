from .nodes import Any2ListNode, ParseJsonNode, TD_VideoCombine
from .ltxv_grid_guide import TD_LTXVAddGuideFromGrid

NODE_CLASS_MAPPINGS = {
    "Any2ListNode": Any2ListNode,
    "ParseJsonNode": ParseJsonNode,
    "TD_VideoCombine": TD_VideoCombine,
    "TD_LTXVAddGuideFromGrid": TD_LTXVAddGuideFromGrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Any2ListNode": "Any to List (AI代码侠土豆)",
    "ParseJsonNode": "Parse JSON (AI代码侠土豆)",
    "TD_VideoCombine": "Video Combine (AI代码侠土豆)",
    "TD_LTXVAddGuideFromGrid": "LTXV 四宫格引导 (AI代码侠土豆)",
}

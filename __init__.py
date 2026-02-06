from .nodes import Any2ListNode, ParseJsonNode, TD_VideoCombine

NODE_CLASS_MAPPINGS = {
    "Any2ListNode": Any2ListNode,
    "ParseJsonNode": ParseJsonNode,
    "TD_VideoCombine": TD_VideoCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Any2ListNode": "Any to List (AI代码侠土豆)",
    "ParseJsonNode": "Parse JSON (AI代码侠土豆)",
    "TD_VideoCombine": "Video Combine (AI代码侠土豆)",
}

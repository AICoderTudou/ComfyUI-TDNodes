from .nodes import Any2ListNode, ParseJsonNode, TD_VideoCombine
from .ltxv_grid_guide import TD_LTXVAddGuideFromGrid

# 单一配置源:加/改节点只动这一处,自动生成两个 mapping(参考 KJNodes 的 NODE_CONFIG 模式)
NODE_CONFIG = {
    "Any2ListNode":            {"class": Any2ListNode,             "name": "Any to List (AI代码侠土豆)"},
    "ParseJsonNode":           {"class": ParseJsonNode,            "name": "Parse JSON (AI代码侠土豆)"},
    "TD_VideoCombine":         {"class": TD_VideoCombine,          "name": "Video Combine (AI代码侠土豆)"},
    "TD_LTXVAddGuideFromGrid": {"class": TD_LTXVAddGuideFromGrid,  "name": "LTXV 四宫格引导 (AI代码侠土豆)"},
}


def generate_node_mappings(node_config):
    class_mappings, display_mappings = {}, {}
    for node_id, info in node_config.items():
        class_mappings[node_id] = info["class"]
        display_mappings[node_id] = info.get("name", info["class"].__name__)
    return class_mappings, display_mappings


NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

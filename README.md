# ComfyUI-TDNodes

Custom nodes for ComfyUI.
ComfyUI 自定义节点合集。

## Update Log / 更新日志

### 2026-06-20
- **English**: Added **LTXV Add Guide from Grid** node (`TD_LTXVAddGuideFromGrid`). Feed a single grid image (e.g. a 2×2 四宫格) and it internally splits → encodes → adds all keyframe guides to the LTX video latent in one node. Replaces the `分割 + ImageFromBatch×4 + ImageResize×4 + LTXVAddGuideMulti` chain. Built on ComfyUI core (`comfy_extras.nodes_lt.LTXVAddGuide`), no KJNodes dependency; pure in-memory split (no disk writes).
- **中文**: 新增 **LTXV 四宫格引导** (`TD_LTXVAddGuideFromGrid`) 节点。直接传入一张网格图(如 2×2 四宫格),节点内部完成 切分→编码→给 LTX 视频 latent 追加全部关键帧引导,一个节点顶替 `分割 + 4×ImageFromBatch + 4×ImageResize + LTXVAddGuideMulti` 整条链路。基于 ComfyUI 核心实现,不依赖 KJNodes;纯内存切分,不落盘。

### 2026-02-06
- **English**: Added **Parse JSON** node and **TD Video Combine** node.
- **中文**: 新增 **JSON解析** 节点和 **TD视频合成** (TD_VideoCombine) 节点。

### 2026-02-05
- **English**: Added **Any to List** node (Any2ListNode).
- **中文**: 新增 **Any to List** (任意转列表) 节点。

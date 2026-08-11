# ComfyUI-TDNodes

Custom nodes for ComfyUI.
ComfyUI 自定义节点合集。

## Update Log / 更新日志

## Aitudou API

The TD image-generation nodes use the official [Aitudou API](https://api.aitudou.net/). Create and manage your API key from the official site before using these nodes.

## TD Midjourney Text to Image

`TD Midjourney Text to Image` uses the [official Aitudou API](https://api.aitudou.net/) Midjourney flow:
`POST /v1/midjourney/generations` followed by `GET /v1/midjourney/tasks/{task_id}` polling.

- Enter an Aitudou API key in `api_key`.
- The node uses `https://api.aitudou.net` with the built-in polling settings.
- The only generation controls are a common-ratio dropdown and `version`.
- It returns only a ComfyUI `IMAGE` batch.

## TD GPT Image2 Text to Image

`TD GPT Image2 Text to Image` uses [Aitudou API](https://api.aitudou.net/) model `aitudou-image-g-v2-lowprice` through `POST /v1/image/generations`.

- Inputs: `prompt`, `api_key`, an aspect-ratio dropdown, and resolution (`1k`, `2k`, or `4k`).
- Returns an `IMAGE` batch after polling the image task to completion.

More details: [TD image node guide](docs/td-image-nodes.md).

### 2026-08-11
- **English**: Added **TD Midjourney Text to Image** (`TD_MidjourneyTextToImage`). It submits an Aitudou Midjourney Imagine task, polls it to completion, and returns the results as a ComfyUI image batch.

### 2026-07-04
- **English**: Added **MSR Identity Only** node (`TD_MSRIdentityOnly`) for LTX IC-LoRA workflows. It builds a subject-only reference frame sequence from up to 4 character images, without requiring a background image. This is intended for workflows where the TD four-grid guide controls scene/storyboard/composition, while MSR/IC-LoRA controls character identity consistency.
- **中文**: 新增 **MSR Identity Only** (`TD_MSRIdentityOnly`) 节点,用于 LTX IC-LoRA / MSR 人物一致性工作流。节点可接入最多 4 张人物参考图,生成仅包含主体身份参考的帧序列,不再要求传入 background 背景图。适合“四宫格负责场景/分镜/构图,MSR/IC-LoRA 只负责人物一致性”的工作流。
- **English**: Updated the recommended LTX workflow pattern: `TD_LTXVAddGuideFromGrid` handles the empty or storyboard grid image with built-in split/crop/guide encoding; `TD_MSRIdentityOnly` feeds character references into `LTXAddVideoICLoRAGuide`; `LTXICLoRALoaderModelOnly` loads the MSR IC-LoRA model. This avoids external grid cropping nodes and avoids conflicts between storyboard backgrounds and MSR background slots.
- **中文**: 更新推荐的 LTX 工作流结构:`TD_LTXVAddGuideFromGrid` 处理四宫格场景/分镜图,内部完成切分、可选去边与 guide 编码;`TD_MSRIdentityOnly` 只把人物参考图传给 `LTXAddVideoICLoRAGuide`;`LTXICLoRALoaderModelOnly` 加载 MSR IC-LoRA 模型。这样可以去掉工作流里的外部四宫格裁剪链路,同时避免四宫格场景图与 MSR background 输入发生冲突。

### 2026-06-20
- **English**: Added **LTXV Add Guide from Grid** node (`TD_LTXVAddGuideFromGrid`). Feed a single grid image (e.g. a 2×2 四宫格) and it internally splits → encodes → adds all keyframe guides to the LTX video latent in one node. Replaces the `分割 + ImageFromBatch×4 + ImageResize×4 + LTXVAddGuideMulti` chain. Built on ComfyUI core (`comfy_extras.nodes_lt.LTXVAddGuide`), no KJNodes dependency; pure in-memory split (no disk writes).
- **中文**: 新增 **LTXV 四宫格引导** (`TD_LTXVAddGuideFromGrid`) 节点。直接传入一张网格图(如 2×2 四宫格),节点内部完成 切分→编码→给 LTX 视频 latent 追加全部关键帧引导,一个节点顶替 `分割 + 4×ImageFromBatch + 4×ImageResize + LTXVAddGuideMulti` 整条链路。基于 ComfyUI 核心实现,不依赖 KJNodes;纯内存切分,不落盘。

### 2026-02-06
- **English**: Added **Parse JSON** node and **TD Video Combine** node.
- **中文**: 新增 **JSON解析** 节点和 **TD视频合成** (TD_VideoCombine) 节点。

### 2026-02-05
- **English**: Added **Any to List** node (Any2ListNode).
- **中文**: 新增 **Any to List** (任意转列表) 节点。

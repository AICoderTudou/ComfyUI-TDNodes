"""
TD_LTXVAddGuideFromGrid —— 四宫格(网格)图片一体化引导节点。

把原工作流里的:
    孤海自动分割图像 + 4×ImageFromBatch+ + 4×ImageResizeKJv2 + LTXVAddGuideMulti
合并成一个节点:输入一张网格图,内部切分 → 直接逐块给 LTX 视频 latent 添加多关键帧引导。

实现说明:
- 真正的引导编码逻辑(encode / get_latent_index / append_keyframe)直接复用 ComfyUI
  核心模块 comfy_extras.nodes_lt.LTXVAddGuide,**不依赖 KJNodes**,行为与 LTXVAddGuideMulti 一致。
- encode() 内部已做 center-crop + 缩放到视频尺寸,所以这里**不需要再单独缩放**分块。
- 切分纯内存完成,**不再像孤海节点那样把每块存盘到 output/**。

作者: AI代码侠土豆 (AICoderTudou)
"""

import logging

import torch

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 工具函数 / helpers
# --------------------------------------------------------------------------
def _auto_crop(img):
    """去掉四周的纯色画布边(对应孤海"移除画布边缘")。

    img: [H, W, C] float(0~1) 张量。返回裁剪后的 [H', W', C]。
    逻辑与孤海一致:取四角中位数作背景色,阈值 30/255,取非背景区域的外接框。
    """
    rgb = img[..., :3]
    h, w = rgb.shape[0], rgb.shape[1]
    if h < 2 or w < 2:
        return img
    corners = torch.stack([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]], dim=0)  # [4,3]
    bg = corners.median(dim=0).values  # [3]
    diff = (rgb - bg).abs().amax(dim=-1)  # [H,W]
    mask = diff > (30.0 / 255.0)
    rows = torch.where(mask.any(dim=1))[0]
    cols = torch.where(mask.any(dim=0))[0]
    if rows.numel() == 0 or cols.numel() == 0:
        return img  # 整张都是背景色,放弃裁剪
    top, bottom = int(rows[0]), int(rows[-1])
    left, right = int(cols[0]), int(cols[-1])
    return img[top:bottom + 1, left:right + 1, :]


def _split_grid(img, rows, columns, trim_border):
    """把单张图按 rows×columns 切块,行优先(左上→右上→左下→右下)。

    img: [H, W, C] -> list[ [1, th, tw, C] ]。与孤海的切分次序、尺寸完全一致。
    """
    h, w = img.shape[0], img.shape[1]
    tile_h = h // rows
    tile_w = w // columns
    off_y = (h - tile_h * rows) // 2   # 非整除时把余数居中分摊,而非全丢在底/右边
    off_x = (w - tile_w * columns) // 2
    tiles = []
    for y in range(rows):
        for x in range(columns):
            y0, x0 = off_y + y * tile_h, off_x + x * tile_w
            tile = img[y0:y0 + tile_h, x0:x0 + tile_w, :]
            if trim_border > 0:
                th, tw = tile.shape[0], tile.shape[1]
                tb = min(trim_border, th // 2 - 1, tw // 2 - 1)
                if tb > 0:
                    tile = tile[tb:th - tb, tb:tw - tb, :]
            tiles.append(tile.unsqueeze(0).contiguous())  # [1, th, tw, C]
    return tiles


def _auto_frames(n, pixel_len):
    """按视频像素帧长度,把 n 个引导均匀铺在 [0, pixel_len-1] 上。"""
    if n <= 1:
        return [0]
    return [round(i * (pixel_len - 1) / (n - 1)) for i in range(n)]


def _parse_frames(s, n, pixel_len):
    """解析 frame_indices 文本框。留空 / "auto" / 数量与分块数不符 → 自动均匀分布。"""
    s = (s or "").strip().replace("，", ",")
    if not s or s.lower() == "auto":
        return _auto_frames(n, pixel_len)
    vals = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            vals.append(int(round(float(p))))
        except ValueError:
            pass
    if len(vals) != n:
        logger.warning(
            "[TD_LTXVAddGuideFromGrid] frame_indices 个数(%d)与分块数(%d)不一致,改为自动均匀分布。",
            len(vals), n)
        return _auto_frames(n, pixel_len)
    return vals


def _parse_strengths(s, n):
    """解析 strengths 文本框。留空 → 全 1.0;数量不足按最后一个补齐;过多则截断。"""
    s = (s or "").strip().replace("，", ",")
    if not s:
        return [1.0] * n
    vals = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            vals.append(float(p))
        except ValueError:
            pass
    if not vals:
        return [1.0] * n
    if len(vals) < n:
        vals = vals + [vals[-1]] * (n - len(vals))
    return vals[:n]


# --------------------------------------------------------------------------
# 节点 / node
# --------------------------------------------------------------------------
class TD_LTXVAddGuideFromGrid:
    """四宫格图片 → 内部切分 → 一次性给 LTX 视频 latent 添加多关键帧引导。

    输出 (positive, negative, latent) 与 LTXVAddGuideMulti 完全一致,可直接顶替。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "grid_image": ("IMAGE", {"tooltip": "网格图(如四宫格)。只取批次里的第一张。"}),
                "columns": ("INT", {"default": 2, "min": 1, "max": 100, "tooltip": "水平张数(列)。"}),
                "rows": ("INT", {"default": 2, "min": 1, "max": 100, "tooltip": "垂直张数(行)。"}),
                "frame_indices": ("STRING", {
                    "default": "0,96,192,287",
                    "tooltip": "每块的起始帧,逗号分隔,次序为 左上→右上→左下→右下。\n"
                               "留空或填 auto = 按视频长度自动均匀分布(改时长不会错位)。",
                }),
                "strengths": ("STRING", {
                    "default": "0.7,0.75,0.9,0.85",
                    "tooltip": "每块的引导强度,逗号分隔。个数不足按最后一个补齐,留空=全部 1.0。",
                }),
            },
            "optional": {
                "remove_edge": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "切分前去掉四周纯色画布边(对应孤海「移除画布边缘」)。",
                }),
                "trim_border": ("INT", {
                    "default": 0, "min": 0, "max": 512,
                    "tooltip": "每个分块再向内裁掉的像素(去描边/缝隙)。",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "TDNodes/LTXV"
    DESCRIPTION = "四宫格图一体化引导:输入一张网格图,内部切分→编码→给 LTX 视频 latent 追加多关键帧引导。一个节点替代 孤海分割 + ImageFromBatch + ImageResize + LTXVAddGuideMulti 整条链。"

    def execute(self, positive, negative, vae, latent, grid_image, columns, rows,
                frame_indices, strengths, remove_edge=False, trim_border=0):
        # 延迟导入核心 LTX 模块:即使某些版本缺失也不至于让整个 TDNodes 加载失败
        from comfy_extras.nodes_lt import LTXVAddGuide, get_noise_mask
        try:
            from comfy_extras.nodes_lt import _append_guide_attention_entry
        except Exception:
            _append_guide_attention_entry = None  # 旧版本没有逐引导注意力控制,降级跳过

        # 1) 取第一张图 →(可选)去边 → 按网格切块
        img = grid_image[0]
        if remove_edge:
            img = _auto_crop(img)
        tiles = _split_grid(img, rows, columns, trim_border)
        n = len(tiles)

        # 2) 读取 latent 尺寸,解析帧位置与强度
        scale_factors = vae.downscale_index_formula
        time_scale_factor = scale_factors[0]
        latent_image = latent["samples"]
        noise_mask = get_noise_mask(latent)
        # 注意:latent_length 只在循环外取一次(与 LTXVAddGuideMulti 一致);
        # get_latent_index 内部会根据已追加的关键帧数自行修正。
        _, _, latent_length, latent_height, latent_width = latent_image.shape
        pixel_len = (latent_length - 1) * time_scale_factor + 1

        frame_idxs = _parse_frames(frame_indices, n, pixel_len)
        strs = _parse_strengths(strengths, n)

        # 3) 逐块编码 + 追加引导(复刻 LTXVAddGuideMulti.execute 的循环)
        for i, tile in enumerate(tiles):
            f_idx = frame_idxs[i]
            strength = strs[i]

            image_1, t = LTXVAddGuide.encode(vae, latent_width, latent_height, tile, scale_factors)
            frame_idx, latent_idx = LTXVAddGuide.get_latent_index(
                positive, latent_length, len(image_1), f_idx, scale_factors)
            if latent_idx + t.shape[2] > latent_length:
                raise ValueError(
                    f"[TD_LTXVAddGuideFromGrid] 第 {i + 1} 块的帧索引 {f_idx} 超出 latent 长度,"
                    f"请调小 frame_indices 或加长视频帧数。")

            positive, negative, latent_image, noise_mask = LTXVAddGuide.append_keyframe(
                positive, negative, frame_idx, latent_image, noise_mask, t, strength, scale_factors)

            if _append_guide_attention_entry is not None:
                pre_filter_count = t.shape[2] * t.shape[3] * t.shape[4]
                guide_latent_shape = list(t.shape[2:])  # [F, H, W]
                positive, negative = _append_guide_attention_entry(
                    positive, negative, pre_filter_count, guide_latent_shape, strength=strength)

        logger.info("[TD_LTXVAddGuideFromGrid] 切 %d 块, frames=%s, strengths=%s",
                    n, frame_idxs, strs)
        return (positive, negative, {"samples": latent_image, "noise_mask": noise_mask})

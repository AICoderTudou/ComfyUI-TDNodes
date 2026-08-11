"""Aitudou Image G v2 low-price text-to-image node."""

import time

from .midjourney import (
    ASPECT_RATIO_OPTIONS,
    DEFAULT_API_BASE_URL,
    FAILURE_STATUSES,
    SUCCESS_STATUSES,
    TD_MidjourneyTextToImage,
)


class TD_GPTImage2TextToImage:
    """Generate images with aitudou-image-g-v2-lowprice."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "aspect_ratio": (ASPECT_RATIO_OPTIONS, {"default": "1:1"}),
                "resolution": (["1k", "2k", "4k"], {"default": "1k"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    CATEGORY = "TDNodes/GPT Image2"
    DESCRIPTION = "Aitudou Image G v2 low-price text-to-image with aspect-ratio and resolution selectors."

    def generate(self, prompt: str, api_key: str, aspect_ratio: str, resolution: str):
        prompt = (prompt or "").strip()
        key = (api_key or "").strip()
        if not prompt:
            raise ValueError("GPT Image2 prompt cannot be empty.")
        if not key:
            raise ValueError("Enter api_key.")

        submission = TD_MidjourneyTextToImage._request_json(
            DEFAULT_API_BASE_URL,
            "/v1/image/generations",
            key,
            "POST",
            {
                "model": "aitudou-image-g-v2-lowprice",
                "prompt": prompt,
                "size": aspect_ratio,
                "metadata": {"resolution": resolution},
            },
        )
        task_id = TD_MidjourneyTextToImage._task_id(submission)
        if not task_id:
            raise RuntimeError(f"GPT Image2 submission did not return a task_id: {TD_MidjourneyTextToImage._compact_json(submission)}")

        completed = self._poll_task(task_id, key)
        image_urls = TD_MidjourneyTextToImage._image_urls(completed)
        if not image_urls:
            raise RuntimeError(f"GPT Image2 task {task_id} completed without an image URL: {TD_MidjourneyTextToImage._compact_json(completed)}")
        images = [TD_MidjourneyTextToImage._download_image(url) for url in image_urls]
        return (TD_MidjourneyTextToImage._to_batch(images),)

    @staticmethod
    def _poll_task(task_id: str, api_key: str):
        deadline = time.monotonic() + 30 * 60
        progress_bar = TD_MidjourneyTextToImage._progress_bar()
        while time.monotonic() <= deadline:
            response = TD_MidjourneyTextToImage._request_json(
                DEFAULT_API_BASE_URL,
                f"/v1/image/generations/{task_id}",
                api_key,
                "GET",
            )
            body = TD_MidjourneyTextToImage._response_body(response)
            status = str(body.get("status") or body.get("state") or body.get("task_status") or "").upper().replace("-", "_")
            progress = TD_MidjourneyTextToImage._progress(body)
            if progress is not None and progress_bar is not None:
                progress_bar.update_absolute(progress, 100)
            print(f"[TD GPT Image2] {task_id}: {status or 'POLLING'}" + (f" ({progress}%)" if progress is not None else ""))
            if status in SUCCESS_STATUSES or (not status and TD_MidjourneyTextToImage._image_urls(response)):
                return response
            if status in FAILURE_STATUSES:
                message = TD_MidjourneyTextToImage._message(body) or TD_MidjourneyTextToImage._message(response) or "unknown failure"
                raise RuntimeError(f"GPT Image2 task {task_id} failed: {message}")
            time.sleep(4)
        raise TimeoutError(f"GPT Image2 task {task_id} did not finish within 30 minutes.")

"""Midjourney text-to-image node backed by the Aitudou public API."""

import io
import json
import time
from typing import Any

import numpy as np
import requests
import torch
from PIL import Image


DEFAULT_API_BASE_URL = "https://api.aitudou.net"
SUCCESS_STATUSES = {"SUCCESS", "SUCCEEDED", "COMPLETED", "COMPLETE", "DONE", "FINISHED"}
FAILURE_STATUSES = {"FAIL", "FAILED", "FAILURE", "ERROR", "CANCELED", "CANCELLED"}
ASPECT_RATIO_OPTIONS = ["1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "21:9"]


class TD_MidjourneyTextToImage:
    """Submit a Midjourney imagine task and return its completed image batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "aspect_ratio": (ASPECT_RATIO_OPTIONS, {"default": "1:1"}),
                "version": (["8.2", "8.1", "7", "6.1", "6", "5.2", "5.1"], {"default": "8.2"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    CATEGORY = "TDNodes/Midjourney"
    DESCRIPTION = (
        "Aitudou Midjourney Imagine with an API key, image aspect ratio, and version selector."
    )

    def generate(
        self,
        prompt: str,
        api_key: str,
        aspect_ratio: str,
        version: str,
    ):
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Midjourney prompt cannot be empty.")

        key = api_key.strip()
        if not key:
            raise ValueError("Enter api_key.")

        payload = self._imagine_payload(prompt, aspect_ratio, version)
        submission = self._request_json(DEFAULT_API_BASE_URL, "/v1/midjourney/generations", key, "POST", payload)
        task_id = self._task_id(submission)
        if not task_id:
            raise RuntimeError(f"Midjourney submission did not return a task_id: {self._compact_json(submission)}")

        completed = self._poll_task(
            DEFAULT_API_BASE_URL,
            task_id,
            key,
            4,
            30 * 60,
        )
        image_urls = self._image_urls(completed)
        if not image_urls:
            raise RuntimeError(f"Midjourney task {task_id} completed without an image URL: {self._compact_json(completed)}")

        images = [self._download_image(url) for url in image_urls]
        return (self._to_batch(images),)

    @staticmethod
    def _imagine_payload(
        prompt: str, aspect_ratio: str, version: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"prompt": prompt, "version": version}
        if aspect_ratio:
            payload["size"] = aspect_ratio
        return payload

    def _poll_task(self, base_url: str, task_id: str, api_key: str, interval_seconds: int, timeout_seconds: int):
        deadline = time.monotonic() + timeout_seconds
        progress_bar = self._progress_bar()
        while time.monotonic() <= deadline:
            response = self._request_json(base_url, f"/v1/midjourney/tasks/{task_id}", api_key, "GET")
            body = self._response_body(response)
            status = str(body.get("status") or body.get("state") or body.get("task_status") or "").upper().replace("-", "_")
            progress = self._progress(body)
            if progress is not None and progress_bar is not None:
                progress_bar.update_absolute(progress, 100)
            print(f"[TD Midjourney] {task_id}: {status or 'POLLING'}" + (f" ({progress}%)" if progress is not None else ""))
            if status in SUCCESS_STATUSES or (not status and self._image_urls(response)):
                return response
            if status == "MODAL":
                raise RuntimeError(f"Midjourney task {task_id} requires an interactive MODAL action and cannot finish in this text-to-image node.")
            if status in FAILURE_STATUSES:
                message = self._message(body) or self._message(response) or "unknown failure"
                raise RuntimeError(f"Midjourney task {task_id} failed: {message}")
            time.sleep(interval_seconds)
        raise TimeoutError(f"Midjourney task {task_id} did not finish within {timeout_seconds // 60} minutes.")

    @staticmethod
    def _request_json(base_url: str, path: str, api_key: str, method: str, payload: dict[str, Any] | None = None):
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "ComfyUI-TDNodes/1.0 (+https://github.com/AICoderTudou/ComfyUI-TDNodes)",
        }
        try:
            response = requests.request(method, f"{base_url}{path}", headers=headers, json=payload, timeout=60)
        except requests.RequestException as error:
            raise RuntimeError(f"Cannot reach Aitudou API: {error}") from error
        if not response.ok:
            raise RuntimeError(f"Aitudou API HTTP {response.status_code}: {response.text[:500]}")
        try:
            result = response.json()
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Aitudou API returned invalid JSON: {response.text[:500]}") from error
        TD_MidjourneyTextToImage._raise_api_error(result)
        return result

    @staticmethod
    def _raise_api_error(raw: Any):
        if not isinstance(raw, dict):
            return
        code = raw.get("code")
        failed = code is False or (isinstance(code, int) and not isinstance(code, bool) and code not in (0, 200)) or (
            isinstance(code, str) and code.lower() in {"error", "failed", "failure", "unauthorized", "forbidden"}
        )
        if failed:
            raise RuntimeError(TD_MidjourneyTextToImage._message(raw) or f"Aitudou API error: {code}")

    @staticmethod
    def _task_id(raw: Any) -> str:
        for record in TD_MidjourneyTextToImage._records(raw):
            for key in ("task_id", "id"):
                value = record.get(key)
                if isinstance(value, (str, int)) and str(value).strip():
                    return str(value)
        return ""

    @staticmethod
    def _response_body(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        data = raw.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0]
        if isinstance(data, dict):
            return data
        return raw

    @staticmethod
    def _image_urls(raw: Any) -> list[str]:
        body = TD_MidjourneyTextToImage._response_body(raw)
        result = body.get("result") if isinstance(body.get("result"), dict) else {}
        urls: list[str] = []
        for record in (body, result):
            image_urls = record.get("image_urls")
            if isinstance(image_urls, list):
                urls.extend(url for url in image_urls if isinstance(url, str) and url.startswith(("http://", "https://")))
        if not urls:
            for record in (body, result):
                for key in ("grid_image_url", "result_url", "image_url", "url"):
                    url = record.get(key)
                    if isinstance(url, str) and url.startswith(("http://", "https://")):
                        urls.append(url)
        return list(dict.fromkeys(urls))

    @staticmethod
    def _download_image(url: str) -> torch.Tensor:
        try:
            response = requests.get(url, headers={"Accept": "image/*"}, timeout=60)
            response.raise_for_status()
            image_data = response.content
        except requests.RequestException as error:
            raise RuntimeError(f"Failed to download Midjourney result {url}: {error}") from error
        with Image.open(io.BytesIO(image_data)) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(pixels)).unsqueeze(0)

    @staticmethod
    def _to_batch(images: list[torch.Tensor]) -> torch.Tensor:
        if not images:
            raise RuntimeError("No Midjourney images were downloaded.")
        shape = images[0].shape[1:3]
        if any(image.shape[1:3] != shape for image in images):
            raise RuntimeError("Midjourney returned images with different dimensions, which cannot be combined into one ComfyUI IMAGE batch.")
        return torch.cat(images, dim=0)

    @staticmethod
    def _records(raw: Any):
        if isinstance(raw, dict):
            yield raw
            data = raw.get("data")
            if isinstance(data, dict):
                yield data
            elif isinstance(data, list):
                yield from (item for item in data if isinstance(item, dict))

    @staticmethod
    def _message(raw: Any) -> str:
        if not isinstance(raw, dict):
            return ""
        for key in ("error", "message", "msg", "fail_reason", "failure_reason"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, dict) and isinstance(value.get("message"), str):
                return value["message"].strip()
        return ""

    @staticmethod
    def _progress(raw: dict[str, Any]) -> int | None:
        for key in ("progress", "percentage", "percent"):
            value = raw.get(key)
            try:
                number = float(str(value).replace("%", ""))
            except (TypeError, ValueError):
                continue
            if 0 < number <= 1:
                number *= 100
            return max(0, min(100, round(number)))
        return None

    @staticmethod
    def _progress_bar():
        try:
            from comfy.utils import ProgressBar

            return ProgressBar(100)
        except ImportError:
            return None

    @staticmethod
    def _compact_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)[:1000]

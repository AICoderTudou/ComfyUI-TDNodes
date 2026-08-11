# TD Image Nodes

This package includes two text-to-image nodes backed by the [official Aitudou API](https://api.aitudou.net/). Both submit an asynchronous task, poll until it succeeds, download the signed result URL immediately, and return a ComfyUI `IMAGE` batch.

## TD Midjourney Text to Image

Uses Aitudou Midjourney Imagine: `POST /v1/midjourney/generations` followed by `GET /v1/midjourney/tasks/{task_id}`.

| Input | Description |
| --- | --- |
| `prompt` | Midjourney text prompt. |
| `api_key` | Aitudou API key. |
| `aspect_ratio` | Dropdown: `1:1`, `3:2`, `2:3`, `4:3`, `3:4`, `16:9`, `9:16`, or `21:9`. |
| `version` | Midjourney version: `8.2`, `8.1`, `7`, `6.1`, `6`, `5.2`, or `5.1`. |

Output: `images`

## TD GPT Image2 Text to Image

Uses `aitudou-image-g-v2-lowprice`: `POST /v1/image/generations` followed by `GET /v1/image/generations/{task_id}`.

| Input | Description |
| --- | --- |
| `prompt` | Image description. |
| `api_key` | Aitudou API key. |
| `aspect_ratio` | Dropdown: `1:1`, `3:2`, `2:3`, `4:3`, `3:4`, `16:9`, `9:16`, or `21:9`. |
| `resolution` | `1k`, `2k`, or `4k`. |

Output: `images`

## Notes

- Install dependencies from `requirements.txt`; the nodes require `requests`.
- The API key is a node input and can be saved into a ComfyUI workflow. Do not share a workflow containing an active key.
- Both nodes use the Aitudou API directly. Generation is billable; failed remote tasks follow the provider's refund policy.
- The HTTP client identifies this integration as `ComfyUI-TDNodes`, avoiding the Cloudflare block applied to Python's default `urllib` signature.

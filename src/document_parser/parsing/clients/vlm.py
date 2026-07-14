"""VLM client for figure captioning.

Assumes a Gemini-compatible endpoint fronted by the team's own Azure
deployment ("in4u azure"), called via the google-genai SDK pointed at a
custom base_url -- the same pattern the sibling skep_parser project uses for
Databricks-hosted Gemini (skep_parser.enrichers.vlm.VLMEnricher). Swap this
module if the real deployment's auth/endpoint shape differs.

Required env vars:
  AZURE_VLM_ENDPOINT   base URL of the Azure-hosted Gemini-compatible endpoint
  AZURE_VLM_API_KEY

TODO: DEFAULT_MODEL and the endpoint URL shape are placeholders ("gemini
flash lite 3.5" as named by the team) -- confirm both against the real in4u
Azure deployment once it exists.
"""

from __future__ import annotations

import os

from document_parser.core.exceptions import MissingDependencyError

DEFAULT_MODEL = "gemini-3.5-flash-lite"


class VLMClient:
    def __init__(self, model: str | None = None) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise MissingDependencyError(
                "VLM support requires the 'vlm' extra: pip install 'document-parser[vlm]'"
            ) from exc

        endpoint = os.environ.get("AZURE_VLM_ENDPOINT")
        api_key = os.environ.get("AZURE_VLM_API_KEY")
        if not endpoint or not api_key:
            raise MissingDependencyError(
                "AZURE_VLM_ENDPOINT / AZURE_VLM_API_KEY environment variables are required"
            )

        self._types = types
        self._model = model or DEFAULT_MODEL
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(base_url=endpoint),
        )

    def caption_image(self, image_bytes: bytes, prompt: str, mime_type: str = "image/png") -> str:
        types = self._types
        response = self._client.models.generate_content(
            model=self._model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        types.Part(text=prompt),
                    ],
                ),
            ],
            config=types.GenerateContentConfig(max_output_tokens=2048),
        )
        return response.text

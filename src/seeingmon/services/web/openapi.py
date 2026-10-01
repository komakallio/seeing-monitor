"""Render the OpenAPI description of the REST API as JSON, for `docs/openapi.json`.

FastAPI builds the description from the routes, and `seeingmon.records.api_schema` supplies the
record schemas. The description does not depend on any data, so the function builds the app over an
empty temporary store and a fake `core`. `seeingmon web openapi` prints the text, and with
`--output` it writes `docs/openapi.json`. A test fails when the committed file differs from this
text, so regenerate the file after you change a route, a model, or a record declaration:

    seeingmon web openapi --output docs/openapi.json
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from seeingmon.services.web.app import create_app
from seeingmon.services.web.config import WebSettings
from seeingmon.services.web.core_client import FakeCoreClient
from seeingmon.services.web.images import ImageStore
from seeingmon.store.db import Store, StoreReader
from seeingmon.store.layout import DataLayout

COMMAND = "seeingmon web openapi --output docs/openapi.json"


def render_openapi() -> str:
    """The OpenAPI document as JSON text with sorted keys, so that the file is stable."""
    settings = WebSettings()
    with tempfile.TemporaryDirectory(prefix="seeingmon-openapi-") as folder:
        layout = DataLayout(Path(folder))
        layout.create()
        with Store.open(layout.db_path), StoreReader.open(layout.db_path) as reader:
            app = create_app(
                settings, reader, ImageStore(layout, settings.images), FakeCoreClient()
            )
            document = app.openapi()
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

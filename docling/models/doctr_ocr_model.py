import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Type
import numpy as np
from io import BytesIO

from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    OcrOptions,
    DoctrOcrOptions,
)
from docling.datamodel.settings import settings
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class DoctrOcrModel(BaseOcrModel):
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: OcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )

        if self.enabled:
            try:
                from doctr.models import ocr_predictor
            except ImportError:
                raise ImportError(
                    "The 'python-doctr' library is not installed. Install it via `pip install python-doctr`."
                )

            _log.debug("Initializing Doctr OCR engine")
            self.model = ocr_predictor(pretrained=True)
        else:
            self.model = None

    def __call__(self, conv_res: ConversionResult, page_batch: Iterable[Page]) -> Iterable[Page]:
        if not self.enabled or self.model is None:
            yield from page_batch
            return

        from doctr.io import DocumentFile

        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "ocr"):
                    pil_image = page._backend.get_page_image(scale=1).convert("RGB")
                    width, height = pil_image.size

                    buf = BytesIO()
                    pil_image.save(buf, format="PNG")
                    img_bytes = buf.getvalue()

                    doc = DocumentFile.from_images([img_bytes])
                    result = self.model(doc)

                    all_cells = []
                    if len(result.pages) > 0:
                        doc_page = result.pages[0]
                        for block in doc_page.blocks:
                            for line in block.lines:
                                line_text = " ".join(w.value for w in line.words)
                                (left_norm, top_norm), (right_norm, bottom_norm) = line.geometry

                                # Scale normalized coords to pixel coords (no flipping)
                                left = left_norm * width
                                right = right_norm * width
                                top = top_norm * height
                                bottom = bottom_norm * height

                                if line.words:
                                    conf = float(np.mean([w.confidence for w in line.words]))
                                else:
                                    conf = 0.0

                                all_cells.append(
                                    TextCell(
                                        index=len(all_cells),
                                        text=line_text,
                                        orig=line_text,
                                        from_ocr=True,
                                        confidence=conf,
                                        rect=BoundingRectangle.from_bounding_box(
                                            BoundingBox.from_tuple(
                                                coord=(left, top, right, bottom),
                                                origin=CoordOrigin.TOPLEFT,
                                            )
                                        ),
                                    )
                                )

                    page.cells = self.post_process_cells(all_cells, page.cells)

                if settings.debug.visualize_ocr:
                    self.draw_ocr_rects_and_cells(conv_res, page, [])

                yield page

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return DoctrOcrOptions

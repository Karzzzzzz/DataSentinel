
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from docx import Document
from pptx import Presentation


def extract_document_data(filename: str, file_bytes: bytes) -> dict:
    """
    Extracts content from PDF, DOCX, PPTX and TXT files.

    Preserves document structure such as pages,
    paragraphs, tables and presentation slides.
    """

    extension = Path(filename).suffix.lower()

    if not file_bytes:
        return {
            "success": False,
            "filename": filename,
            "format": extension,
            "content": {},
            "error": "The uploaded file is empty."
        }

    try:

        # ======================================
        # 1. PDF EXTRACTION
        # ======================================

        if extension == ".pdf":

            reader = PdfReader(
                BytesIO(file_bytes),
                strict=False
            )

            if reader.is_encrypted:
                raise ValueError(
                    "Password-protected PDFs are not supported yet."
                )

            pages = []

            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):

                page_text = page.extract_text() or ""

                pages.append({
                    "page_number": page_number,
                    "text": page_text
                })

            full_text = "\n".join(
                page["text"] for page in pages
            )

            content = {
                "page_count": len(pages),
                "pages": pages,
                "full_text": full_text,
                "has_readable_text": bool(full_text.strip())
            }

            detected_format = "pdf"

        # ======================================
        # 2. WORD DOCUMENT EXTRACTION
        # ======================================

        elif extension == ".docx":

            document = Document(BytesIO(file_bytes))

            paragraphs = []

            for paragraph in document.paragraphs:

                text = paragraph.text.strip()

                if text:
                    paragraphs.append({
                        "text": text,
                        "style": paragraph.style.name
                    })

            tables = []

            for table_number, table in enumerate(
                document.tables,
                start=1
            ):

                rows = []

                for row in table.rows:

                    rows.append([
                        cell.text.strip()
                        for cell in row.cells
                    ])

                tables.append({
                    "table_number": table_number,
                    "rows": rows
                })

            # Extract headers and footers
            headers = []
            footers = []

            for section in document.sections:

                if section.header.paragraphs:
                    headers.extend([
                        p.text.strip()
                        for p in section.header.paragraphs
                        if p.text.strip()
                    ])

                if section.footer.paragraphs:
                    footers.extend([
                        p.text.strip()
                        for p in section.footer.paragraphs
                        if p.text.strip()
                    ])

            full_text = "\n".join(
                item["text"] for item in paragraphs
            )

            content = {
                "paragraphs": paragraphs,
                "tables": tables,
                "headers": headers,
                "footers": footers,
                "paragraph_count": len(paragraphs),
                "table_count": len(tables),
                "full_text": full_text
            }

            detected_format = "docx"

        # ======================================
        # 3. POWERPOINT EXTRACTION
        # ======================================

        elif extension == ".pptx":

            presentation = Presentation(
                BytesIO(file_bytes)
            )

            slides = []

            for slide_number, slide in enumerate(
                presentation.slides,
                start=1
            ):

                slide_text = []
                slide_tables = []
                embedded_image_count = 0

                for shape in slide.shapes:

                    # Extract text
                    if shape.has_text_frame:

                        text = shape.text.strip()

                        if text:
                            slide_text.append(text)

                    # Extract tables
                    if shape.has_table:

                        table_data = []

                        for row in shape.table.rows:

                            table_data.append([
                                cell.text.strip()
                                for cell in row.cells
                            ])

                        slide_tables.append(table_data)

                    # Count embedded images
                    if shape.shape_type == 13:
                        embedded_image_count += 1

                slides.append({
                    "slide_number": slide_number,
                    "text": slide_text,
                    "tables": slide_tables,
                    "embedded_image_count": embedded_image_count
                })

            full_text = "\n".join(
                text
                for slide in slides
                for text in slide["text"]
            )

            content = {
                "slide_count": len(slides),
                "slides": slides,
                "full_text": full_text
            }

            detected_format = "pptx"

        # ======================================
        # 4. TEXT FILE EXTRACTION
        # ======================================

        elif extension == ".txt":

            text = None

            for encoding in (
                "utf-8-sig",
                "utf-8",
                "utf-16",
                "cp1252"
            ):

                try:
                    text = file_bytes.decode(encoding)
                    break

                except UnicodeDecodeError:
                    continue

            if text is None:
                raise ValueError(
                    "Unable to decode the text file."
                )

            content = {
                "full_text": text,
                "character_count": len(text)
            }

            detected_format = "txt"

        else:

            return {
                "success": False,
                "filename": filename,
                "format": extension,
                "content": {},
                "error": "Unsupported document format."
            }

        # ======================================
        # 5. RETURN RESULTS
        # ======================================

        return {
            "success": True,
            "filename": filename,
            "format": detected_format,
            "content": content,
            "error": None
        }

    except Exception as error:

        return {
            "success": False,
            "filename": filename,
            "format": extension,
            "content": {},
            "error": str(error)
        }

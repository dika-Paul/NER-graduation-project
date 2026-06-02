from __future__ import annotations

import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "origional_pdf"
OUTPUT_DIR = BASE_DIR / "excel_data"
OUTPUT_FILE = OUTPUT_DIR / "pdf_abstracts.xlsx"
MAX_PAGES_FOR_ABSTRACT = 2


ABSTRACT_HEADING_RE = re.compile(r"^abstract(?:\s*[:.-]\s*(.*))?$", re.IGNORECASE)
SECTION_HEADING_RE = re.compile(
    r"^(keywords?|index terms?|introduction|1\s+introduction|background\s*&\s*summary|"
    r"related work|materials?\s+and\s+methods?|methods?|results?|discussion)\b",
    re.IGNORECASE,
)
INVALID_XML_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def normalize_ligatures(text: str) -> str:
    return (
        text.replace("ﬁ", "fi")
        .replace("ﬂ", "fl")
        .replace("ﬀ", "ff")
        .replace("ﬃ", "ffi")
        .replace("ﬄ", "ffl")
    )


def clean_line(line: str) -> str:
    line = normalize_ligatures(line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def clean_abstract(text: str) -> str:
    text = normalize_ligatures(text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" - ", "-")
    return text.strip()


def read_first_pages(pdf_path: Path, max_pages: int = MAX_PAGES_FOR_ABSTRACT) -> list[str]:
    reader = PdfReader(str(pdf_path))
    page_texts: list[str] = []
    for page in reader.pages[:max_pages]:
        page_texts.append(page.extract_text() or "")
    return page_texts


def extract_lines(page_texts: list[str]) -> list[str]:
    lines: list[str] = []
    for page_text in page_texts:
        for raw_line in page_text.splitlines():
            line = clean_line(raw_line)
            if line:
                lines.append(line)
    return lines


def is_metadata_line(line: str) -> bool:
    lower = line.lower()
    return (
        "https://doi.org" in lower
        or lower.startswith("www.")
        or lower.startswith("npj ")
        or lower.startswith("scientific data")
        or lower.startswith("journal of ")
        or lower.startswith("vol.:")
    )


def extract_by_abstract_heading(lines: list[str]) -> str:
    for index, line in enumerate(lines):
        match = ABSTRACT_HEADING_RE.match(line)
        if not match:
            continue

        collected: list[str] = []
        if match.group(1):
            collected.append(match.group(1))

        for next_line in lines[index + 1 :]:
            if SECTION_HEADING_RE.match(next_line):
                break
            if is_metadata_line(next_line):
                continue
            collected.append(next_line)

        abstract = clean_abstract(" ".join(collected))
        if len(abstract.split()) >= 20:
            return abstract
    return ""


def extract_from_front_matter(lines: list[str]) -> str:
    section_index = len(lines)
    for index, line in enumerate(lines):
        if SECTION_HEADING_RE.match(line):
            section_index = index
            break

    front_lines = lines[:section_index]
    start_index: int | None = None
    for index, line in enumerate(front_lines):
        lower = line.lower()
        if "✉" in line or "@" in line or "published online" in lower or lower.startswith("received:"):
            start_index = index + 1

    if start_index is None:
        return ""

    collected: list[str] = []
    for line in front_lines[start_index:]:
        if is_metadata_line(line):
            break
        if not collected and line.startswith("& "):
            continue
        collected.append(line)

    abstract = clean_abstract(" ".join(collected))
    if len(abstract.split()) >= 20:
        return abstract
    return ""


def extract_abstract(pdf_path: Path) -> str:
    page_texts = read_first_pages(pdf_path)
    lines = extract_lines(page_texts)
    return extract_by_abstract_heading(lines) or extract_from_front_matter(lines)


def xml_text(value: object) -> str:
    text = INVALID_XML_RE.sub("", str(value))
    return escape(text, {'"': "&quot;"})


def column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def cell_xml(row_index: int, column_index: int, value: object, style: int = 0) -> str:
    ref = f"{column_name(column_index)}{row_index}"
    return f'<c r="{ref}" s="{style}" t="inlineStr"><is><t>{xml_text(value)}</t></is></c>'


def sheet_xml(rows: list[list[object]]) -> str:
    row_nodes: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        style = 1 if row_index == 1 else 2
        cells = "".join(
            cell_xml(row_index, column_index, value, style)
            for column_index, value in enumerate(row, start=1)
        )
        height = ' ht="90" customHeight="1"' if row_index > 1 else ""
        row_nodes.append(f'<row r="{row_index}"{height}>{cells}</row>')

    dimension = f"A1:C{len(rows)}"
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <dimension ref="{dimension}"/>
  <cols>
    <col min="1" max="1" width="8" customWidth="1"/>
    <col min="2" max="2" width="32" customWidth="1"/>
    <col min="3" max="3" width="110" customWidth="1"/>
  </cols>
  <sheetData>
    {''.join(row_nodes)}
  </sheetData>
</worksheet>'''


def styles_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><name val="Calibri"/></font>
  </fonts>
  <fills count="2">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
  </fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="3">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1">
      <alignment vertical="top" wrapText="1"/>
    </xf>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''


def write_xlsx(output_path: Path, rows: list[list[object]]) -> None:
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    files = {
        "[Content_Types].xml": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>''',
        "_rels/.rels": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>''',
        "docProps/app.xml": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Python</Application>
</Properties>''',
        "docProps/core.xml": f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:creator>extract_pdf_abstracts_to_excel.py</dc:creator>
  <cp:lastModifiedBy>extract_pdf_abstracts_to_excel.py</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{created}</dcterms:modified>
</cp:coreProperties>''',
        "xl/workbook.xml": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="abstracts" sheetId="1" r:id="rId1"/></sheets>
</workbook>''',
        "xl/_rels/workbook.xml.rels": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>''',
        "xl/styles.xml": styles_xml(),
        "xl/worksheets/sheet1.xml": sheet_xml(rows),
    }

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, content in files.items():
            archive.writestr(path, content)


def main() -> None:
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {PDF_DIR}")

    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    rows: list[list[object]] = [["id", "pdf_file", "abstract"]]

    for index, pdf_path in enumerate(pdf_paths, start=1):
        abstract = extract_abstract(pdf_path)
        rows.append([index, pdf_path.name, abstract])
        print(f"[{index}/{len(pdf_paths)}] {pdf_path.name}: {len(abstract)} chars")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_xlsx(OUTPUT_FILE, rows)
    print(f"Saved Excel file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

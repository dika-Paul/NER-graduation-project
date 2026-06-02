from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import zipfile
from html import unescape
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "excel_data"
OUTPUT_FILE = OUTPUT_DIR / "openalex_materials_abstracts.xlsx"

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
MATERIALS_SCIENCE_CONCEPT_ID = "C192562407"
TARGET_COUNT = 800
SAMPLE_SIZE = 1000
RANDOM_SEED = 42
PER_PAGE = 200
MIN_ABSTRACT_WORDS = 50

FILTERS = ",".join(
    [
        "has_abstract:true",
        f"concepts.id:{MATERIALS_SCIENCE_CONCEPT_ID}",
        "language:en",
        "type:article",
        "primary_location.source.type:journal",
        "from_publication_date:2020-01-01",
        "to_publication_date:2026-05-09",
    ]
)

SELECT_FIELDS = ",".join(
    [
        "id",
        "doi",
        "display_name",
        "publication_year",
        "publication_date",
        "abstract_inverted_index",
        "cited_by_count",
        "primary_location",
        "concepts",
        "type",
        "language",
    ]
)

INVALID_XML_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def request_json(url: str, retries: int = 3) -> dict[str, Any]:
    headers = {
        "User-Agent": "NRE_project_materials_abstract_collector/1.0",
    }
    request = urllib.request.Request(url, headers=headers)
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)

    raise RuntimeError(f"OpenAlex request failed after {retries} attempts: {last_error}")


def abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""

    max_position = max(position for positions in index.values() for position in positions)
    words = [""] * (max_position + 1)
    for word, positions in index.items():
        for position in positions:
            words[position] = word

    abstract = " ".join(words)
    abstract = re.sub(r"\s+([,.;:!?%)\]])", r"\1", abstract)
    abstract = re.sub(r"([(\\[])\s+", r"\1", abstract)
    abstract = re.sub(r"\s+", " ", abstract)
    return abstract.strip()


def clean_text(text: str) -> str:
    text = unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_source_name(work: dict[str, Any]) -> str:
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    return source.get("display_name") or ""


def get_landing_page_url(work: dict[str, Any]) -> str:
    primary_location = work.get("primary_location") or {}
    return primary_location.get("landing_page_url") or work.get("doi") or work.get("id") or ""


def get_top_concepts(work: dict[str, Any], limit: int = 5) -> str:
    concepts = work.get("concepts") or []
    names = [concept.get("display_name", "") for concept in concepts[:limit]]
    return "; ".join(name for name in names if name)


def build_openalex_url(page: int) -> str:
    params = {
        "filter": FILTERS,
        "sample": str(SAMPLE_SIZE),
        "seed": str(RANDOM_SEED),
        "per-page": str(PER_PAGE),
        "page": str(page),
        "select": SELECT_FIELDS,
    }
    return f"{OPENALEX_WORKS_URL}?{urllib.parse.urlencode(params)}"


def collect_records(target_count: int = TARGET_COUNT) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    page = 1

    while len(records) < target_count:
        data = request_json(build_openalex_url(page))
        results = data.get("results") or []
        meta = data.get("meta") or {}

        if not results:
            break

        for work in results:
            openalex_id = work.get("id") or ""
            if not openalex_id or openalex_id in seen_ids:
                continue

            abstract = abstract_from_inverted_index(work.get("abstract_inverted_index"))
            if len(abstract.split()) < MIN_ABSTRACT_WORDS:
                continue

            seen_ids.add(openalex_id)
            records.append(
                {
                    "id": len(records) + 1,
                    "title": clean_text(work.get("display_name") or ""),
                    "abstract": abstract,
                    "publication_year": work.get("publication_year") or "",
                    "publication_date": work.get("publication_date") or "",
                    "doi": work.get("doi") or "",
                    "source": get_source_name(work),
                    "cited_by_count": work.get("cited_by_count") or 0,
                    "openalex_id": openalex_id,
                    "landing_page_url": get_landing_page_url(work),
                    "top_concepts": get_top_concepts(work),
                }
            )

            if len(records) >= target_count:
                break

        print(f"Collected {len(records)}/{target_count} records")

        if page * PER_PAGE >= int(meta.get("count") or 0):
            break
        page += 1
        time.sleep(0.2)

    return records


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


def sheet_xml(rows: list[list[object]], column_widths: list[int]) -> str:
    row_nodes: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        style = 1 if row_index == 1 else 2
        cells = "".join(
            cell_xml(row_index, column_index, value, style)
            for column_index, value in enumerate(row, start=1)
        )
        height = ' ht="95" customHeight="1"' if row_index > 1 else ' ht="24" customHeight="1"'
        row_nodes.append(f'<row r="{row_index}"{height}>{cells}</row>')

    cols = "".join(
        f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
        for index, width in enumerate(column_widths, start=1)
    )
    last_cell = f"{column_name(len(rows[0]))}{len(rows)}"
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <dimension ref="A1:{last_cell}"/>
  <sheetViews>
    <sheetView workbookViewId="0">
      <pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>
      <selection pane="bottomLeft"/>
    </sheetView>
  </sheetViews>
  <cols>{cols}</cols>
  <sheetData>{''.join(row_nodes)}</sheetData>
  <autoFilter ref="A1:{last_cell}"/>
</worksheet>'''


def styles_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><name val="Calibri"/></font>
  </fonts>
  <fills count="3">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFEAF2F8"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="3">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1">
      <alignment vertical="center" wrapText="1"/>
    </xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1">
      <alignment vertical="top" wrapText="1"/>
    </xf>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''


def write_xlsx(output_path: Path, rows: list[list[object]], column_widths: list[int]) -> None:
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
  <dc:creator>collect_openalex_materials_abstracts_to_excel.py</dc:creator>
  <cp:lastModifiedBy>collect_openalex_materials_abstracts_to_excel.py</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{created}</dcterms:modified>
</cp:coreProperties>''',
        "xl/workbook.xml": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="materials_abstracts" sheetId="1" r:id="rId1"/></sheets>
</workbook>''',
        "xl/_rels/workbook.xml.rels": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>''',
        "xl/styles.xml": styles_xml(),
        "xl/worksheets/sheet1.xml": sheet_xml(rows, column_widths),
    }

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, content in files.items():
            archive.writestr(path, content)


def records_to_rows(records: list[dict[str, Any]]) -> list[list[object]]:
    headers = [
        "id",
        "title",
        "abstract",
        "publication_year",
        "publication_date",
        "doi",
        "source",
        "cited_by_count",
        "openalex_id",
        "landing_page_url",
        "top_concepts",
    ]
    return [headers] + [[record.get(header, "") for header in headers] for record in records]


def main() -> None:
    records = collect_records()
    if len(records) < TARGET_COUNT:
        raise RuntimeError(f"Only collected {len(records)} records; expected {TARGET_COUNT}.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = records_to_rows(records)
    column_widths = [8, 55, 120, 16, 18, 34, 36, 16, 34, 52, 52]
    write_xlsx(OUTPUT_FILE, rows, column_widths)
    print(f"Saved Excel file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

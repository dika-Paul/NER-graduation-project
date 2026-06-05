import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree

from ..graph_state import GraphState


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
SHEET_NAME = "materials_abstracts"
MAX_SAMPLE_TOKENS = 100
TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+(?:[+\-−‐‑‒–—./][A-Za-z0-9]+)*|[^\s]",
    re.UNICODE,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def _namespaced(tag: str) -> str:
    return f"{{{MAIN_NS}}}{tag}"


def _column_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        raise ValueError(f"Invalid Excel cell reference: {cell_ref}")

    index = 0
    for char in match.group(1):
        index = index * 26 + ord(char) - ord("A") + 1
    return index - 1


def _read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []

    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    strings = []
    for item in root.findall(_namespaced("si")):
        parts = [node.text or "" for node in item.findall(f".//{_namespaced('t')}")]
        strings.append("".join(parts))
    return strings


def _sheet_path_for_name(archive: zipfile.ZipFile, sheet_name: str) -> str:
    workbook_root = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    rels_root = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))

    rel_targets = {
        relationship.attrib["Id"]: relationship.attrib["Target"]
        for relationship in rels_root.findall(f"{{{PKG_REL_NS}}}Relationship")
    }

    sheets = workbook_root.find(_namespaced("sheets"))
    if sheets is None:
        raise ValueError("Workbook does not contain sheets.")

    for sheet in sheets.findall(_namespaced("sheet")):
        if sheet.attrib.get("name") != sheet_name:
            continue

        relationship_id = sheet.attrib.get(f"{{{REL_NS}}}id")
        if not relationship_id or relationship_id not in rel_targets:
            raise ValueError(f"Cannot resolve worksheet path for sheet: {sheet_name}")

        target = rel_targets[relationship_id]
        if target.startswith("/"):
            return target.lstrip("/")
        return str(PurePosixPath("xl") / target)

    raise ValueError(f"Workbook does not contain sheet: {sheet_name}")


def _cell_text(cell: ElementTree.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(f".//{_namespaced('t')}"))

    value = cell.find(_namespaced("v"))
    if value is None or value.text is None:
        return ""

    if cell_type == "s":
        shared_index = int(value.text)
        if shared_index >= len(shared_strings):
            return ""
        return shared_strings[shared_index]

    return value.text


def _read_sheet_rows(archive: zipfile.ZipFile, sheet_path: str) -> list[list[str]]:
    shared_strings = _read_shared_strings(archive)
    root = ElementTree.fromstring(archive.read(sheet_path))
    sheet_data = root.find(_namespaced("sheetData"))
    if sheet_data is None:
        return []

    rows = []
    for row in sheet_data.findall(_namespaced("row")):
        row_values: list[str] = []
        for cell in row.findall(_namespaced("c")):
            cell_ref = cell.attrib.get("r", "")
            column_index = _column_index(cell_ref)
            while len(row_values) <= column_index:
                row_values.append("")
            row_values[column_index] = _cell_text(cell, shared_strings)
        rows.append(row_values)

    return rows


def read_openalex_excel_records(path: str | Path) -> list[dict[str, str]]:
    excel_path = Path(path).expanduser()
    if not excel_path.exists() or not excel_path.is_file():
        raise FileNotFoundError(f"Excel pool does not exist or is not a file: {excel_path}")

    with zipfile.ZipFile(excel_path) as archive:
        sheet_path = _sheet_path_for_name(archive, SHEET_NAME)
        rows = _read_sheet_rows(archive, sheet_path)

    if not rows:
        return []

    headers = [str(value).strip() for value in rows[0]]
    header_index = {header: index for index, header in enumerate(headers)}
    required_headers = ("id", "title", "abstract", "openalex_id")
    missing_headers = [header for header in required_headers if header not in header_index]
    if missing_headers:
        raise ValueError(f"Excel pool is missing required headers: {missing_headers}")

    records = []
    for row in rows[1:]:
        record = {}
        for header in headers:
            index = header_index[header]
            record[header] = str(row[index]).strip() if index < len(row) else ""

        if record.get("title") or record.get("abstract"):
            records.append(record)

    return records


def _paper_id(record: dict[str, str]) -> str:
    openalex_id = record.get("openalex_id", "").strip()
    raw_id = openalex_id.rstrip("/").rsplit("/", 1)[-1] if openalex_id else ""
    if not raw_id:
        raw_id = record.get("id", "").strip()

    paper_id = re.sub(r"[^A-Za-z0-9_-]+", "-", raw_id).strip("-")
    if not paper_id:
        raise ValueError(f"Cannot build paper id from record: {record}")
    return paper_id


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _split_sentences(text: str) -> list[str]:
    normalized_text = _normalize_text(text)
    if not normalized_text:
        return []
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(normalized_text) if part.strip()]


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def _sentence_samples(
    *,
    paper_id: str,
    source_name: str,
    text: str,
) -> list[tuple[str, list[str]]]:
    samples = []
    for sentence in _split_sentences(text):
        tokens = _tokenize(sentence)
        if not tokens:
            continue

        if len(tokens) <= MAX_SAMPLE_TOKENS:
            samples.append((sentence, tokens))
            continue

        for start in range(0, len(tokens), MAX_SAMPLE_TOKENS):
            token_chunk = tokens[start:start + MAX_SAMPLE_TOKENS]
            samples.append((" ".join(token_chunk), token_chunk))

    return samples


def _record_to_batch_samples(record: dict[str, str]) -> dict[str, dict[str, Any]]:
    paper_id = _paper_id(record)
    batch_samples = {}

    for source_name in ("title", "abstract"):
        text = record.get(source_name, "")
        source_samples = _sentence_samples(
            paper_id=paper_id,
            source_name=source_name,
            text=text,
        )

        source_prefix = "ttl" if source_name == "title" else "abs"
        for index, (sample_text, tokens) in enumerate(source_samples):
            sample_id = f"openalex-{paper_id}-{source_prefix}-{index:03d}"
            batch_samples[sample_id] = {
                "text": sample_text,
                "tokens": tokens,
                "paper_id": paper_id,
                "source": source_name,
                "source_index": index,
            }

    return batch_samples


def get_excel_batch_node(graph_state: GraphState) -> dict:
    """
    Read the next sequential paper batch from the OpenAlex Excel pool and turn
    title/abstract text into sentence-level samples for the existing NER/LLM flow.
    """
    paper_batch_size = int(graph_state.paper_batch_size or 100)
    if paper_batch_size <= 0:
        raise ValueError("paper_batch_size must be a positive integer.")

    records = read_openalex_excel_records(graph_state.unlabeled_pool_path)
    processed_paper_ids = list(graph_state.processed_paper_ids)
    processed_paper_id_set = set(processed_paper_ids)

    selected_records = []
    selected_paper_ids = []
    for record in records:
        paper_id = _paper_id(record)
        if paper_id in processed_paper_id_set:
            continue

        selected_records.append(record)
        selected_paper_ids.append(paper_id)
        processed_paper_id_set.add(paper_id)
        if len(selected_records) >= paper_batch_size:
            break

    current_batch = {}
    for record in selected_records:
        current_batch.update(_record_to_batch_samples(record))

    return {
        "current_batch": current_batch,
        "processed_paper_ids": processed_paper_ids + selected_paper_ids,
        "total_paper_count": len(records),
    }

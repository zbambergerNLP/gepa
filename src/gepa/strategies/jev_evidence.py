"""Factor repeated Tau KB document bodies without dropping recorded evidence."""

from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy
from typing import Any

TAU_KB_REFERENCE_FORMAT = "tau_kb_document_references_v1"
TAU_KB_JEV_POLICY = "jev_joint_action_section_v7_tau_kb_document_references"
TAU_KB_REFERENCE_GUIDANCE = (
    "A record with format=tau_kb_document_references_v1 contains its original record and a documents table. "
    "In a tool message, content.parts replaces only that message's content string. Read parts in order: "
    "a string is literal text; {document: N} inserts the full content of documents[N]. "
    "N is a zero-based index in that record's table. "
    "Each table entry retains its document_id. References preserve every original occurrence, message position, "
    "and order. Treat each reference as the complete document text at that position, not as missing evidence. "
    "Unwrapped records and other fields keep their original meaning."
)
TAU_KB_REFERENCE_CONTRACT = {
    "format": TAU_KB_REFERENCE_FORMAT,
    "applicability": "explicit Tau banking single-component JEV opt-in; other benchmarks remain verbatim",
    "matching": "same document ID and exact body within one original record",
    "scope": "JEV only; native traces and other optimizer roles remain unchanged",
    "reconstruction": "exact original structured record, including all message strings and occurrence order",
    "unsupported_tool_format": "retain verbatim; unchanged size guard still applies",
    "record_boundaries": "unchanged; one encoded record counts as one original record",
    "model_equivalence": "not claimed; model-visible evidence representation changes",
    "guidance": TAU_KB_REFERENCE_GUIDANCE,
}
_HEADER = re.compile(r"(?m)^(\d+)\. [^\n]*\n   ID: (doc_[^\n]+)\n   Score: -?\d+\.\d{4}\n   Content: ")
_TIMING = re.compile(r"\n\n\[Timing: retrieval=\d+ms(?:, reranking=\d+ms)?, total=\d+ms\]\Z")


class EvidenceReferenceError(ValueError):
    """Reject an invalid reference or an incompatible evidence shape."""


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _tau_record(record: Any) -> bool:
    return (
        isinstance(record, dict)
        and isinstance(record.get("Inputs"), dict)
        and record["Inputs"].get("domain") == "banking_knowledge"
        and isinstance(record.get("Generated Outputs"), list)
        and all(isinstance(message, dict) for message in record["Generated Outputs"])
    )


def _document_spans(content: str) -> list[tuple[int, int, str, str]]:
    """Recognize only complete native ranked KB returns; keep other text intact."""
    headers, timing = list(_HEADER.finditer(content)), _TIMING.search(content)
    if not headers or headers[0].start() != 0 or timing is None:
        return []
    if [int(header[1]) for header in headers] != list(range(1, len(headers) + 1)):
        return []
    spans = []
    for index, header in enumerate(headers):
        following = headers[index + 1].start() if index + 1 < len(headers) else timing.start()
        separator = "\n\n" if index + 1 < len(headers) else "\n"
        end = following - len(separator)
        if end < header.end() or content[end:following] != separator:
            return []
        spans.append((header.end(), end, header[2], content[header.end() : end]))
    return spans


def decode_tau_record(encoded: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct one encoded record, rejecting invalid or unused references."""
    if set(encoded) != {"format", "record", "documents"} or encoded["format"] != TAU_KB_REFERENCE_FORMAT:
        raise EvidenceReferenceError("Unknown Tau evidence reference envelope")
    record, documents = deepcopy(encoded["record"]), encoded["documents"]
    if not _tau_record(record) or not isinstance(documents, list) or not documents:
        raise EvidenceReferenceError("Missing Tau record or document table")
    pairs = []
    for document in documents:
        if (
            not isinstance(document, dict)
            or set(document) != {"document_id", "content"}
            or not isinstance(document["document_id"], str)
            or not document["document_id"].startswith("doc_")
            or "\n" in document["document_id"]
            or not isinstance(document["content"], str)
        ):
            raise EvidenceReferenceError("Invalid Tau document table entry")
        pairs.append((document["document_id"], document["content"]))
    if len(set(pairs)) != len(pairs):
        raise EvidenceReferenceError("Duplicate Tau document table entry")
    used = Counter()
    for message in record["Generated Outputs"]:
        content = message.get("content")
        if not isinstance(content, dict):
            continue
        if message.get("role") != "tool" or set(content) != {"parts"} or not isinstance(content["parts"], list):
            raise EvidenceReferenceError("Invalid Tau referenced message")
        text, references = [], []
        position = 0
        for part in content["parts"]:
            if isinstance(part, str):
                text.append(part)
                position += len(part)
            elif isinstance(part, dict) and set(part) == {"document"}:
                index = part["document"]
                if type(index) is not int or not 0 <= index < len(documents):
                    raise EvidenceReferenceError("Missing or invalid Tau document reference")
                body = documents[index]["content"]
                references.append((position, position + len(body), *pairs[index]))
                used[index] += 1
                text.append(body)
                position += len(body)
            else:
                raise EvidenceReferenceError("Invalid Tau message content part")
        message["content"] = "".join(text)
        spans = _document_spans(message["content"])
        if not references or any(reference not in spans for reference in references):
            raise EvidenceReferenceError("Tau reference differs from its document ID or body position")
    if set(used) != set(range(len(documents))) or any(count < 2 for count in used.values()):
        raise EvidenceReferenceError("Unused or unrepeated Tau document reference")
    return record


def encode_tau_record(record: dict[str, Any]) -> dict[str, Any]:
    """Encode repeated document bodies and prove exact reconstruction locally."""
    if not _tau_record(record):
        raise EvidenceReferenceError("Tau KB references require banking_knowledge message records")
    occurrences, counts = {}, Counter()
    for index, message in enumerate(record["Generated Outputs"]):
        content = message.get("content")
        if isinstance(content, dict):
            raise EvidenceReferenceError("Tau input already contains structured message content")
        if message.get("role") == "tool" and isinstance(content, str):
            spans = _document_spans(content)
            occurrences[index] = spans
            counts.update((doc_id, body) for _, _, doc_id, body in spans)
    references, documents = {}, []
    for spans in occurrences.values():
        for _, _, doc_id, body in spans:
            key = (doc_id, body)
            if counts[key] > 1 and key not in references:
                references[key] = len(documents)
                documents.append({"document_id": doc_id, "content": body})
    if not documents:
        return deepcopy(record)
    result = deepcopy(record)
    for index, spans in occurrences.items():
        content = record["Generated Outputs"][index]["content"]
        parts, position = [], 0
        for start, end, doc_id, body in spans:
            key = (doc_id, body)
            if key in references:
                parts.extend([content[position:start], {"document": references[key]}])
                position = end
        if parts:
            parts.append(content[position:])
            result["Generated Outputs"][index]["content"] = {"parts": parts}
    encoded = {"format": TAU_KB_REFERENCE_FORMAT, "record": result, "documents": documents}
    if decode_tau_record(encoded) != record:
        raise EvidenceReferenceError("Tau evidence did not reconstruct exactly")
    return encoded if len(_json(encoded).encode()) < len(_json(record).encode()) else deepcopy(record)


def factor_tau_evidence(traces: str) -> tuple[str, bool]:
    """Factor explicitly selected Tau evidence without changing record counts."""

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise EvidenceReferenceError("Duplicate JSON key in Tau evidence")
            result[key] = value
        return result

    try:
        records = json.loads(traces, object_pairs_hook=unique_keys)
    except (TypeError, json.JSONDecodeError) as exc:
        raise EvidenceReferenceError("Tau KB references require a JSON record array") from exc
    if not isinstance(records, list) or not records:
        raise EvidenceReferenceError("Tau KB references require a nonempty record array")
    encoded = [encode_tau_record(record) for record in records]
    changed = encoded != records
    return (_json(encoded), True) if changed else (traces, False)

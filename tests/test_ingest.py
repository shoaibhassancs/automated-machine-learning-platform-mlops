"""CSV intake tests, using small invented CSVs only (no real data)."""

from __future__ import annotations

import codecs
from pathlib import Path

import pytest

from automl_ops.ingest import IngestError, IngestLimits, prepare_csv, read_limited, sha256_bytes

CANONICAL = 'id,name,note\n1,Ana,"hello, world"\n2,Bo,\n3,Cy,"two\nlines"\n4,Dé,ünïcode\n'


def codes(data: bytes, limits: IngestLimits | None = None) -> str:
    with pytest.raises(IngestError) as info:
        if limits is None:
            prepare_csv(data)
        else:
            prepare_csv(data, limits)
    return info.value.code


# --- accepted input ------------------------------------------------------------------------


def test_valid_csv_is_accepted_and_cells_are_kept_exactly() -> None:
    prepared = prepare_csv(CANONICAL.encode())
    assert prepared.rows == 4
    assert prepared.columns == ("id", "name", "note")
    assert prepared.canonical == CANONICAL.encode()  # quoted comma, embedded newline, empty cell
    assert prepared.sha256 == prepared.original_sha256
    assert not prepared.reencoded
    assert prepared.original_encoding == "utf-8"
    assert prepared.original_delimiter == ","
    assert prepared.size_bytes == len(CANONICAL.encode())


def test_cell_text_is_not_trimmed_or_typed() -> None:
    prepared = prepare_csv(b"a,b\n 007 ,1.50\n")
    assert prepared.canonical == b"a,b\n 007 ,1.50\n"


def _variants() -> dict[str, bytes]:
    plain = "id,name\n1,Ana\n2,Bo\n"
    return {
        "utf8": plain.encode(),
        "utf8_bom": codecs.BOM_UTF8 + plain.encode(),
        "utf16_le": codecs.BOM_UTF16_LE + plain.encode("utf-16-le"),
        "utf16_be": codecs.BOM_UTF16_BE + plain.encode("utf-16-be"),
        "semicolon": plain.replace(",", ";").encode(),
        "tab": plain.replace(",", "\t").encode(),
        "pipe": plain.replace(",", "|").encode(),
        "crlf": plain.replace("\n", "\r\n").encode(),
        "cr_only": plain.replace("\n", "\r").encode(),
        "no_final_newline": plain.rstrip("\n").encode(),
        "trailing_blank_lines": (plain + "\n\n").encode(),
    }


@pytest.mark.parametrize("label", sorted(_variants()))
def test_same_data_in_different_packaging_gives_the_same_canonical_file_and_hash(
    label: str,
) -> None:
    reference = prepare_csv(_variants()["utf8"])
    prepared = prepare_csv(_variants()[label])
    assert prepared.canonical == reference.canonical == b"id,name\n1,Ana\n2,Bo\n"
    assert prepared.sha256 == reference.sha256
    assert prepared.rows == 2 and prepared.columns == ("id", "name")


def test_original_hash_and_format_details_are_recorded_as_provenance() -> None:
    raw = codecs.BOM_UTF16_LE + "id;name\n1;Ana\n".encode("utf-16-le")
    prepared = prepare_csv(raw)
    assert prepared.original_sha256 == sha256_bytes(raw)
    assert prepared.original_sha256 != prepared.sha256
    assert prepared.original_encoding == "utf-16"
    assert prepared.original_delimiter == ";"
    assert prepared.original_size_bytes == len(raw)
    assert prepared.reencoded


def test_trailing_blank_lines_are_ignored_and_counted() -> None:
    prepared = prepare_csv(b"a,b\n1,2\n\n\n")
    assert prepared.rows == 1
    assert prepared.trailing_blank_lines_ignored == 2


def test_preparation_is_deterministic() -> None:
    assert prepare_csv(CANONICAL.encode()) == prepare_csv(CANONICAL.encode())


# --- rejected input ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "code"),
    [
        (b"", "empty_file"),
        (b"\n\n", "empty_file"),
        (b"a,b\n", "no_data_rows"),
        (b"a,b\n\n", "no_data_rows"),
        (b"a\n1\n2\n", "bad_delimiter"),  # a single column
        (b"a,,c\n1,2,3\n", "bad_header"),  # empty column name
        (b"a, ,c\n1,2,3\n", "bad_header"),  # blank-looking column name
        (b"a,b,a\n1,2,3\n", "bad_header"),  # duplicate name
        (b"Age,b, age \n1,2,3\n", "bad_header"),  # duplicate ignoring case and spaces
        (b"a,b,c\n1,2,3\n4,5\n", "ragged_row"),  # too few fields
        (b"a,b,c\n1,2,3\n4,5,6,7\n", "ragged_row"),  # too many fields
        (b"a,b\n1,2\n\n3,4\n", "blank_line"),  # blank line in the middle
        (b"\na,b\n1,2\n", "blank_line"),  # blank line before the header
        (b"a,b\n1,\x00\n", "nul_byte"),
        (b"a,b\n1,caf\xe9\n", "bad_encoding"),  # Latin-1 byte, not valid UTF-8
        (codecs.BOM_UTF32_LE + "a,b\n1,2\n".encode("utf-32-le"), "bad_encoding"),
        (b'a,b\n1,"unterminated\n', "csv_error"),  # strict quoting
        (b'a,b\n1,"x"y\n', "csv_error"),  # stray text after a closing quote
    ],
)
def test_bad_files_are_rejected_with_a_stable_code(data: bytes, code: str) -> None:
    assert codes(data) == code


def test_ambiguous_delimiter_is_rejected_not_guessed() -> None:
    assert codes(b"a,b;c\n1,2;3\n4,5;6\n") == "bad_delimiter"


def test_ragged_row_message_names_the_line() -> None:
    with pytest.raises(IngestError, match="line 3.*2 fields.*header has 3"):
        prepare_csv(b"a,b,c\n1,2,3\n4,5\n")


def test_oversized_single_cell_is_rejected() -> None:
    assert codes(b"a,b\n" + b"x" * 200_000 + b",1\n") == "csv_error"


# --- limits (small injected limits; the defaults are checked once below) -------------------


def test_row_limit_boundary() -> None:
    limits = IngestLimits(max_rows=3)
    assert prepare_csv(b"a,b\n1,1\n2,2\n3,3\n", limits).rows == 3
    assert codes(b"a,b\n1,1\n2,2\n3,3\n4,4\n", limits) == "too_many_rows"


def test_column_limit_boundary() -> None:
    limits = IngestLimits(max_columns=3)
    assert prepare_csv(b"a,b,c\n1,2,3\n", limits).columns == ("a", "b", "c")
    assert codes(b"a,b,c,d\n1,2,3,4\n", limits) == "too_many_columns"


def test_byte_limit_boundary() -> None:
    data = b"a,b\n1,2\n"
    assert prepare_csv(data, IngestLimits(max_bytes=len(data))).rows == 1
    assert codes(data, IngestLimits(max_bytes=len(data) - 1)) == "too_large"


def test_default_limits_are_the_approved_provisional_values() -> None:
    limits = IngestLimits()
    assert (limits.max_bytes, limits.max_rows, limits.max_columns) == (
        50 * 1024 * 1024,
        100_000,
        100,
    )


def test_default_byte_limit_rejects_before_any_parsing() -> None:
    too_big = b"x" * (IngestLimits().max_bytes + 1)  # not even CSV: must fail on size first
    assert codes(too_big) == "too_large"


def test_read_limited_refuses_big_files_without_reading_them(tmp_path: Path) -> None:
    path = tmp_path / "big.csv"
    path.write_bytes(b"a,b\n" + b"1,2\n" * 100)
    limits = IngestLimits(max_bytes=50)
    with pytest.raises(IngestError) as info:
        read_limited(path, limits)
    assert info.value.code == "too_large"
    assert read_limited(path, IngestLimits(max_bytes=10_000)) == path.read_bytes()

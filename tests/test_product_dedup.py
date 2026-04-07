"""Tests for product-level deduplication in Copernicus downloads.

Verifies that:
- find_product_on_disk correctly detects already-downloaded products by UUID
- process_products skips downloads when a product is already on disk
- Different queries returning the same product ID share the download
- Edge cases: empty files, missing dirs, no product ID, corrupted zips
"""

import zipfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

from src.data.copernicus.common import find_product_on_disk, process_products

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PRODUCT_UUID_A = "a8dd0899-7a3b-4e4b-9b3a-5e7f1234abcd"
PRODUCT_UUID_B = "b7cc1234-5d6e-4f7a-8b9c-0d1e2f3a4b5c"
PRODUCT_UUID_C = "c6bb5678-9a0b-1c2d-3e4f-567890abcdef"


def _make_product(product_id: str, name: str, size: int = 1000) -> Dict[str, Any]:
    """Create a fake Copernicus product dict matching the API shape."""
    return {
        "Id": product_id,
        "Name": name,
        "ContentLength": size,
        "ContentDate": {"Start": "2022-01-01T00:00:00.000Z"},
    }


def _create_fake_file(directory: Path, filename: str, size: int = 100) -> Path:
    """Create a fake file with some content.

    For .zip files, creates a valid zip archive so it passes integrity checks.
    For other files, writes raw bytes.
    """
    path = directory / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if filename.endswith(".zip"):
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("placeholder.txt", "x" * size)
    else:
        path.write_bytes(b"x" * size)
    return path


def _make_mock_client(cache_dir: Path) -> MagicMock:
    """Create a mock CopernicusClient with the given cache_dir."""
    client = MagicMock()
    client.cache_dir = cache_dir
    return client


# ---------------------------------------------------------------------------
# find_product_on_disk
# ---------------------------------------------------------------------------


def test_finds_existing_zip(tmp_path: Path) -> None:
    """Product zip with embedded UUID is found."""
    _create_fake_file(tmp_path / "s2", f"{PRODUCT_UUID_A}__S2A_MSIL1C_20220101_R10m.zip")
    result = find_product_on_disk(tmp_path, "s2", PRODUCT_UUID_A)
    assert result is not None
    assert PRODUCT_UUID_A in result.name


def test_finds_existing_metadata_json(tmp_path: Path) -> None:
    """Metadata json with embedded UUID is found."""
    _create_fake_file(tmp_path / "s1", f"{PRODUCT_UUID_B}__S1A_IW_GRDH_metadata.json")
    result = find_product_on_disk(tmp_path, "s1", PRODUCT_UUID_B)
    assert result is not None


def test_returns_none_when_not_present(tmp_path: Path) -> None:
    """Returns None when no file matches the product ID."""
    (tmp_path / "s2").mkdir(parents=True)
    _create_fake_file(tmp_path / "s2", f"{PRODUCT_UUID_A}__some_product.zip")
    assert find_product_on_disk(tmp_path, "s2", PRODUCT_UUID_B) is None


def test_returns_none_when_subdir_missing(tmp_path: Path) -> None:
    """Returns None when the satellite subdirectory doesn't exist."""
    assert find_product_on_disk(tmp_path, "s2", PRODUCT_UUID_A) is None


def test_ignores_empty_files(tmp_path: Path) -> None:
    """Empty files (0 bytes) are not considered valid downloads."""
    path = tmp_path / "s2" / f"{PRODUCT_UUID_A}__S2A_product.zip"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")
    assert find_product_on_disk(tmp_path, "s2", PRODUCT_UUID_A) is None


def test_detects_corrupted_zip(tmp_path: Path) -> None:
    """A truncated/corrupted zip is rejected and deleted."""
    path = tmp_path / "s2" / f"{PRODUCT_UUID_A}__S2A_product.zip"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"PK\x03\x04" + b"\x00" * 100)
    assert find_product_on_disk(tmp_path, "s2", PRODUCT_UUID_A) is None
    assert not path.exists()


def test_accepts_valid_zip(tmp_path: Path) -> None:
    """A valid zip file is accepted."""
    path = tmp_path / "s2" / f"{PRODUCT_UUID_A}__S2A_product.zip"
    path.parent.mkdir(parents=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("test.txt", "hello")
    assert find_product_on_disk(tmp_path, "s2", PRODUCT_UUID_A) is not None


def test_non_zip_files_skip_zip_check(tmp_path: Path) -> None:
    """Non-zip files (like metadata json) only need size > 0 check."""
    path = tmp_path / "s1" / f"{PRODUCT_UUID_A}__S1A_metadata.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"product_id": "test"}')
    assert find_product_on_disk(tmp_path, "s1", PRODUCT_UUID_A) is not None


def test_does_not_match_partial_uuid(tmp_path: Path) -> None:
    """A file whose name starts with a prefix of the UUID should not match."""
    _create_fake_file(tmp_path / "s2", "a8dd0899__truncated.zip")
    assert find_product_on_disk(tmp_path, "s2", PRODUCT_UUID_A) is None


def test_correct_subdir_isolation(tmp_path: Path) -> None:
    """A product in s1/ is not found when searching s2/."""
    _create_fake_file(tmp_path / "s1", f"{PRODUCT_UUID_A}__S1A_product.zip")
    assert find_product_on_disk(tmp_path, "s2", PRODUCT_UUID_A) is None
    assert find_product_on_disk(tmp_path, "s1", PRODUCT_UUID_A) is not None


# ---------------------------------------------------------------------------
# process_products — dedup integration
# ---------------------------------------------------------------------------


def test_skips_download_when_product_on_disk(tmp_path: Path) -> None:
    """download_func should NOT be called for a product already on disk."""
    client = _make_mock_client(tmp_path)
    existing_file = _create_fake_file(tmp_path / "s2", f"{PRODUCT_UUID_A}__S2A_MSIL1C_R10m.zip")

    download_func = MagicMock(return_value=None)
    paths = process_products(
        client=client,
        products=[_make_product(PRODUCT_UUID_A, "S2A_MSIL1C_20220101")],
        download_data=True,
        satellite="SENTINEL-2",
        download_func=download_func,
        metadata_func=MagicMock(),
    )

    download_func.assert_not_called()
    assert paths == [existing_file]


def test_downloads_when_product_not_on_disk(tmp_path: Path) -> None:
    """download_func IS called when the product is not on disk."""
    (tmp_path / "s2").mkdir(parents=True)
    client = _make_mock_client(tmp_path)

    new_file = tmp_path / "s2" / f"{PRODUCT_UUID_A}__new.zip"
    download_func = MagicMock(return_value=new_file)

    paths = process_products(
        client=client,
        products=[_make_product(PRODUCT_UUID_A, "S2A_MSIL1C_20220101")],
        download_data=True,
        satellite="SENTINEL-2",
        download_func=download_func,
        metadata_func=MagicMock(),
    )

    download_func.assert_called_once()
    assert len(paths) == 1


def test_mixed_skip_and_download(tmp_path: Path) -> None:
    """With 3 products, 1 already on disk and 2 new, only 2 downloads happen."""
    client = _make_mock_client(tmp_path)
    existing = _create_fake_file(tmp_path / "s2", f"{PRODUCT_UUID_A}__S2A_existing.zip")

    products = [
        _make_product(PRODUCT_UUID_A, "S2A_existing"),
        _make_product(PRODUCT_UUID_B, "S2A_new_b"),
        _make_product(PRODUCT_UUID_C, "S2A_new_c"),
    ]

    call_count = 0

    def fake_download(client, product, index, **kwargs):
        nonlocal call_count
        call_count += 1
        pid = product["Id"]
        path = tmp_path / "s2" / f"{pid}__downloaded.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("data.txt", "satellite-data")
        return path

    paths = process_products(
        client=client,
        products=products,
        download_data=True,
        satellite="SENTINEL-2",
        download_func=fake_download,
        metadata_func=MagicMock(),
    )

    assert call_count == 2  # Only B and C downloaded
    assert len(paths) == 3  # All 3 returned
    assert paths[0] == existing  # A was the cached one


def test_skips_metadata_when_product_on_disk(tmp_path: Path) -> None:
    """metadata_func should NOT be called for a product already on disk."""
    client = _make_mock_client(tmp_path)
    existing = _create_fake_file(tmp_path / "s1", f"{PRODUCT_UUID_A}__S1A_metadata.json")

    metadata_func = MagicMock(return_value=None)
    paths = process_products(
        client=client,
        products=[_make_product(PRODUCT_UUID_A, "S1A_IW_GRDH")],
        download_data=False,
        satellite="SENTINEL-1",
        download_func=MagicMock(),
        metadata_func=metadata_func,
    )

    metadata_func.assert_not_called()
    assert paths == [existing]


def test_no_dedup_when_product_has_no_id(tmp_path: Path) -> None:
    """Products without an Id field should always go through download."""
    (tmp_path / "s2").mkdir(parents=True)
    client = _make_mock_client(tmp_path)

    dummy_path = tmp_path / "s2" / "downloaded.zip"
    dummy_path.write_bytes(b"data")
    download_func = MagicMock(return_value=dummy_path)

    paths = process_products(
        client=client,
        products=[{"Name": "S2A_unknown", "ContentLength": 100}],
        download_data=True,
        satellite="SENTINEL-2",
        download_func=download_func,
        metadata_func=MagicMock(),
    )

    download_func.assert_called_once()
    assert len(paths) == 1


# ---------------------------------------------------------------------------
# Scenario: two different bboxes, same tile
# ---------------------------------------------------------------------------


def test_second_query_skips_download(tmp_path: Path) -> None:
    """
    Query 1 (bbox_a) downloads product X.
    Query 2 (bbox_b, slightly shifted) returns the same product X.
    Product X should NOT be downloaded again.
    """
    client = _make_mock_client(tmp_path)
    product_x = _make_product(PRODUCT_UUID_A, "S2A_MSIL1C_20220101_T31UGQ")
    download_calls: List[str] = []

    def fake_download(client, product, index, **kwargs):
        pid = product["Id"]
        download_calls.append(pid)
        path = tmp_path / "s2" / f"{pid}__S2A_MSIL1C_20220101_T31UGQ_R10m.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("data.txt", "satellite-data")
        return path

    # Query 1: first time, product gets downloaded
    paths_1 = process_products(
        client=client,
        products=[product_x],
        download_data=True,
        satellite="SENTINEL-2",
        download_func=fake_download,
        metadata_func=MagicMock(),
    )
    assert len(download_calls) == 1

    # Query 2: same product returned, should be skipped
    paths_2 = process_products(
        client=client,
        products=[product_x],
        download_data=True,
        satellite="SENTINEL-2",
        download_func=fake_download,
        metadata_func=MagicMock(),
    )
    assert len(download_calls) == 1  # NOT called again
    assert paths_1[0] == paths_2[0]  # Same file returned

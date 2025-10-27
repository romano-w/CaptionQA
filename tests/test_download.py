import hashlib
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest
import responses

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from captionqa.data import download
from captionqa.data.download import DownloadArtifact


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(download.time, "sleep", lambda _: None)


def test_http_download_dry_run(tmp_path, capsys):
    destination = tmp_path / "sample.bin"
    download._download_http_file(
        "https://example.com/sample.bin", destination, overwrite=False, dry_run=True
    )

    assert not destination.exists()
    captured = capsys.readouterr()
    assert "[dry-run]" in captured.out


def test_http_download_overwrite_protection(tmp_path):
    destination = tmp_path / "sample.bin"
    destination.write_text("existing")

    with pytest.raises(FileExistsError):
        download._download_http_file(
            "https://example.com/sample.bin", destination, overwrite=False, dry_run=False
        )


@responses.activate
def test_http_download_retries_and_succeeds(tmp_path):
    url = "https://example.com/retry.bin"
    destination = tmp_path / "retry.bin"

    responses.add(responses.GET, url, status=500)
    responses.add(
        responses.GET,
        url,
        body=b"ok",
        status=200,
        headers={"Content-Length": "2"},
    )

    download._download_http_file(url, destination, overwrite=True, dry_run=False)

    assert destination.read_bytes() == b"ok"
    assert len(responses.calls) == 2


def test_download_artifacts_checksum_success(tmp_path, httpserver):
    data = b"checksum"
    httpserver.expect_request("/file").respond_with_data(data)
    url = httpserver.url_for("/file")

    artifact = DownloadArtifact(
        url=url,
        checksum=hashlib.sha256(data).hexdigest(),
    )

    download._download_artifacts([artifact], tmp_path, overwrite=True, dry_run=False)

    downloaded = tmp_path / Path(urlparse(url).path).name
    assert downloaded.exists()
    assert downloaded.read_bytes() == data


@responses.activate
def test_download_artifacts_checksum_failure(tmp_path):
    url = "https://example.com/bad.bin"
    destination = tmp_path / "bad.bin"
    data = b"bad"

    responses.add(
        responses.GET,
        url,
        body=data,
        status=200,
        headers={"Content-Length": str(len(data))},
    )

    artifact = DownloadArtifact(
        url=url,
        checksum="sha256:deadbeef",
    )

    with pytest.raises(ValueError) as excinfo:
        download._download_artifacts([artifact], tmp_path, overwrite=True, dry_run=False)

    assert "Checksum mismatch" in str(excinfo.value)
    assert not destination.exists()

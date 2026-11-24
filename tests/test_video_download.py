from pathlib import Path
import hashlib
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import download_video

class MockResponse:
    def __init__(self):
        self.body=b'00000018ftypmp42'+b'examplemocktest'
        self.offset=0
        self.headers={'Content-Type':'video/mp4'}
    def read(self,n):
        part=self.body[self.offset:self.offset+n];self.offset+=len(part);return part
    def __enter__(self):return self
    def __exit__(self,*a):pass

def test_download_streaming_and_hash(tmp_path,monkeypatch):
    monkeypatch.setattr(download_video.urllib.request,'urlopen',lambda *a,**kw:MockResponse())
    p=tmp_path/'clip.mp4'
    download_video.run('https://media.example.org/test.mp4',p,1)
    assert p.exists() and p.with_suffix('.download.json').exists()
    assert b'examplemocktest' in p.read_bytes()

def test_insecure_or_unsupported_url_rejected(tmp_path):
    with pytest.raises(ValueError):
        download_video.run('http://example.org/foo.mp4',tmp_path/'bad.mp4')

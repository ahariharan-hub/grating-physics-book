"""Build the two self-contained Jupyter Book downloads from current sources.

Run this file before building/publishing the book. Archives contain their own
wave_optics package and exclude environments, caches, tests and notebook outputs.
Only the standard library is needed to build the archives.
Text entries use LF line endings so Windows and Linux produce the same bundle.
"""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
COLLECTIONS=ROOT/'Wave_solver_py_scripts'
OUTPUT=ROOT/'downloads'


def archive_bytes(path):
    return path.read_bytes().replace(b'\r\n', b'\n')


def build(collection):
    folder=COLLECTIONS/collection
    files=sorted(p for p in folder.rglob('*')
                 if p.is_file() and p.suffix in {'.py','.txt','.md','.json'}
                 and '__pycache__' not in p.parts
                 and not any(part.lower() in {'tests','validation','supplement'} for part in p.relative_to(folder).parts)
                 and not p.name.lower().startswith('test_'))
    backend=sorted(p for p in (ROOT/'wave_optics').iterdir()
                   if p.is_file() and p.suffix == '.py' and not p.name.lower().startswith('test_'))
    entries={f'Grating_Physics/{p.relative_to(ROOT).as_posix()}':archive_bytes(p) for p in files+backend}
    entries['Grating_Physics/requirements.txt']=archive_bytes(folder/'requirements.txt')
    entries['Grating_Physics/README.txt']=archive_bytes(folder/'README.txt')
    archive=OUTPUT/f'grating_physics_{collection}.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as bundle:
        for name,data in sorted(entries.items()):
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o644 << 16
            bundle.writestr(info,data)
    return dict(file=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                bytes=archive.stat().st_size,entries=len(entries),
                entrypoints=[p.relative_to(folder).as_posix() for p in files
                             if p.suffix=='.py' and not p.name.startswith('_')])


def build_all():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    records=[build(name) for name in ('numerical_explorer','scripts')]
    (OUTPUT/'manifest.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    for record in records: print(record['file'],record['bytes'],'bytes;',len(record['entrypoints']),'numbered programs')
    return records


if __name__=='__main__':
    build_all()

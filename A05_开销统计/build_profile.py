from pathlib import Path
import os, subprocess, shutil, json, hashlib
SRC = Path(__file__).resolve().parent / '代码'
compiler = Path('C:/msys64/mingw64/bin/g++.exe')
cmake = compiler.with_name('cmake.exe')
env = os.environ.copy()
env['PATH'] = str(compiler.parent) + os.pathsep + env.get('PATH', '')
build = SRC / 'build'
build.mkdir(exist_ok=True)
with (build / 'configure.log').open('w', encoding='utf-8') as log:
    r = subprocess.run([str(cmake), '-S', str(SRC), '-B', str(build), '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_COMPILER=' + str(compiler), '-DCMAKE_MAKE_PROGRAM=' + str(compiler.with_name('ninja.exe'))], env=env, stdout=log, stderr=subprocess.STDOUT)
if r.returncode:
    raise RuntimeError((build / 'configure.log').read_text(encoding='utf-8', errors='replace'))
with (build / 'build.log').open('w', encoding='utf-8') as log:
    r = subprocess.run([str(cmake), '--build', str(build), '--parallel', '2'], env=env, stdout=log, stderr=subprocess.STDOUT)
if r.returncode:
    raise RuntimeError((build / 'build.log').read_text(encoding='utf-8', errors='replace'))
for name in ['libgcc_s_seh-1.dll', 'libstdc++-6.dll', 'libwinpthread-1.dll']:
    shutil.copy2(compiler.with_name(name), build / name)
manifest = {'scope': 'Frozen 20261104 original benchmark with measurement-only instrumentation; see ../instrumentation.diff.', 'files': {str(p.relative_to(SRC)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in SRC.rglob('*') if p.is_file() and ('generated' in p.parts or p.suffix in ['.cpp', '.py'])}}
(SRC / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('BUILD_OK')
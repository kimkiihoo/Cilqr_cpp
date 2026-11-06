from pathlib import Path
import sys, subprocess, os, json, datetime, argparse
OUT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--pilot', action='store_true')
parser.add_argument('--output', type=Path)
args = parser.parse_args()
pilot = args.pilot
name = '预运行核验_B201' if pilot else 'B201_正式统计'
destination = args.output.resolve() if args.output else OUT / name
env = os.environ.copy()
env['PYTHONUTF8'] = '1'
cmd = [sys.executable, str(OUT / '代码/run_maps.py'), '--mode', 'pilot' if pilot else 'full', '--maps', 'B201', '--methods', 'r', 'al', 'sqp', '--pilot-cycles', '3', '--output', str(destination)]
if destination.exists():
    raise FileExistsError('Refusing to overwrite ' + str(destination))
with (destination.parent / (destination.name + '_运行日志.txt')).open('x', encoding='utf-8') as log:
    result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
print('EXIT', result.returncode, 'OUTPUT', destination, flush=True)
raise SystemExit(result.returncode)
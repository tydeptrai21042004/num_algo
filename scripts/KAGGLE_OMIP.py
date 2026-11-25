"""Copy into a Kaggle cell or run as Python. Fresh checkout per commit; no old checkpoints."""
import os,sys,subprocess,tarfile,shutil
from pathlib import Path
WORK=Path('/kaggle/working');WORK.mkdir(exist_ok=True,parents=True)
URL='https://github.com/tydeptrai21042004/num_algo.git'
os.environ.update({k:'1' for k in ['FCAOIP_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']})
commit=subprocess.check_output(['git','ls-remote',URL,'refs/heads/main'],text=True).split()[0]
repo=WORK/f'num_algo_omip_{commit[:12]}'
def run(*cmd):subprocess.run([str(x) for x in cmd],cwd=repo,check=True)
if not repo.exists():subprocess.run(['git','clone','--depth','1','--branch','main',URL,str(repo)],check=True)
if subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()!=commit:raise RuntimeError('Git commit mismatch')
if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip():raise RuntimeError('Tracked source files changed; use a fresh checkout')
if not (repo/'scripts/audit_attacks.py').exists():raise RuntimeError('Upload OMIP patch before running')
if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):subprocess.run(['apt-get','update','-qq'],check=True);subprocess.run(['apt-get','install','-y','ffmpeg'],check=True)
run(sys.executable,'-m','pip','install','-r','requirements.txt');run(sys.executable,'scripts/verify_frozen_core.py');run(sys.executable,'-m','pytest','-q')
root=repo/'results/omip_final60_cleanvideo';root.mkdir(parents=True,exist_ok=True);(root/'git_commit.txt').write_text(commit+'\n')
try:
    for stage in ['attack_audit','mathematics','calibration','hyperparameters','test','ablation','payload_robustness','security','video','compute','summary','audit']:
        run(sys.executable,'scripts/run_12_images.py','--stage',stage,'--keys','5','--video-frames','100','--resume')
finally:
    archive=WORK/f'OMIP_{commit[:12]}_results.tar.gz';tmp=archive.with_suffix('.tmp')
    with tarfile.open(tmp,'w:gz',compresslevel=1) as tar:tar.add(repo/'results',arcname='results')
    tmp.replace(archive);print('Results archive:',archive)

"""Additional exploratory label-remapping stress; see the dated separate protocol."""
from pathlib import Path
import csv,hashlib,json,sys,time,platform,os,argparse
import numpy as np
import run_neural as s

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=s.ROOT/'results'/'neural_mnist_stress_v1');args=parser.parse_args()
 root=s.ROOT;out=args.output
 if out.exists() and any(out.iterdir()):raise RuntimeError('Use a new empty output directory')
 out.mkdir(parents=True,exist_ok=True);(out/'checkpoints').mkdir()
 train,y=s.load_split('train');tx,ty=s.load_split('test');cfg=s.Config();cfg.novelty_threshold,calibration=s.calibrate(train,cfg)
 s.SCENARIOS=('remapped',)
 for src in [Path(__file__),Path(s.__file__)]: (out/src.name).write_bytes(src.read_bytes())
 manifest=dict(status='running',scenarios=['remapped'],policies=s.POLICIES,seeds=list(range(5)),config=s.asdict(cfg),calibration=calibration,source_sha256=hashlib.sha256(Path(s.__file__).read_bytes()).hexdigest(),wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((root/'05b_标签重映射压力测试.md').read_bytes()).hexdigest(),dataset_provenance=json.loads((root/'data'/'mnist'/'provenance.json').read_text()),python=sys.version,numpy=np.__version__,platform=platform.platform(),blas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),design_note='Exploratory stress added after the two-stream MNIST primary experiment started; no hyperparameter changes',label_maps=[0,3,6])
 s.json_write(out/'manifest.json',manifest);records=[];traces=[];matrices=[];begin=time.perf_counter()
 try:
  for seed in range(5):
   stream,perms,_,indices=s.build_stream(train,y,'permuted',seed,cfg);h=hashlib.sha256()
   for batch,priority,segment,domain,last in stream:
    batch['y']=(batch['y']+3*domain)%10
    for k in ('x','y','ids'):h.update(batch[k].tobytes())
   data_hash=h.hexdigest();np.savez_compressed(out/f'design_remapped_{seed}.npz',source_indices=indices,permutations=np.array(perms))
   tests=[(s.transform(tx,d,'permuted',perms),(ty+3*d)%10) for d in range(3)]
   for policy in s.POLICIES:
    result,trace,matrix=s.run_one(policy,'remapped',seed,cfg,stream,tests,data_hash,out/'checkpoints'/f'remapped_{seed}_{policy}.npz');records.append(result);traces.extend(trace);matrices.extend(matrix)
    s.csv_write(out/'per_run.csv',records);s.csv_write(out/'summary.csv',s.aggregate(records))
    print(f'remapped seed={seed} {policy}: accuracy={result["final_accuracy"]:.4f}, experts={result["experts"]}; elapsed={time.perf_counter()-begin:.1f}s',flush=True)
  s.csv_write(out/'window_log.csv',traces);s.csv_write(out/'accuracy_matrix.csv',matrices);s.csv_write(out/'paired_comparisons.csv',s.paired(records));manifest.update(status='completed',completed_runs=len(records),total_seconds=time.perf_counter()-begin)
 except Exception as e:manifest.update(status='failed',error=repr(e));raise
 finally:s.json_write(out/'manifest.json',manifest)

if __name__=='__main__':main()


"""MNIST-derived continual learning with two-hidden-layer ReLU experts.
NumPy float32 implementation; no domain IDs or boundaries reach the learner.
Run with OPENBLAS_NUM_THREADS=1 to limit small-matrix threading overhead.
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, json, os, platform, struct, sys, time, urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
RESOURCES={
 'train-images-idx3-ubyte.gz':'f68b3c2dcbeaaa9fbdd348bbdeb94873',
 'train-labels-idx1-ubyte.gz':'d53e105ee54ea40749a09fcbcd1e9432',
 't10k-images-idx3-ubyte.gz':'9fb629c4189551a2d022fa330f9573f3',
 't10k-labels-idx1-ubyte.gz':'ec29112dd5afa0611ce80d1b7f02629c'}
POLICIES=('fixed','novelty','interference','dual','dual_latched')
SCENARIOS=('contrast','permuted')

@dataclass
class Config:
 batch_size:int=100
 examples_per_segment:int=6000
 replay_size:int=512
 replay_batch:int=64
 reference_batch:int=64
 expert_cap:int=4
 prototype_cap:int=12
 persistence:int=3
 gallery_persistence:int=3
 cooldown:int=3
 updates:int=4
 warmup_windows:int=60
 learning_rate:float=.1
 replay_weight:float=.25
 l2:float=.0001
 interference_threshold:float=.2
 gradient_floor:float=1e-8
 evidence_ttl:int=30
 hidden1:int=128
 hidden2:int=64
 novelty_threshold:float=0.
 test_size:int=10000

def json_write(p,obj): p.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def csv_write(p,rows):
 if not rows:return
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def prepare_data():
 out=ROOT/'data'/'mnist';out.mkdir(parents=True,exist_ok=True);records=[]
 for filename,checksum in RESOURCES.items():
  path=out/filename;url='https://ossci-datasets.s3.amazonaws.com/mnist/'+filename
  if not path.exists():
   print('Downloading',filename,flush=True)
   with urllib.request.urlopen(url,timeout=60) as response: payload=response.read()
   if hashlib.md5(payload).hexdigest()!=checksum:raise ValueError('Download checksum mismatch '+filename)
   path.write_bytes(payload)
  payload=path.read_bytes()
  if hashlib.md5(payload).hexdigest()!=checksum:raise ValueError('Existing checksum mismatch '+filename)
  records.append({'file':filename,'url':url,'md5':checksum,'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)})
 json_write(out/'provenance.json',{'source_documentation':'https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py','records':records})
 return out

def load_split(split):
 out=ROOT/'data'/'mnist';prefix='train' if split=='train' else 't10k'
 with gzip.open(out/(prefix+'-images-idx3-ubyte.gz'),'rb') as f:
  magic,n,h,w=struct.unpack('>IIII',f.read(16));x=np.frombuffer(f.read(),dtype=np.uint8).reshape(n,h*w)
  assert magic==2051 and h==w==28 and n==(60000 if split=='train' else 10000)
 with gzip.open(out/(prefix+'-labels-idx1-ubyte.gz'),'rb') as f:
  magic,n=struct.unpack('>II',f.read(8));y=np.frombuffer(f.read(),dtype=np.uint8).astype(np.int64)
  assert magic==2049 and len(y)==n==len(x)
 return x.astype(np.float32)/np.float32(255),y

class Network:
 def __init__(self,seed=0,dims=(784,128,64,10),dtype=np.float32):
  rng=np.random.default_rng(seed);self.dims=tuple(dims)
  self.p=[]
  for a,b in zip(dims[:-1],dims[1:]):
   self.p.extend([(rng.normal(size=(a,b))*np.sqrt(2/a)).astype(dtype),np.zeros(b,dtype=dtype)])
 def clone(self):
  new=object.__new__(Network);new.dims=self.dims;new.p=[v.copy() for v in self.p];return new
 def forward(self,x):
  w1,b1,w2,b2,w3,b3=self.p
  h1=np.maximum(x@w1+b1,0);h2=np.maximum(h1@w2+b2,0)
  return h2@w3+b3
 def loss_grad(self,x,y):
  w1,b1,w2,b2,w3,b3=self.p
  h1=np.maximum(x@w1+b1,0);h2=np.maximum(h1@w2+b2,0);z=h2@w3+b3
  z=z-z.max(axis=1,keepdims=True);exp=np.exp(z);prob=exp/exp.sum(axis=1,keepdims=True)
  loss=float(np.mean(np.log(exp.sum(axis=1))-z[np.arange(len(y)),y]))
  dz=prob;dz[np.arange(len(y)),y]-=1;dz/=len(y)
  gw3=h2.T@dz;gb3=dz.sum(axis=0);dh2=(dz@w3.T)*(h2>0)
  gw2=h1.T@dh2;gb2=dh2.sum(axis=0);dh1=(dh2@w2.T)*(h1>0)
  gw1=x.T@dh1;gb1=dh1.sum(axis=0)
  return loss,[gw1,gb1,gw2,gb2,gw3,gb3]
 def nbytes(self):return sum(p.nbytes for p in self.p)

def alignment(a,b,floor):
 aa=sum(float(np.sum(x*x,dtype=np.float64)) for x in a);bb=sum(float(np.sum(x*x,dtype=np.float64)) for x in b)
 if aa<floor*floor or bb<floor*floor:return 0.,False
 ab=sum(float(np.sum(x*y,dtype=np.float64)) for x,y in zip(a,b))
 return float(np.clip(-ab/np.sqrt(aa*bb),0,1)),True

class Memory:
 def __init__(self,cap):
  self.cap=cap;self.x=np.empty((0,784),dtype=np.float32);self.y=np.empty(0,dtype=np.int64);self.ids=np.empty(0,dtype=np.int64);self.priority=np.empty(0,dtype=np.float64)
 def add(self,batch,priorities):
  if np.any(batch['ids']<0):raise AssertionError('Test data cannot enter memory')
  for k in ('x','y','ids'):setattr(self,k,np.concatenate([getattr(self,k),batch[k]]))
  self.priority=np.r_[self.priority,priorities]
  chosen=np.argsort(self.priority,kind='stable')[-self.cap:]
  for k in ('x','y','ids','priority'):setattr(self,k,getattr(self,k)[chosen])
  assert len(self.ids)<=self.cap
 def nbytes(self):return sum(getattr(self,k).nbytes for k in ('x','y','ids','priority'))

class Learner:
 def __init__(self,policy,cfg,seed):
  self.policy,self.cfg=policy,cfg;self.experts=[Network(seed+40000,(784,cfg.hidden1,cfg.hidden2,10))]
  self.anchors=[];self.anchor_experts=[];self.expiry=[];self.pending=[]
  self.memory=Memory(cfg.replay_size);self.rng=np.random.default_rng(seed+50000)
  self.streak=0;self.previous_candidate=None;self.last_expansion=-100000
  self.expansion_events=0;self.blocked_events=0;self.gradient_calls=0;self.gradient_examples=0
 def route(self,x):
  if not self.anchors:return np.zeros(len(x),dtype=np.int64)
  a=np.array(self.anchors);dist=(x*x).sum(axis=1)[:,None]+(a*a).sum(axis=1)[None,:]-2*(x@a.T)
  return np.asarray(self.anchor_experts,dtype=np.int64)[np.argmin(dist,axis=1)]
 def gradient(self,e,x,y):
  self.gradient_calls+=1;self.gradient_examples+=len(y)
  return self.experts[e].loss_grad(x,y)[1]
 def train_window(self,batch,priority,t):
  cfg=self.cfg;x,y,ids=(batch[k] for k in ('x','y','ids'))
  if np.any(ids<0):raise AssertionError('Evaluation samples cannot train')
  if len(self.memory.ids):assert self.memory.ids.max()<ids.min(),'Future sample leakage'
  mean=x.mean(axis=0)
  if not self.anchors:self.anchors.append(mean.copy());self.anchor_experts.append(0);self.expiry.append(-1)
  distances=np.linalg.norm(np.array(self.anchors)-mean,axis=1);nearest=int(np.argmin(distances));novelty=float(distances[nearest]);candidate=int(self.anchor_experts[nearest]);original_candidate=candidate
  eligible=np.flatnonzero(self.route(self.memory.x)==candidate)
  interference=None;reliable=False
  if self.policy in ('interference','dual','dual_latched') and len(eligible):
   ref=eligible[:cfg.reference_batch]
   interference,reliable=alignment(self.gradient(candidate,x,y),self.gradient(candidate,self.memory.x[ref],self.memory.y[ref]),cfg.gradient_floor)
  is_novel=novelty>cfg.novelty_threshold
  retained=self.policy=='dual_latched' and t<=self.expiry[nearest]
  is_conflict=interference is not None and interference>cfg.interference_threshold
  condition={'fixed':False,'novelty':is_novel,'interference':is_conflict,'dual':is_novel and is_conflict,'dual_latched':(is_novel or retained) and is_conflict}[self.policy]
  if t<cfg.warmup_windows:condition=False
  if candidate!=self.previous_candidate:self.streak=0
  self.previous_candidate=candidate;self.streak=self.streak+1 if condition else 0
  if is_novel:self.pending=(self.pending+[mean.copy()])[-cfg.gallery_persistence:]
  else:self.pending=[]
  expanded=blocked=admitted=False
  if self.streak>=cfg.persistence and t-self.last_expansion>=cfg.cooldown:
   if len(self.experts)<cfg.expert_cap and (not is_novel or len(self.anchors)<cfg.prototype_cap):
    self.experts.append(self.experts[candidate].clone());candidate=len(self.experts)-1;expanded=True;self.expansion_events+=1;self.last_expansion=t
   else:blocked=True;self.blocked_events+=1
   self.streak=0
  if expanded and is_novel:
   self.anchors.append(np.mean(self.pending,axis=0));self.anchor_experts.append(candidate);self.expiry.append(-1);self.pending=[];admitted=True
  elif is_novel and len(self.pending)>=cfg.gallery_persistence:
   if len(self.anchors)<cfg.prototype_cap:
    self.anchors.append(np.mean(self.pending,axis=0));self.anchor_experts.append(candidate);self.expiry.append(t+cfg.evidence_ttl if self.policy=='dual_latched' else -1);admitted=True
   self.pending=[]
  elif expanded:
   self.anchor_experts[nearest]=candidate;self.expiry[nearest]=-1
  elif not is_novel:self.anchors[nearest]=np.float32(.95)*self.anchors[nearest]+np.float32(.05)*mean
  eligible=np.flatnonzero(self.route(self.memory.x)==candidate)
  for _ in range(cfg.updates):
   grad=self.gradient(candidate,x,y)
   if len(eligible):
    chosen=self.rng.choice(eligible,cfg.replay_batch,replace=len(eligible)<cfg.replay_batch)
    old=self.gradient(candidate,self.memory.x[chosen],self.memory.y[chosen]);grad=[np.float32(1-cfg.replay_weight)*g+np.float32(cfg.replay_weight)*h for g,h in zip(grad,old)]
   for i,(p,g) in enumerate(zip(self.experts[candidate].p,grad)):
    if i%2==0:g=g+np.float32(cfg.l2)*p
    p-=np.float32(cfg.learning_rate)*g
  self.memory.add(batch,priority)
  assert len(self.experts)<=cfg.expert_cap and len(self.anchors)<=cfg.prototype_cap
  return dict(window=t,novelty=novelty,interference=interference,gradient_reliable=reliable,raw_novel=is_novel,retained_evidence=bool(retained),condition=bool(condition),expanded=expanded,blocked=blocked,admitted=admitted,experts=len(self.experts),prototypes=len(self.anchors),candidate=original_candidate,train_expert=candidate,replay_count=len(self.memory.ids))
 def evaluate(self,x,y):
  routes=self.route(x);correct=0
  for e,net in enumerate(self.experts):
   idx=np.flatnonzero(routes==e)
   for start in range(0,len(idx),512):
    part=idx[start:start+512];correct+=int(np.sum(np.argmax(net.forward(x[part]),axis=1)==y[part]))
  return correct/len(y)
 def array_bytes(self):
  return sum(n.nbytes() for n in self.experts)+self.memory.nbytes()+np.asarray(self.anchors,dtype=np.float32).nbytes+np.asarray(self.anchor_experts,dtype=np.int64).nbytes+np.asarray(self.expiry,dtype=np.int64).nbytes+np.asarray(self.pending,dtype=np.float32).nbytes

def calibrate(train,cfg):
 rng=np.random.default_rng(20261024);pool=train[:2000];distances=[]
 for _ in range(512):
  means=pool[rng.integers(0,len(pool),size=(2,cfg.batch_size))].mean(axis=1)
  distances.append(np.linalg.norm(means[0]-means[1]))
 return float(np.quantile(distances,.995)),dict(seed=20261024,train_indices=[0,1999],labels_used=False,paired_means=512,quantile=.995)

def permutations(seed):
 rng=np.random.default_rng(seed+60000)
 return [np.arange(784),rng.permutation(784),rng.permutation(784)]

def transform(x,domain,scenario,perms):
 if scenario=='contrast':return [lambda z:z,lambda z:np.float32(.55)*z,lambda z:np.float32(.55)*z+np.float32(.25)][domain](x)
 return np.ascontiguousarray(x[:,perms[domain]])

def build_stream(train,y,scenario,seed,cfg,development=False):
 rng=np.random.default_rng(seed+10000);order=[0,1,2,0,1,2];n=cfg.examples_per_segment
 pool=np.arange(0,8000) if development else np.arange(10000,60000)
 assert len(order)*n<=len(pool),'Require disjoint source images within a run'
 selected=rng.permutation(pool)[:len(order)*n];perms=permutations(seed);stream=[]
 priority_rng=np.random.default_rng(seed+30000);h=hashlib.sha256()
 for segment,domain in enumerate(order):
  inds=selected[segment*n:(segment+1)*n];data=transform(train[inds],domain,scenario,perms)
  for start in range(0,n,cfg.batch_size):
   global_start=segment*n+start
   b={'x':data[start:start+cfg.batch_size],'y':y[inds[start:start+cfg.batch_size]],'ids':np.arange(global_start,global_start+cfg.batch_size,dtype=np.int64)}
   for k in ('x','y','ids'):h.update(b[k].tobytes())
   stream.append((b,priority_rng.random(cfg.batch_size),segment,domain,start+cfg.batch_size==n))
 return stream,perms,h.hexdigest(),selected

def run_one(policy,scenario,seed,cfg,stream,tests,data_hash,checkpoint=None):
 learner=Learner(policy,cfg,seed);trace=[];matrix=[];seen=set();train_seconds=0.
 for t,(batch,priority,segment,domain,last) in enumerate(stream):
  start=time.perf_counter();record=learner.train_window(batch,priority,t);train_seconds+=time.perf_counter()-start
  record.update(policy=policy,scenario=scenario,seed=seed,segment=segment,domain=domain);trace.append(record);seen.add(domain)
  if last:
   for d in sorted(seen):matrix.append(dict(policy=policy,scenario=scenario,seed=seed,segment=segment,test_domain=d,accuracy=learner.evaluate(*tests[d])))
 final={d:next(r['accuracy'] for r in reversed(matrix) if r['test_domain']==d) for d in sorted(seen)}
 forgetting=float(np.mean([max(r['accuracy'] for r in matrix if r['test_domain']==d)-final[d] for d in sorted(seen)]))
 params=sum(p.size for p in learner.experts[0].p)
 result=dict(scenario=scenario,policy=policy,seed=seed,final_accuracy=float(np.mean(list(final.values()))),forgetting=forgetting,experts=len(learner.experts),expansions=learner.expansion_events,blocked_events=learner.blocked_events,prototypes=len(learner.anchors),gradient_calls=learner.gradient_calls,gradient_examples=learner.gradient_examples,train_seconds=train_seconds,parameters_per_expert=params,expert_parameters=params*len(learner.experts),expert_array_bytes=sum(e.nbytes() for e in learner.experts),replay_array_bytes=learner.memory.nbytes(),persistent_array_bytes=learner.array_bytes(),data_sha256=data_hash)
 if checkpoint:
  content={f'expert{e}_array{i}':p for e,n in enumerate(learner.experts) for i,p in enumerate(n.p)}
  content.update(anchors=np.asarray(learner.anchors),anchor_experts=np.asarray(learner.anchor_experts),expiry=np.asarray(learner.expiry))
  np.savez_compressed(checkpoint,**content)
 return result,trace,matrix

def aggregate(records):
 rows=[]
 for scenario in SCENARIOS:
  for policy in POLICIES:
   rr=[r for r in records if r['scenario']==scenario and r['policy']==policy]
   if not rr:continue
   row=dict(scenario=scenario,policy=policy,n=len(rr))
   for metric in ('final_accuracy','forgetting','experts','blocked_events','gradient_examples','train_seconds','expert_array_bytes','replay_array_bytes','persistent_array_bytes'):
    a=np.asarray([r[metric] for r in rr]);row[metric+'_mean']=float(a.mean());row[metric+'_sd']=float(a.std(ddof=1)) if len(a)>1 else 0.
   rows.append(row)
 return rows

def paired(records):
 rng=np.random.default_rng(20261025);rows=[]
 for scenario in SCENARIOS:
  for p in ('dual','dual_latched'):
   a={r['seed']:r for r in records if r['scenario']==scenario and r['policy']==p};b={r['seed']:r for r in records if r['scenario']==scenario and r['policy']=='novelty'};seeds=sorted(a.keys()&b.keys())
   if not seeds:continue
   for metric in ('final_accuracy','forgetting','experts','gradient_examples','persistent_array_bytes'):
    delta=np.array([a[s][metric]-b[s][metric] for s in seeds]);boot=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
    rows.append(dict(scenario=scenario,comparison=p+'_minus_novelty',metric=metric,n_pairs=len(seeds),mean_difference=float(delta.mean()),bootstrap_low95=float(np.quantile(boot,.025)),bootstrap_high95=float(np.quantile(boot,.975))))
 return rows

def pilot(train,y,cfg,out):
 # Training and validation both come only from the first 10,000 OFFICIAL TRAIN images.
 model=Network(991,(784,cfg.hidden1,cfg.hidden2,10));rng=np.random.default_rng(992);ix=rng.permutation(8000)[:6000];start=time.perf_counter();loss=[]
 for pos in range(0,len(ix),cfg.batch_size):
  j=ix[pos:pos+cfg.batch_size]
  for _ in range(cfg.updates):
   value,g=model.loss_grad(train[j],y[j]);loss.append(value)
   for i,(p,h) in enumerate(zip(model.p,g)):p-=np.float32(cfg.learning_rate)*(h+np.float32(cfg.l2)*p if i%2==0 else h)
 accuracy=float(np.mean(np.argmax(model.forward(train[8000:10000]),axis=1)==y[8000:10000]));result=dict(train_indices=[0,7999],validation_indices=[8000,9999],train_examples=6000,validation_examples=2000,validation_accuracy=accuracy,first_loss=loss[0],last_loss=loss[-1],seconds=time.perf_counter()-start,parameters=sum(p.size for p in model.p),official_test_opened=False,config=asdict(cfg))
 json_write(out,result);print(json.dumps(result,indent=2),flush=True)
 if accuracy<.85:raise RuntimeError('Development learnability check below 85%; stop before formal testing')

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--pilot',action='store_true');p.add_argument('--seeds',type=int,default=5);p.add_argument('--output',type=Path,default=ROOT/'results'/'neural_mnist_v1');p.add_argument('--development',action='store_true');p.add_argument('--policies',nargs='+',choices=POLICIES,default=list(POLICIES));a=p.parse_args()
 if a.prepare:prepare_data();return
 train,y=load_split('train');cfg=Config();cfg.novelty_threshold,calibration=calibrate(train,cfg)
 if a.pilot:pilot(train,y,cfg,ROOT/'results'/'neural_development_pilot.json');return
 if a.output.exists() and any(a.output.iterdir()):p.error('Use an empty output directory to preserve runs')
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'checkpoints').mkdir()
 if a.development:
  cfg.examples_per_segment=1000;cfg.warmup_windows=10;cfg.test_size=2000;test_x,test_y=train[8000:10000],y[8000:10000]
 else:test_x,test_y=load_split('test')
 source=Path(__file__).read_bytes();(a.output/'run_neural.py').write_bytes(source)
 manifest=dict(status='running',protocol_sha256=hashlib.sha256((ROOT/'05_神经网络补充实验方案.md').read_bytes()).hexdigest(),config=asdict(cfg),calibration=calibration,seeds=list(range(a.seeds)),policies=a.policies,scenarios=SCENARIOS,development=a.development,source_sha256=hashlib.sha256(source).hexdigest(),dataset_provenance=json.loads((ROOT/'data'/'mnist'/'provenance.json').read_text()),test_sha256=hashlib.sha256(test_x.tobytes()+test_y.tobytes()).hexdigest(),python=sys.version,numpy=np.__version__,platform=platform.platform(),blas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),context='Same raw pixels as network input; no supplied domain/context coordinate',memory_metric='Retained NumPy array payloads; excludes process RSS and peak allocations',checkpoint_scope='Inference parameters and routing only; not a resumable optimizer checkpoint',design_note='MNIST extension designed after observing synthetic failures; protocol frozen before official MNIST testing')
 json_write(a.output/'manifest.json',manifest);all_records=[];all_trace=[];all_matrix=[];begin=time.perf_counter()
 try:
  for scenario in SCENARIOS:
   for seed in range(a.seeds):
    stream,perms,data_hash,indices=build_stream(train,y,scenario,seed,cfg,a.development)
    np.savez_compressed(a.output/f'design_{scenario}_{seed}.npz',source_indices=indices,permutations=np.array(perms))
    tests=[(transform(test_x,d,scenario,perms),test_y) for d in range(3)]
    for policy in a.policies:
     result,trace,matrix=run_one(policy,scenario,seed,cfg,stream,tests,data_hash,a.output/'checkpoints'/f'{scenario}_{seed}_{policy}.npz')
     all_records.append(result);all_trace.extend(trace);all_matrix.extend(matrix)
     csv_write(a.output/'per_run.csv',all_records);csv_write(a.output/'summary.csv',aggregate(all_records))
     print(f'{scenario} seed={seed} {policy}: accuracy={result["final_accuracy"]:.4f}, experts={result["experts"]}, train={result["train_seconds"]:.2f}s; elapsed={time.perf_counter()-begin:.1f}s',flush=True)
  csv_write(a.output/'window_log.csv',all_trace);csv_write(a.output/'accuracy_matrix.csv',all_matrix);csv_write(a.output/'paired_comparisons.csv',paired(all_records));manifest.update(status='completed',completed_runs=len(all_records),total_seconds=time.perf_counter()-begin)
 except Exception as e:manifest.update(status='failed',error=repr(e));raise
 finally:json_write(a.output/'manifest.json',manifest)

if __name__=='__main__':main()


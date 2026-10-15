#!/usr/bin/env python3
"""Offline receipt aggregation; no tolerance changes and no GPU use."""
import array,math,json,re,hashlib,sys
from pathlib import Path
p=Path(sys.argv[1]);summary={}
seeds=[]
for seed in [6,2,3,4,5,7,8]:
 row={'seed':seed}
 for value in ['0','1']:
  text=(p/f'strata-qsum-green-v1/numerics-{value}-{seed}.log').read_text()
  row[value]=float(re.search(r'^full GPU vs original unrounded baseline\s+rel_L1=([^ ]+)',text,re.M)[1])
  if value=='1':assert row[value]<.03
 seeds.append(row)
summary['seven_seeds']=seeds
for value in ['0','1']:
 text=(p/f'strata-qsum-green-v1/contract-{value}.log').read_text()
 assert len(re.findall(r'^PASS type=',text,re.M))==18 and 'FAIL' not in text
assert json.loads((p/'strata-qsum-red-v1/receipt.json').read_text())['optin_exit']==1
for r in json.loads((p/'strata-qsum-capture-v1/receipt.json').read_text())['runs']:
 assert r['exit']==0, r
summary['contract_cases_per_mode']=18
summary['default_fp_gate_failures']=sum(x['0']>=.03 for x in seeds);summary['variant_fp_gate_failures']=sum(x['1']>=.03 for x in seeds)
gens={}
for quant in ['Q4_K_M','IQ2_M']:
 r={};tokens=[]
 for value in ['0','1']:
  text=(p/f'strata-qsum-model-v1/generation-{quant}-{value}.log').read_text();ids=list(map(int,re.search(r'^gen: (.*)$',text,re.M)[1].split()));assert len(ids)==256;tokens.append(ids)
  m=re.search(r'prompt (\d+) tokens in ([\d.]+) ms.*?256 generated in ([\d.]+) ms',text);assert int(m[1])==105
  r[value]={'prefill_ms':float(m[2]),'decode_ms':float(m[3]),'normalized_decode_tps':255000/float(m[3])}
 r['identical_prefix_tokens']=next((i for i,(a,b) in enumerate(zip(*tokens)) if a!=b),256);r['equal_ids_at_same_position']=sum(a==b for a,b in zip(*tokens));r['all_ids_equal']=tokens[0]==tokens[1];gens[quant]=r
summary['generation_single_pair_pilot']=gens
v=248320;arrays=[]
for value in ['0','1']:
 a=array.array('f');a.frombytes((p/f'strata-qsum-capture-v1/logits-{value}.bin').read_bytes());assert len(a)==17*v;assert all(math.isfinite(x) for x in a);arrays.append(a)
metrics=[]
for i in range(17):
 a,b=[list(x[i*v:(i+1)*v]) for x in arrays]
 ma,mb=max(a),max(b);za=math.log(sum(math.exp(x-ma) for x in a))+ma;zb=math.log(sum(math.exp(x-mb) for x in b))+mb
 kl=sum(math.exp(x-za)*(x-za-y+zb) for x,y in zip(a,b));metrics.append(dict(step=i,kl_default_to_variant=kl,top1_equal=a.index(ma)==b.index(mb),max_abs_logit_difference=max(abs(x-y) for x,y in zip(a,b))))
summary['teacher_forced_drift']={'reference':'default same quant/model; NOT full precision or ground truth','steps':17,'forced_continuation_tokens':16,'mean_kl':sum(x['kl_default_to_variant'] for x in metrics)/17,'top1_agreement':sum(x['top1_equal'] for x in metrics)/17,'max_abs_logit_difference':max(x['max_abs_logit_difference'] for x in metrics),'per_step':metrics}
summary['quality']='Task accuracy / held-out perplexity / quality improvement NOT measured.'
(p/'verified-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

import sys,os,re,json,glob,ast,random,time,csv
import numpy as np
sys.path.insert(0,'/scratch/manishn_iitp/PhD/revision2-workspace/external/Quantized-Reasoning-Models')
import logging; logging.disable(logging.CRITICAL)
from lighteval_custom.metrics.dynamic_metrics import ExprExtractionConfig, LatexExtractionConfig
from lighteval.metrics.utils.extractive_match_utils import get_extraction_regexes, extract_target_from_pred
from lighteval.metrics.utils.math_comparison import compare_gold_target
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language
from multiprocessing import Pool
W='/scratch/manishn_iitp/PhD/revision2-workspace/'; OUT=W+'exports/c2_build_2026-10-23/'
CAMP='/scratch/manishn_iitp/reasoning-compression-lab/outputs-hpc-campaign-2026-08-14/inference/'
FAM={'Qwen':'DeepSeek-R1-Distill-Qwen-7B','Llama':'DeepSeek-R1-Distill-Llama-8B'}
key=lambda s: re.search(r'<｜User｜>(.*?)\n\nPlease reason',s,re.S).group(1).strip()
def pg(g):
    try:
        v=ast.literal_eval(g) if isinstance(g,str) and g[:1] in '[(' else g
        return v[0] if isinstance(v,(list,tuple)) else v
    except Exception: return g
# canonical items
base=json.load(open(CAMP+FAM['Qwen']+'-seed42/MATH-500.jsonl'))
KEYS=[key(r['full_prompt']) for r in base]; POS={k:i for i,k in enumerate(KEYS)}; GOLD=[pg(r['gold']) for r in base]
assert len(POS)==500
def aligned(rows_text, rows_key, correct, finished):
    """-> dict item -> (text, correct, finished)"""
    d={}
    for t,k,c,f in zip(rows_text,rows_key,correct,finished): d[POS[k]]=(t,bool(c),bool(f))
    assert len(d)==500; return d
cells={}   # arm -> list of (cellname, dict)
def add(arm,name,d): cells.setdefault(arm,[]).append((name,d))
def mcheck(name,raw,det):
    a=[float(eval(r['metrics'])['extractive_match']) if isinstance(r['metrics'],str) else float(r['metrics']['extractive_match']) for r in raw]
    b=[float(x['extractive_match']) for x in sorted(det,key=lambda x:x['row'])]
    if a!=b: print('ALIGNMENT MISMATCH',name,sum(x!=y for x,y in zip(a,b))); return False
    return True
skipped=[]
# campaign
for fam,fn in FAM.items():
    for fmt in ('BF16','FP8','AWQ-4','GPTQ-4'):
        cn=fn if fmt=='BF16' else f'{fn}-{fmt}'
        for s in range(42,47):
            raw=json.load(open(f'{CAMP}{cn}-seed{s}/MATH-500.jsonl')); det=json.load(open(f'/scratch/manishn_iitp/PhD/reasoning-compression-lab/results/math500/{cn}_math500_n500_seed{s}.json'))['details']
            det=sorted(det,key=lambda x:x['row'])
            if not mcheck(cn+str(s),raw,det): skipped.append((cn,s,'alignment')); continue
            d=aligned([r['generated_text'] for r in raw],[key(r['full_prompt']) for r in raw],[x['extractive_match'] for x in det],[not x['hit_token_limit'] for x in det])
            add(f'camp_{fam}_{fmt}',f'{cn}-seed{s}',d)
# revision2/3 verified cells
def rev(arm,base_dir,name,seeds=range(42,47),vd=None):
    for s in seeds:
        cd=f'{W}results/{base_dir}/inference/{name}-seed{s}/'
        vf=glob.glob(f'{W}results/{base_dir}/validation/{name}_math500_n500_seed{s}.json')
        if not (os.path.exists(cd+'VERIFIED') and os.path.exists(cd+'MATH-500.jsonl') and vf): skipped.append((name,s,'no VERIFIED MATH-500 raw/validation')); continue
        raw=json.load(open(cd+'MATH-500.jsonl')); det=sorted(json.load(open(vf[0]))['details'],key=lambda x:x['row'])
        if len(raw)!=500 or not mcheck(name+str(s),raw,det): skipped.append((name,s,'alignment/len')); continue
        add(arm,f'{name}-seed{s}',aligned([r['generated_text'] for r in raw],[key(r['full_prompt']) for r in raw],[x['extractive_match'] for x in det],[not x['hit_token_limit'] for x in det]))
rev('e1_Llama_selfawq4','revision2/E1_awq','DeepSeek-R1-Distill-Llama-8B-selfawq')
rev('e1c_Llama_selfawq4_gemm','revision2/E1c_awq_gemm','DeepSeek-R1-Distill-Llama-8B-selfawq_gemm')
for fam,fn in FAM.items():
    rev(f'e2_{fam}_fp16','revision2/E2_fp16',fn+'_fp16'); rev(f'e3_{fam}_T0.8','revision2/E3_T08',fn+'_T0.8')
# P2 Qwen controlled (job 107150): text from predictions, correctness/finish from the v4b compact (extractive_match, finish_reason)
comp={}
for f in glob.glob(W+'exports/v4b_build_2026-10-22/compact/107150__*.json'):
    h=json.load(open(f)); comp[h['cell'].split('/')[-2]+'/'+h['cell'].split('/')[-1]]=h
for cellpath,h in sorted(comp.items()):
    pf='/scratch/manishn_iitp/PhD/reasoning-compression-lab/papers/'+h['cell']+'/artifacts/predictions.jsonl'
    rows=[json.loads(l) for l in open(pf)]
    assert len(rows)==500 and len(h['details'])==500
    method=h['variant_id'].split('4_')[0] if '4_' in h['variant_id'] else h['variant_id']
    arm='p2_Qwen_'+('awq4' if h['variant_id'].startswith('awq') else 'gptq4')+'_controlled_draws0-2_seeds42-44'
    add(arm,f"{h['variant_id']}-seed{h['seed']}",aligned([r['completion'] for r in rows],[r['problem'].strip() for r in rows],[x['extractive_match'] for x in h['details']],[x['finish_reason']!='length' for x in h['details']]))
# ---- scorer pieces
LANG=Language.ENGLISH; DOC=Doc(query='',choices=[''],gold_index=0,specific={})
GREG=get_extraction_regexes(DOC,(LatexExtractionConfig(),),LANG); PREG=get_extraction_regexes(DOC,(ExprExtractionConfig(),LatexExtractionConfig(boxed_match_priority=0)),LANG)
_gcache={}
def gold_ext(i):
    if i not in _gcache:
        try: _gcache[i]=extract_target_from_pred(str(GOLD[i]),GREG,'first_match','any_match',5)
        except Exception: _gcache[i]=[]
    return _gcache[i]
def boxed_all(t):
    out=[]; i=0
    while True:
        j=t.find('\\boxed',i)
        if j<0: break
        k=t.find('{',j)
        if k<0 or k-j>8: i=j+6; continue
        d=0;e=None
        for p in range(k,len(t)):
            if t[p]=='{': d+=1
            elif t[p]=='}':
                d-=1
                if d==0: e=p;break
        if e is None: break
        out.append((t[k+1:e],j)); i=e+1
    return out
def flat_str(x):
    r=[]
    for e in x:
        if isinstance(e,str): r.append(e)
        elif isinstance(e,(list,tuple)): r+=flat_str(e)
    return r
def work(a):
    arm,cell,i,t=a
    g=gold_ext(i); pa=False; pa_pos=None
    if g:
        seen=set()
        for b,pos in boxed_all(t)[:40]:
            if b in seen: continue
            seen.add(b)
            if len(seen)>12: break
            try:
                ep=extract_target_from_pred('\\boxed{'+b+'}',PREG,'first_match','any_match',5)
                if ep and compare_gold_target(g,ep,5,timeout_seconds=5): pa=True; pa_pos=pos; break
            except Exception: pass
    gs=[s for s in dict.fromkeys(flat_str(g)+[str(GOLD[i])]) if s and len(s)<=60]
    # gold_ext strings only (normalized by the scorer); raw gold added for latex forms
    half=len(t)//2; tail=t[half:]; pb=False; pb_pos=None
    for s in gs:
        m=re.search(r'(?<![A-Za-z0-9_.\\])'+re.escape(s)+r'(?![A-Za-z0-9_]|\.\d)',tail)
        if m: pb=True; pb_pos=half+m.start(); break
    lastb=boxed_all(t); lp=lastb[-1][1] if lastb else len(t)
    def sn(p): s=max(0,p-200); return t[s:s+400].replace('\t',' ').replace('\n',' ')
    return arm,cell,i,pa,pb,sn(pa_pos) if pa_pos is not None else sn(max(0,lp-100)),sn(pb_pos) if pb_pos is not None else sn(max(0,len(t)-300))
if __name__=='__main__':
    t0=time.time(); tasks=[]
    for arm,lst in cells.items():
        for cell,d in lst:
            for i,(t,c,f) in d.items():
                if (not c) and f: tasks.append((arm,cell,i,t))
    print('arms',len(cells),'cells',sum(len(v) for v in cells.values()),'incorrect finished completions',len(tasks),'skipped',len(skipped),flush=True)
    with Pool(4) as p: res=p.map(work,tasks,chunksize=8)
    print('scored',len(res),f'{time.time()-t0:.0f}s',flush=True)
    R={(a,c,i):(pa,pb,sa,sb) for a,c,i,pa,pb,sa,sb in res}
    INC={};FA={};FB={};NC={};NCOMP={}
    for arm,lst in cells.items():
        inc=np.zeros(500);fa=np.zeros(500);fb=np.zeros(500)
        for cell,d in lst:
            for i,(t,c,f) in d.items():
                if (not c) and f:
                    inc[i]+=1; pa,pb,_,_=R[(arm,cell,i)]; fa[i]+=pa; fb[i]+=pb
        INC[arm]=inc;FA[arm]=fa;FB[arm]=fb;NC[arm]=len(lst);NCOMP[arm]=500*len(lst)
    rng=np.random.default_rng(0); B=10000; IDX=rng.integers(0,500,(B,500))
    def share(num,den):
        n=num[IDX].sum(1); d=den[IDX].sum(1); return np.where(d>0,n/np.where(d>0,d,1),np.nan)
    ci=lambda x:(float(np.nanpercentile(x,2.5)),float(np.nanpercentile(x,97.5)))
    S={}
    with open(OUT+'c2_arm_table.tsv','w') as f:
        f.write('arm\tn_cells\tn_completions\tn_incorrect_finished\tn_flag_Pa\tn_flag_Pb\tshare_Pa\tPa_ci95_lo\tPa_ci95_hi\tshare_Pb\tPb_ci95_lo\tPb_ci95_hi\n')
        for arm in cells:
            sa=share(FA[arm],INC[arm]); sb=share(FB[arm],INC[arm]); S[arm]=(sa,sb)
            n=INC[arm].sum(); a=FA[arm].sum(); b=FB[arm].sum()
            la=ci(sa);lb=ci(sb)
            f.write('\t'.join(map(str,[arm,NC[arm],NCOMP[arm],int(n),int(a),int(b),f'{a/n:.4f}' if n else 'NA',f'{la[0]:.4f}',f'{la[1]:.4f}',f'{b/n:.4f}' if n else 'NA',f'{lb[0]:.4f}',f'{lb[1]:.4f}']))+'\n')
    # BF16 references: same family; seeds 42-44 subset for P2 arms
    def bf16_arm(fam,seeds):
        name=f'camp_{fam}_BF16'+('' if seeds is None else '_s42-44')
        if name not in S:
            inc=np.zeros(500);fa=np.zeros(500);fb=np.zeros(500)
            for cell,d in cells[f'camp_{fam}_BF16']:
                if seeds is not None and int(cell.rsplit('seed',1)[1]) not in seeds: continue
                for i,(t,c,f_) in d.items():
                    if (not c) and f_:
                        inc[i]+=1; pa,pb,_,_=R[(f'camp_{fam}_BF16',cell,i)]; fa[i]+=pa; fb[i]+=pb
            INC[name]=inc;FA[name]=fa;FB[name]=fb;S[name]=(share(fa,inc),share(fb,inc))
        return name
    with open(OUT+'c2_pairwise_vs_bf16.tsv','w') as f:
        f.write('arm\treference\tdiff_Pa_arm_minus_ref\tPa_ci95_lo\tPa_ci95_hi\tdiff_Pb_arm_minus_ref\tPb_ci95_lo\tPb_ci95_hi\tsame_sign_Pa_Pb\tn_inc_fin_arm\tn_inc_fin_ref\n')
        for arm in cells:
            if arm.endswith('_BF16'): continue
            fam='Qwen' if 'Qwen' in arm else 'Llama'
            ref=bf16_arm(fam,{42,43,44} if arm.startswith('p2_') else None)
            da=S[arm][0]-S[ref][0]; db=S[arm][1]-S[ref][1]
            pa=float(np.nanmean(da)); pb=float(np.nanmean(db))
            pa_pt=FA[arm].sum()/INC[arm].sum()-FA[ref].sum()/INC[ref].sum(); pb_pt=FB[arm].sum()/INC[arm].sum()-FB[ref].sum()/INC[ref].sum()
            la=ci(da);lb=ci(db)
            f.write('\t'.join(map(str,[arm,ref,f'{pa_pt:.4f}',f'{la[0]:.4f}',f'{la[1]:.4f}',f'{pb_pt:.4f}',f'{lb[0]:.4f}',f'{lb[1]:.4f}',(pa_pt>0)==(pb_pt>0),int(INC[arm].sum()),int(INC[ref].sum())]))+'\n')
    with open(OUT+'c2_cell_counts.tsv','w') as f:
        f.write('arm\tcell\tn_incorrect_finished\tn_Pa\tn_Pb\n')
        for arm,lst in cells.items():
            for cell,d in lst:
                ks=[(i) for i,(t,c,f_) in d.items() if (not c) and f_]
                f.write(f'{arm}\t{cell}\t{len(ks)}\t{sum(R[(arm,cell,i)][0] for i in ks)}\t{sum(R[(arm,cell,i)][1] for i in ks)}\n')
    rnd=random.Random(0); allk=list(R.keys())
    with open(OUT+'c2_validation_snippets.tsv','w') as f:
        f.write('proxy\tflag\tarm\tcell\titem_dataset_row\tsnippet_max400chars\n')
        for pxi,px in ((0,'P-a'),(1,'P-b')):
            fl=[k for k in allk if R[k][pxi]]; un=[k for k in allk if not R[k][pxi]]
            for flag,pool in (('flagged',fl),('unflagged',un)):
                for k in rnd.sample(pool,min(40,len(pool))):
                    f.write('\t'.join([px,flag,k[0],k[1],str(k[2]),R[k][2+pxi].replace('\t',' ')[:400]])+'\n')
    json.dump({'skipped':skipped,'n_arms':len(cells),'cells_per_arm':NC,'method':'see c2_notes.md'},open(OUT+'c2_meta.json','w'),indent=1)
    print('DONE',f'{time.time()-t0:.0f}s')

"""Validate all tables without PyBullet, scoring code or the large R6/R7 source archives."""
from pathlib import Path
import sys, csv, json, hashlib, math
from collections import defaultdict, Counter
from recalculate import effects
V=Path(__file__).resolve().parents[1]
def read(n):
    with (V/'tables'/n).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))
def must(v,why):
    if not v:raise ValueError(why)
def eq(a,b,why):
    must(math.isclose(float(a),float(b),rel_tol=0.,abs_tol=1e-10),why)
g={'R6':defaultdict(dict),'R7':defaultdict(dict)}
rows=read('trial_rows.csv')
must(len(rows)==960 and len({r['trial_id'] for r in rows})==960,'source trial rows')
for raw in rows:
    row=dict(raw);row['candidate']=row['candidate'] or None
    for key in ('static_pass','physical_returns','success'):row[key]=int(row[key])
    q=g[row['phase']][row['unit_id']]
    must(row['method'] not in q,'method duplicate')
    q[row['method']]=row
for uid,r in g['R7'].items():
    must('C0' not in r,'reference is not a new R7 trial')
    r['C0']=g['R6'][uid]['C0']
rates={(r['phase'],r['stratum'],r['method']):r for r in read('rates.csv')}
pairs={(r['phase'],r['stratum'],r['contrast']):r for r in read('paired_effects.csv')}
failures=defaultdict(dict)
for r in read('failure_counts.csv'):
    failures[r['phase'],r['stratum'],r['method']][r['detail']]=int(r['n'])
proof=json.loads((V/'tables/audit.json').read_text('utf8'))
by4=defaultdict(lambda:{'success':[],'failure':[]});totals={};rc=pc=fc=0
for phase,arms in [('R6',['Raw','A','C0']),('R7',['C0','C0_no_instance','C0_depth_only','C0_warp_only'])]:
    units=g[phase]
    must(len(units)==(240 if phase=='R6' else 80),'paired sample size')
    for uid,methods in units.items():
        must(set(methods)==set(arms),'arms')
        if phase=='R7':
            for m in arms:
                by4[m]
                vals=[r['success'] for k,r in methods.items() if m!=k]
                if methods[m]['success'] and max(vals)==0:by4[m]['success'].append(uid)
                if not methods[m]['success'] and min(vals)==1:by4[m]['failure'].append(uid)
    compares=[('C0','A'),('A','Raw'),('C0','Raw')] if phase=='R6' else [(m,'C0') for m in arms[1:]]
    for sub in ('all','clean','joint','014','017','021'):
        uu={i:r for i,r in units.items() if sub=='all' or sub in (r[arms[0]]['condition'],r[arms[0]]['object_id'])}
        x=effects(uu,arms,compares,proof['bootstrap_seed'],proof['bootstrap_draws'])
        for m,u in x['arms'].items():
            t=rates[phase,sub,m]
            for k,actual in {'assigned':u['n'],'success':u['success'],'rate_pct':100*u['rate'],
                    'wilson_low_pct':100*u['wilson95'][0],'wilson_high_pct':100*u['wilson95'][1],
                    'empty':u['empty'],'static_rejected':u['static_fail'],'static_admitted':u['static_pass'],
                    'static_pass_all_pct':100*u['static_rate'],'dynamics_entered':u['dynamic'],
                    'native_returned_steps':u['native_returns']}.items():eq(actual,t[k],k)
            if u['conditional'] is None:must(t['conditional_success_pct']=='','missing conditional denominator')
            else:eq(100*u['conditional'],t['conditional_success_pct'],'conditional rate')
            must(dict(u['detail'])==failures[phase,sub,m],'route table');fc+=len(u['detail']);rc+=1
        for key,u in x['pairs'].items():
            t=pairs[phase,sub,key]
            for k,a in {'assigned':u['n'],'first_only':u['first_only'],'second_only':u['second_only'],'both_success':u['both_success'],
                    'both_failure':u['both_fail'],'delta_pp':u['delta_pp'],'CI_low_pp':u['paired95_pp'][0],'CI_high_pp':u['paired95_pp'][1],
                    'exact_mcnemar_two_sided':u['exact_mcnemar'],'discordant':u['discordant_n'],'different_candidates':u['different'],
                    'different_pct':100*u['different_rate'],'both_empty':u['both_empty'],'same_nonempty':u['same_nonempty']}.items():eq(a,t[k],k)
            pc+=1
    picked=[r for unit in units.values() for m,r in unit.items() if phase!='R7' or m!='C0']
    totals[phase]=dict(trials=len(picked),admitted=sum(r['static_pass'] for r in picked),success=sum(r['success'] for r in picked),
                       native_returns=sum(r['physical_returns'] for r in picked),routes=dict(Counter(r['detail'] for r in picked)))
must((rc,pc,fc)==(len(rates),len(pairs),sum(len(x) for x in failures.values())),'table coverage')
manifest=V/'SHA256SUMS.txt'
if manifest.is_file():
    for line in manifest.read_text('utf8').splitlines():
        dig,rel=line.split('  ',1);p=(V/rel).resolve()
        must(p.is_relative_to(V) and hashlib.sha256(p.read_bytes()).hexdigest()==dig,rel)
print(json.dumps(dict(status='passed',unique_trials=len(rows),reference_links=80,rates_verified=rc,paired_comparisons_verified=pc,
                      failure_cells_verified=fc,totals=totals,R7_unique_among_four=dict(by4)),ensure_ascii=False,indent=2))

"""Describe evidence membership separately from scientific file reads. No historical payload written into runtime."""
import collections,gzip,hashlib,json
from pathlib import Path
V=Path(__file__).resolve().parents[1];OLD=V.parent/'V1_repro_dependency_closure03';B=V/'adapter';P=V/'cold/p'
def read(p):return json.loads(p.read_bytes())
def sha(f):
    with f.open('rb')as s:return hashlib.file_digest(s,'sha256').hexdigest()
def put(p,d):
    with p.open('x',encoding='utf8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
a=read(V/'audit/font_AB_and_rank_gate.json');assert a['passed']
provenance=read(V/'provenance_evidence_appendix/manifest.json')['files'];names={r['registered_path'] for r in provenance}
g=OLD/'audit/runtime_file_open_graph.jsonl.gz';proj=collections.defaultdict(lambda:dict(calls=0,sites=set(),cases=set()))
for l in gzip.open(g,'rt',encoding='utf8'):
    r=json.loads(l)
    if r['write'] or r['status']!='opened_file_present_after_probe' or r['purpose'] not in ('exact_frozen_project','authorized_smoke11_volume'):continue
    path=r['registered_path']
    assert path not in names
    item=proj[path];item['cases'].add(r['case']);item['sites'].add((r['caller'],r['location'],r['function']));item['calls']+=r['repeats']
    assert not item.get('sha256') or item['sha256']==r['sha256']
    item.update(sha256=r['sha256'],bytes=r['bytes'])
for path,item in proj.items():
    q=P/path
    if q.is_file():assert q.stat().st_size==item['bytes'] and sha(q)==item['sha256']
    item.update(three_case_initialization_evidence=True,available_before_cold=q.is_file(),cases=sorted(item['cases']),sites=sorted(item['sites']))
host=read(OLD/'audit/host_data_dependencies.json')
fonts=[{k:r[k] for k in ('actual_file','bytes','sha256')} for r in host['opened_files_outside_deployment_and_system_root'] if 'kingsoft' in r['actual_file'].casefold()]
assert len(fonts)==2
put(V/'audit/dependency_classes.json',dict(scientific_runtime=dict(project_opened_paths=proj,frozen_project_lock='adapter/locks/execution_project.jsonl.gz',runtime_supplement='audit/runtime_supplement.json',environment_locks='adapter/locks',actual_cold_file_origins='cold/cold_receipts/<case>/<role>_data_scope.json and actual native source receipts, where dispatched',previous_open_graph_sha256=sha(g)),provenance_only=dict(entries=158,registered_bytes=sum(r['bytes']for r in provenance),opened_in_three_case_precheck=0,deployed_paths_before_cold=sum((P/n).exists() for n in names),index='provenance_evidence_appendix/manifest.json',payload='provenance_evidence_appendix/registered_originals.zip',mounted=False),incidental_font_resources=dict(historical_host_files=fonts,copied_host_fonts=0,scope='These three frozen cases, fieldwise scientific equality and both secured arms.',AB_gate_sha256=sha(V/'audit/font_AB_and_rank_gate.json'))))
print('Closure indexed',len(proj),'observed project paths')

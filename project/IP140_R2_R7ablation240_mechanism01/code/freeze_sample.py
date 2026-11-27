"""Choose units without access to any method output. Only pre-treatment unit labels and hashes rank scenes."""
from collections import Counter, defaultdict
from itertools import product, permutations
from common import *

PARENT = ROOT / 'IP140_R2_R6formal720_static_distance_uncertified_resume01'
ARMS = ('C0_no_instance', 'C0_depth_only', 'C0_warp_only')
STRATA = ('014', '017', '021')

def unit_hash(identifier):
    return hashlib.sha256(identifier.encode('ascii')).hexdigest()

def main():
    design_path = PARENT/'preregistration/design.json'
    require(sha(design_path) == read(PARENT/'preregistration/design_seal.json')['design_sha256'], 'R6 design hash')
    original = read(design_path)
    units = original['units']
    require(len(units) == len({u['unit_id'] for u in units}) == 240, 'formal input registry')
    clusters = defaultdict(list)
    for u in units:
        clusters[(u['object_id'], u['position'], u['scene'])].append(u)
    blocks = defaultdict(list)
    for (obj, pos, scene), pair in clusters.items():
        require(sorted(u['condition'] for u in pair) == ['clean','joint'], 'base scene pair')
        key = tuple(sorted(unit_hash(u['unit_id']) for u in pair))
        blocks[obj,pos].append((key,scene))
    require(len(blocks)==12 and all(len(x)==10 for x in blocks.values()), 'stratum coverage')
    for values in blocks.values():
        values.sort()
    # Three base scenes per stratum, then one additional scene at each position.
    # The least sum of fourth-pair digests assigns the four extra places under the object quota.
    def cost(objects):
        return sum(int(blocks[obj,pos][3][0][0],16) for pos,obj in enumerate(objects,1)), objects
    options = (choice for choice in product(STRATA,repeat=4)
               if sorted(Counter(choice).values())==[1,1,2])
    extras = min(options,key=cost)
    selected = {scene for (obj,pos),values in blocks.items()
                for _,scene in values[:3+int(obj==extras[pos-1])]}
    keep = sorted((u for u in units if u['scene'] in selected),key=lambda u:unit_hash(u['unit_id']))
    require(len(keep)==80 and len(selected)==40 and Counter(u['condition'] for u in keep)=={'clean':40,'joint':40}, 'sample count')
    require(sorted(Counter(u['position'] for u in keep).values())==[20]*4 and
            sorted(Counter(u['object_id'] for u in keep).values())==[26,26,28], 'balanced sample')
    schedule = list(permutations(ARMS))
    trials = []
    for seq,u in enumerate(keep):
        for i,arm in enumerate(schedule[seq%6],1):
            key = f'R7ABL-{len(trials)+1:03d}'
            seed = int(unit_hash('r7-method|'+u['unit_id']+'|'+arm)[:8],16)
            trials.append(dict(trial_id=key, unit_id=u['unit_id'],method=arm,order=len(trials)+1,
                               within_unit=i,worker_seed=seed))
    require(len(set(t['worker_seed'] for t in trials))==240,'seed collision')
    doc = dict(version=HERE.name, parent=PARENT.name, utc=now(), n_units=80, n_trials=240, n_scenes=40,
               groups=ARMS, reference='archived R6 C0 per identical unit (no new C0 trial)',
               original_proposal='docs/RGB-D六自由度机器人抓取研究重规划与仿真验证方案_20261103.md',
               frozen_definitions='IP140-R1_参数与证据附表_小修02.pdf',
               explicit_80_unit_roster_found=False, selection_data=['R6 preregistered unit_id, base scene, object, position, condition'],
               method_results_accessed_by_sampler=0,
               sampling='rank each clean/joint pair by sorted SHA256(unit_id) tuple. In each object-position take first three pairs. One extra pair per position, one or two per object; minimize sum of fourth-pair primary digest integers, then lexical object tuple.',
               all_unit_hashes={u['unit_id']:unit_hash(u['unit_id']) for u in units},
               extra_object_per_position=list(extras), scenes=[s for s in original['scenes'] if s['base_scene'] in selected],
               units=keep,trials=trials,ordered_methods='round-robin six permutations in unit SHA order',
               bootstrap_seed=original['bootstrap_seed'],bootstrap_reps=original['bootstrap_reps'],
               statistical_plan='all 80 pairs. Signed difference ablation minus full C0. Percentile 95% using 20,000 paired draws of whole base scenes inside object-position blocks, preserving clean/joint. Conditional dynamic success denominator = methods with positive native physical returns. Selection divergence denominator = all assigned pairs. Unique successes/failures reported both versus paired full C0 and across all four configurations.',
               formulas={'C0_no_instance':'set all 81 association multipliers to 1, keep frozen prior and fixed-depth bilateral support and collision',
                         'C0_depth_only':'unit mass at zero registration offset, fixed t3 scale .005 m for contact and collision',
                         'C0_warp_only':'empirical 81-offset prior, native point-depth contact and collision'},
               old_E='use archived target-independent prelock eligibility, fresh postlock target-specific path checks; no rank fallback',
               no_new_confirmations=True)
    d=put(HERE/'preregistration/design.json',doc)
    put(HERE/'preregistration/design_seal.json',dict(design_sha256=d,parent_design_sha256=sha(design_path),definitions_sha256=sha(ROOT/doc['frozen_definitions']),plan_sha256=sha(ROOT/doc['original_proposal']),sample_source_sha256=sha(__file__),selection_frozen_before_outcomes=True))
    for i in range(1,5):
        lane=HERE/'workspaces'/f'cpu{i:02d}'
        put(lane/'workspace.json',dict(lane=i,version=HERE.name))
    print('selected',len(keep),len(trials),d,flush=True)

if __name__=='__main__':
    main()

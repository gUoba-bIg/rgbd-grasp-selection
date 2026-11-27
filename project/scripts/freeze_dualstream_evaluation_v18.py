"""Bind the predeclared evaluation only after both final training runs finish."""
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8-sig'))


def digest(name):
    return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()


def write_text(name, text):
    file = ROOT/name
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(file.name+'.preparation.tmp')
    assert not file.is_symlink() and not temporary.is_symlink()
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(file)


def write(name, value):
    write_text(name, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+'\n')


def install_stages(protocol):
    stages = read('configs/stages.json')
    stage_inputs = sorted(set(['configs/dualstream_evaluation_v18.json']+list(protocol['source_hashes'])+list(protocol['input_hashes'])))
    exe = 'runtime/envs/grasp_eval/Scripts/python.exe'
    out = protocol['outputs']
    definitions = [
        ('P4S18', 'P4T18', '第十八轮四条件最终权重评分与逐帧恢复',
            [exe, '-B', 'scripts/run_dualstream_evaluation_v18.py', '--phase', 'scores'],
            [out['scores_root'], out['score_resume_report']]),
        ('P4A18', 'P4S18', '第十八轮四条件88块官方实例AP及恢复',
            [exe, '-B', 'scripts/run_dualstream_evaluation_v18.py', '--phase', 'metrics'],
            [out['metrics_root'], out['metrics_report'], out['metrics_report']+'.sha256.json', out['metrics_resume_report']]),
        ('P4E18', 'P4A18', '第十八轮独立四条件配对评价验收',
            [exe, '-B', 'scripts/accept_dualstream_evaluation_v18.py'],
            [out['acceptance_report'], out['acceptance_report']+'.sha256.json', out['visuals_dir']])]
    for identifier, dependency, title, command, outputs in definitions:
        expected = dict(id=identifier, title=title, implemented=True,
            depends_on=[dependency], command=command, inputs=stage_inputs, outputs=outputs)
        existing = next((stage for stage in stages['stages'] if stage['id'] == identifier), None)
        if existing is None:
            stages['stages'].append(expected)
        else:
            assert existing == expected, 'Existing evaluation stage differs: '+identifier
    p4 = next(stage for stage in stages['stages'] if stage['id'] == 'P4')
    if 'P4E18' not in p4['depends_on']:
        p4['depends_on'].append('P4E18')
    if read('configs/stages.json') != stages:
        write('configs/stages.json', stages)


def main():
    name = 'configs/dualstream_evaluation_v18.json'
    protocol = read(name)
    if protocol['status'] == 'frozen':
        for field in ('source_hashes', 'input_hashes'):
            for file, checksum in protocol[field].items():
                assert digest(file) == checksum, file
        install_stages(protocol)
        print(json.dumps({'status': 'verified_frozen_stages_installed', 'protocol_sha256': digest(name)}))
        return
    assert protocol['status'] == 'unfrozen_preparation'
    state = read('runtime/pipeline/state.json')['stages']
    assert all(state[key]['status'] == 'complete' for key in ('P4C18', 'P4T18'))
    assert not any((ROOT/protocol['outputs'][key]).exists() for key in protocol['outputs'])
    training = read(protocol['parent_training_protocol'])
    report = read(protocol['parent_training_report'])
    assert report['schema'] == 18 and report['status'] == 'completed'
    assert report['research_optimizer_steps_completed'] == 1600
    assert report['protocol_sha256'] == digest(protocol['parent_training_protocol'])
    assert report['resume_proof_sha256'] == digest('results/P4/segmentation_dualstream_v18_resume.json')
    assert all(report['arms'][arm]['result']['optimizer_steps'] == 800 for arm in ('fixed', 'quality'))
    assert report['automatic_extension'] is False and report['validation_evaluated'] is False
    log = 'logs/P4/dualstream_evaluation_v18/component_tests.log'
    text = (ROOT/log).read_text(encoding='utf-8')
    assert list(map(int, re.findall(r'Ran (\d+) tests? in', text))) == [22, 30, 13, 44, 3]
    assert text.count('\nOK\n') == 5 and 'FAILED' not in text and 'skipped=' not in text
    readonly_log = 'logs/P4/dualstream_evaluation_v18/readonly_entry_tests.log'
    readonly_text = (ROOT/readonly_log).read_text(encoding='utf-8')
    assert re.search(r'Ran 11 tests in', readonly_text) and '\nOK\n' in readonly_text
    assert 'FAILED' not in readonly_text and 'skipped=' not in readonly_text
    sources, inputs = {}, {}
    for parent in (read('configs/mask_ap_v17.json'), training):
        for target, field in ((sources, 'source_hashes'), (inputs, 'input_hashes')):
            for file, checksum in parent[field].items():
                assert digest(file) == checksum, file
                assert file not in target or target[file] == checksum, file
                target[file] = checksum
    completion = 'results/P4/segmentation_dualstream_v18_stage_completion.json'
    value = {key: state[key] for key in ('P4C18', 'P4T18')}
    if (ROOT/completion).exists():
        assert read(completion) == value
    else:
        write(completion, value)
    extra_sources = [
        'scripts/extract_dualstream_scores_v18.py', 'scripts/evaluate_dualstream_mask_ap_v18.py',
        'scripts/accept_dualstream_evaluation_v18.py', 'scripts/run_dualstream_evaluation_v18.py',
        'scripts/test_dualstream_evaluation_v18.py', 'scripts/resume_dualstream_evaluation_v18.ps1',
        'scripts/resume_dualstream_sequence_v18.ps1',
        'scripts/freeze_dualstream_evaluation_v18.py',
        'scripts/verify_dualstream_evaluation_readonly_v18.py', 'tests/test_dualstream_readonly_v18.py', readonly_log,
        'tests/test_dualstream_scores_v18.py', 'tests/test_dualstream_ap_v18.py',
        'tests/test_dualstream_acceptance_v18.py', 'tests/test_dualstream_evaluation_wrapper_v18.py',
        'tests/test_dualstream_training_entry_v18.py', 'logs/P4/segmentation_dualstream_v18_entry_tests.log', log,
        'logs/P4/dualstream_evaluation_v18/freeze_entry_checks.json',
        'docs/第十八轮四条件评价冻结方案.txt']
    extra_inputs = [protocol['parent_training_protocol'], protocol['parent_training_report'],
        'configs/mask_ap_v17.json', 'results/P4/mask_ap_v17_acceptance.json',
        'results/P4/mask_ap_v17_readonly_verification.json', 'logs/P4/mask_ap_v17/cold_reentry.json',
        'results/P4/segmentation_dualstream_v18_resume.json', 'results/P4/segmentation_dualstream_v18_resume.json.sha256.json',
        'results/P4/segmentation_dualstream_v18_entry_correction.json', 'configs/retired_stage_plans/P4T18_initial_cli.json', completion]
    extra_inputs += list(read('results/P4/segmentation_dualstream_v18_entry_correction.json')['failed_logs'])
    for env in ('vision', 'grasp_eval', 'coco_eval_v17'):
        extra_inputs += ['runtime/envs/'+env+'/'+part for part in ('Scripts/python.exe', 'pyvenv.cfg')]
    for key in ('P4C18', 'P4T18'):
        extra_inputs += [state[key]['stdout'], state[key]['stderr']]
    extra_inputs += [file.relative_to(ROOT).as_posix()
        for file in sorted((ROOT/'logs/P4/segmentation_dualstream_v18').rglob('*')) if file.is_file()]
    for file in extra_sources+extra_inputs:
        assert (ROOT/file).is_file(), file
    for scene in protocol['validation_scenes']:
        for frame in protocol['frames']:
            label = 'data/raw/graspnet_coverage_v8/scenes/scene_%04d/realsense/label/%04d.png' % (scene, frame)
            assert label in inputs and digest(label) == inputs[label], label
    doc = ROOT/'docs/第十八轮四条件评价冻结方案.txt'
    text = doc.read_text(encoding='utf-8')
    old = ('状态：准备中，未冻结；当前不允许实际评分或标签评价。', '实际评分前仍须绑定完成后的最终训练报告及协议SHA。')
    new = ('状态：已冻结；配对研究训练及真实训练恢复均已完成，允许按本合同评分和评价。',
        '最终训练报告及协议SHA已绑定到评价配置；训练更新和评价选择均不再变更。')
    assert all(value in text for value in old) or all(value in text for value in new), 'Incomplete protocol document'
    assert '这里的进程恢复验证不等同真实物理断电试验。' in text
    for before, after in zip(old, new):
        text = text.replace(before, after)
    write_text(doc.relative_to(ROOT), text)
    sources.update({file: digest(file) for file in extra_sources})
    inputs.update({file: digest(file) for file in extra_inputs})
    protocol.update(status='frozen', source_hashes=dict(sorted(sources.items())), input_hashes=dict(sorted(inputs.items())),
        parent_training_protocol_sha256=digest(protocol['parent_training_protocol']),
        parent_training_report_sha256=digest(protocol['parent_training_report']))
    write(name, protocol)
    install_stages(protocol)
    print(json.dumps({'status': 'frozen', 'protocol_sha256': digest(name),
        'source_count': len(sources), 'input_count': len(inputs), 'new_training_updates': 0, 'new_forward_calls': 0}))


if __name__ == '__main__':
    main()

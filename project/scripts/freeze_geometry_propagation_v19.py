"""Freeze synthetic geometry after component review, before production draws."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from scripts.pipeline import project_lock


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8-sig'))


def digest(name):
    h = hashlib.sha256()
    with (ROOT/name).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def write_text(name, text):
    file = ROOT/name; file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(file.name+'.preparation.tmp')
    assert not file.is_symlink() and not temporary.is_symlink()
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(text); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(file)


def write(name, value):
    write_text(name, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+'\n')


def install_stages(protocol):
    stages = read('configs/stages.json')
    inputs = sorted(set(['configs/geometry_propagation_v19.json']+list(protocol['source_hashes'])+list(protocol['input_hashes'])))
    out = protocol['outputs']; exe = protocol['interpreter']
    definitions = [
        ('P6R19', 'P4E18', '第十九轮合成三维传播72案例及实际分段恢复',
            'scripts/run_geometry_propagation_resume_v19.py',
            [out['root'], out['report'], out['report']+'.sha256.json', out['resume_report']]),
        ('P6E19', 'P6R19', '第十九轮合成传播独立数值验收与科学图',
            'scripts/accept_geometry_propagation_v19.py',
            [out['acceptance_report'], out['acceptance_report']+'.sha256.json', out['visuals_dir']])]
    for identifier, dependency, title, script, outputs in definitions:
        expected = dict(id=identifier, title=title, implemented=True, depends_on=[dependency],
            command=[exe, '-B', script], inputs=inputs, outputs=outputs)
        existing = next((row for row in stages['stages'] if row['id'] == identifier), None)
        if existing is None: stages['stages'].append(expected)
        else: assert existing == expected, 'Existing synthetic stage differs: '+identifier
    general = next(row for row in stages['stages'] if row['id'] == 'P6')
    if 'P6E19' not in general['depends_on']: general['depends_on'].append('P6E19')
    if stages != read('configs/stages.json'): write('configs/stages.json', stages)


def local_source_closure(files):
    result = set(files); pending = [name for name in files if name.endswith('.py')]
    def include(file):
        name = file.relative_to(ROOT).as_posix()
        if file.is_file() and name not in result:
            result.add(name); pending.append(name)
    while pending:
        file = pending.pop(); tree = ast.parse((ROOT/file).read_text(encoding='utf-8-sig'))
        package = Path(file).parent.parts
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.extend([alias.name, '.'.join(package+(alias.name,))])
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    assert node.level <= len(package), 'Relative import escapes project package: '+file
                    base = package[:len(package)-node.level+1]
                    prefix = '.'.join(base+tuple((node.module or '').split('.'))) if node.module else '.'.join(base)
                    prefixes = [prefix]
                elif node.module:
                    prefixes = [node.module, '.'.join(package+(node.module,))]
                else: continue
                for prefix in prefixes:
                    modules.append(prefix)
                    modules += [prefix+'.'+alias.name for alias in node.names if alias.name != '*']
        for module in modules:
            if module.split('.')[0] not in ('scripts', 'src', 'tests'): continue
            module_path = ROOT.joinpath(*module.split('.'))
            include(module_path.with_suffix('.py'))
            include(module_path/'__init__.py')
        for depth in range(1, len(package)+1):
            include(ROOT.joinpath(*package[:depth], '__init__.py'))
    return result


def freeze():
    name = 'configs/geometry_propagation_v19.json'; protocol = read(name)
    base = Path(protocol['base_python']['path'])
    assert base.is_file() and hashlib.sha256(base.read_bytes()).hexdigest() == protocol['base_python']['sha256'], 'Base interpreter identity changed'
    if protocol['status'] == 'frozen':
        for field in ('source_hashes', 'input_hashes'):
            for file, checksum in protocol[field].items(): assert digest(file) == checksum, file
        install_stages(protocol)
        print(json.dumps({'status': 'verified_frozen_stages_installed', 'protocol_sha256': digest(name)})); return
    assert protocol['status'] == 'unfrozen_preparation'
    assert not any((ROOT/protocol['outputs'][key]).exists()
        for key in ('root', 'report', 'resume_report', 'acceptance_report', 'visuals_dir', 'readonly_report'))
    log = 'logs/P6/geometry_propagation_v19/component_tests.log'
    text = (ROOT/log).read_text(encoding='utf-8')
    counts = list(map(int, re.findall(r'Ran (\d+) tests? in', text)))
    assert len(counts) == 1 and counts[0] >= 143 and '\nOK\n' in text and 'FAILED' not in text and 'skipped=' not in text
    readonly_log = 'logs/P6/geometry_propagation_v19/readonly_entry_tests.log'
    check = (ROOT/readonly_log).read_text(encoding='utf-8')
    assert re.search(r'Ran [1-9][0-9]* tests? in', check) and '\nOK\n' in check and 'FAILED' not in check and 'skipped=' not in check
    freeze_log = 'logs/P6/geometry_propagation_v19/freeze_entry_tests.log'
    check = (ROOT/freeze_log).read_text(encoding='utf-8')
    assert re.search(r'Ran [1-9][0-9]* tests? in', check) and '\nOK\n' in check and 'FAILED' not in check and 'skipped=' not in check
    sources = {
        'src/geometry_propagation_v19.py', 'tests/test_geometry_propagation_v19.py',
        'scripts/geometry_storage_v19.py', 'scripts/run_geometry_propagation_v19.py',
        'tests/test_geometry_propagation_storage_v19.py', 'scripts/run_geometry_propagation_resume_v19.py',
        'tests/test_geometry_propagation_wrapper_v19.py', 'scripts/accept_geometry_propagation_v19.py',
        'tests/test_geometry_propagation_acceptance_v19.py', 'scripts/verify_geometry_propagation_readonly_v19.py',
        'tests/test_geometry_propagation_readonly_v19.py', 'scripts/test_geometry_propagation_v19.py',
        'scripts/freeze_geometry_propagation_v19.py', 'scripts/resume_geometry_propagation_v19.ps1',
        'scripts/pipeline.py', 'scripts/multiview_common_v14r1.py',
        'docs/第十九轮三维传播合成验证草案.txt', log, readonly_log,
        'logs/P6/geometry_propagation_v19/core_tests.log', 'logs/P6/geometry_propagation_v19/wrapper_tests.log',
        'tests/test_geometry_propagation_freeze_v19.py', freeze_log}
    for file in sources: assert (ROOT/file).is_file(), file
    doc = 'docs/第十九轮三维传播合成验证冻结方案.txt'
    draft = (ROOT/'docs/第十九轮三维传播合成验证草案.txt').read_text(encoding='utf-8')
    assert '十一、实施前必须解决' in draft or '十、工程判定' in draft
    body = ('第十九轮三维传播合成验证冻结合同\n日期：2026年10月26日\n'
        '状态：实现、组件测试和独立审查已完成；正式样本尚未运行。\n'
        '本合同与配置、源码、测试和日志一起冻结。以下附录保留设计阶段原文及其边界；'
        '其中准备阶段的“拟定/未实施”描述记录设计历史，实际执行以本段声明及冻结配置为准。\n'
        '正式支持固定为72案例、每案例16批、每批8192个三元组。不得在观察生产结果后改变seed、N、统计容限或删去失败。\n'
        '两族alpha分别为0.01，不声称整体99%保证；线性支持和覆盖规则进入正式门槛，非线性及对角失配仅描述。\n'
        '组件统一测试数量为'+str(counts[0])+'；最终只读入口测试另有独立日志。小fixture通过不等于正式样本通过。\n'
        'P6R19为受限合成计算/恢复子阶段，P6E19为独立验收子阶段；二者都不使P6总阶段自动完成。\n'
        'P4E18仅作为顺序上的前置验收，不把其真实场景、模型或预测当作本轮数学输入。\n'
        '恢复入口为scripts/resume_geometry_propagation_v19.ps1；完成后再实际执行只读及冷重入并保存证据。\n'
        '\n设计附录（冻结前审查通过的原始范围）\n'+draft)
    if (ROOT/doc).exists(): assert (ROOT/doc).read_text(encoding='utf-8') == body, 'Prepared formal document differs'
    else: write_text(doc, body)
    sources.add(doc); sources = local_source_closure(sources)
    inputs = {'runtime/envs/grasp_eval/Scripts/python.exe', 'runtime/envs/grasp_eval/pyvenv.cfg'}
    fonts = ROOT/'runtime/envs/grasp_eval/Lib/site-packages/matplotlib/mpl-data/fonts/ttf'
    for file in fonts.glob('DejaVuSans*.ttf'): inputs.add(file.relative_to(ROOT).as_posix())
    protocol['source_hashes'] = {file: digest(file) for file in sorted(sources)}
    protocol['input_hashes'] = {file: digest(file) for file in sorted(inputs)}
    for path in ('logs/pipeline/P6R19', 'logs/pipeline/P6E19'):
        if path not in protocol['budget_paths']: protocol['budget_paths'].append(path)
    protocol['status'] = 'frozen'
    write(name, protocol); install_stages(protocol)
    print(json.dumps({'status': 'frozen', 'protocol_sha256': digest(name),
        'source_count': len(sources), 'input_count': len(inputs), 'production_samples_generated': 0}))


if __name__ == '__main__':
    with project_lock(ROOT): freeze()

"""Fetch only four frozen XML members using strictly bounded HTTP ranges."""
import argparse
import json
from pathlib import Path
import struct
import re
import sys
import zlib
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import multiview_common_v14r1 as common
from src.remote_zip import RangeReader
from scripts import multiview_common_v14 as parent_storage


def validate_xml(payload, entry):
    common.require(len(payload) == entry['bytes'] and (zlib.crc32(payload) & 0xffffffff) == entry['crc32'],
                   'Frozen XML length/CRC differs')
    decoded = payload.decode('utf-8-sig')
    common.require('\x00' not in decoded and '<!doctype' not in decoded.lower() and '<!entity' not in decoded.lower(),
                   'XML entity declarations or non-UTF8 encoding forbidden')
    node = ET.fromstring(decoded)
    common.require(node.tag == 'scene' and len(node.findall('obj')) > 0, 'Invalid scene XML structure')
    # Pose/object values are deliberately not inspected before generation seals.


def fetch_member(reader, entry):
    raw = reader.range(entry['header_offset'], 30, cache=False)
    header = struct.unpack('<4s5H3I2H', raw)
    signature, version, flags, method, mtime, mdate, crc, compressed, size, name_size, extra_size = header
    common.require(signature == b'PK\x03\x04' and not flags & 1 and method == entry['compress_type'] == 8,
                   'Unsupported ZIP local header')
    common.require(0 < name_size <= 1024 and extra_size <= 4096 and entry['compressed_bytes'] <= 4096
                   and entry['bytes'] <= 16384, 'ZIP header or payload exceeds bound')
    if not flags & 8:
        common.require((crc, compressed, size) == (entry['crc32'], entry['compressed_bytes'], entry['bytes']),
                       'ZIP local sizes/CRC differ')
    block = reader.range(entry['header_offset'] + 30, name_size + extra_size + entry['compressed_bytes'], cache=False)
    name = block[:name_size].decode('utf-8' if flags & 2048 else 'cp437')
    common.require(name == entry['source_member'], 'ZIP member name differs')
    inflater = zlib.decompressobj(-15)
    payload = inflater.decompress(block[name_size + extra_size:], entry['bytes'] + 1)
    common.require(inflater.eof and not inflater.unused_data and not inflater.unconsumed_tail,
                   'Invalid or oversized deflate stream')
    validate_xml(payload, entry)
    return payload


def run(root=ROOT, config=common.CONFIG, max_new_files=None, check_only=False):
    root, config = Path(root).resolve(), Path(config).resolve()
    protocol = common.read(config)
    common.require(protocol['frames'] == ['0016', '0080', '0160', '0240'] and bool(protocol['source_hashes']),
                   'Protocol not frozen')
    common.require(max_new_files is None or type(max_new_files) is int and max_new_files >= 0, 'Invalid file limit')
    guard = common.Guard(root, protocol, 'assets')
    binding = common.make_binding(root, config, protocol, 'asset_inputs', runtime_name='vision')
    directory = common.safe_path(root, protocol['outputs']['assets_dir'])
    with common.lock(root / 'runtime/grasp_multiview_v14r1/assets_lock'):
        guard.check()
        if (directory / 'completion.json').exists():
            common.verify_assets(root, config, protocol)
            return {'status': 'verified_completed_read_only', 'new_files': 0, 'network_bytes': 0, **guard.check()}
        common.require(not check_only, 'Assets are not complete')
        allowed = {'binding.json', 'report.json', 'report.json.sha256.json', 'completion.json', 'completion.json.sha256.json'}
        for frame in protocol['frames']:
            name = common.safe_path(root, protocol['frame_inputs'][frame]['annotation']).relative_to(directory).as_posix()
            allowed.update((name, name + '.sha256.json'))
        for existing in sorted(directory.rglob('*')) if directory.exists() else []:
            if not existing.is_file():
                continue
            name = existing.relative_to(directory).as_posix()
            temporary = re.fullmatch(r'(.+)\.[0-9a-f]{32}\.tmp', name)
            if temporary and temporary[1] in allowed:
                existing.unlink()  # Only this writer's uncommitted temporary bytes.
            else:
                common.require(name in allowed, 'Unexpected asset file: ' + name)
        common.once(directory / 'binding.json', binding)
        reader, created = None, 0
        source = protocol['asset_source']
        reuse = source.get('reuse_frozen_assets', False)
        if reuse:
            parent = protocol['recovery_sources']['parent_protocol']
            parent_config = common.safe_path(root, parent['path'])
            common.require(common.digest(parent_config) == parent['sha256'], 'Parent protocol changed')
            parent_protocol = common.read(parent_config)
            common.require(parent_protocol['outputs']['assets_dir'] == protocol['recovery_sources']['assets_dir'], 'Parent asset source differs')
            parent_files = parent_storage.verify_assets(root, parent_config, parent_protocol)
            common.require(all(protocol['asset_inputs'].get(name) == sha for name, sha in parent_files.items()), 'Parent assets are not pinned')
        for frame in protocol['frames']:
            entry = source['entries'][frame]
            path = common.safe_path(root, protocol['frame_inputs'][frame]['annotation'])
            common.require(path.is_relative_to(directory), 'Annotation escaped asset directory')
            if path.exists():
                common.verify_bytes(path, binding)
                validate_xml(path.read_bytes(), entry)
                continue
            if max_new_files is not None and created >= max_new_files:
                return {'status': 'partial', 'new_files': created, 'network_bytes': reader.network_bytes if reader else 0,
                        **guard.check()}
            guard.check(entry['bytes'] + 4096)
            if reuse:
                parent_path = common.safe_path(root, parent_protocol['frame_inputs'][frame]['annotation'])
                payload = parent_path.read_bytes()
                common.require(common.digest(parent_path) == protocol['asset_inputs'][parent_path.relative_to(root).as_posix()], 'Parent XML changed')
                validate_xml(payload, entry)
            else:
                if reader is None:
                    reader = RangeReader(source['base_url'] + '/' + source['archive'], timeout=30)
                    common.require(reader.size == source['archive_bytes'] and reader.etag == source['etag'],
                                   'Pinned remote archive length/ETag differs')
                payload = fetch_member(reader, entry)
                common.require(reader.network_bytes <= source['maximum_network_bytes'], 'Network payload budget exceeded')
            common.commit_bytes(path, payload, binding)
            created += 1
            guard.check()
        frames = {frame: {'path': protocol['frame_inputs'][frame]['annotation'],
                  'sha256': common.digest(common.safe_path(root, protocol['frame_inputs'][frame]['annotation'])),
                  'bytes': source['entries'][frame]['bytes'], 'crc32': source['entries'][frame]['crc32']}
                  for frame in protocol['frames']}
        report = {'schema': 14, 'status': 'verified_assets', 'frames': frames,
                  'source_revision': source['revision'], 'archive_bytes': source['archive_bytes'],
                  'archive_etag': source['etag'], 'new_models_downloaded': 0,
                  'asset_acquisition': {'mode': 'frozen_parent_copy' if reuse else 'bounded_http_range',
                      'source_dir': protocol['recovery_sources']['assets_dir'] if reuse else None},
                  'pose_values_decoded': False, 'official_whole_archive_checksum_verified': False}
        guard.check(len(common.json_bytes(report)) + 2048)
        common.commit(directory / 'report.json', report, binding)
        outputs = {common.safe_path(root, v['path']).relative_to(directory).as_posix(): v['sha256'] for v in frames.values()}
        outputs['report.json'] = common.digest(directory / 'report.json')
        completion = {'schema': 14, 'status': 'completed', 'outputs': outputs}
        guard.check(len(common.json_bytes(completion)) + 2048)
        common.commit(directory / 'completion.json', completion, binding)
        common.verify_assets(root, config, protocol)
        return {'status': 'completed', 'new_files': created, 'network_bytes': reader.network_bytes if reader else 0, **guard.check()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=common.CONFIG)
    parser.add_argument('--max-new-files', type=int)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(config=args.config, max_new_files=args.max_new_files, check_only=args.check_only)), flush=True)

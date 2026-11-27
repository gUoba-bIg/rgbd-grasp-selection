"""Read pinned ZIP indexes and quantify the exact missing P2/P3 input scope."""
import os,sys,json,re,time
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.remote_zip import RemoteZip
from scripts.p6_common_v24 import ROOT,read,once,stat,now
from scripts.pipeline import project_lock
LOG=ROOT/'logs/P2P3/prerequisites_v26';REV='9073d1d4cb6c92272b6aa072545def5101e4fecb'

def main():
 LOG.mkdir(parents=True,exist_ok=True);all_entries={};summaries={}
 archives=['train_1.zip','train_2.zip','train_3.zip','train_4.zip','grasp_label.zip','collision_label.zip','models.zip']
 for name in archives:
  p=LOG/'indexes'/(name+'.json')
  if p.exists():row=read(p)
  else:
   errors=[]
   for host in ['https://hf-mirror.com','https://huggingface.co']:
    url=f'{host}/saic3d/graspnet/resolve/{REV}/{name}'
    try:
     z=RemoteZip(url,ROOT/'data/downloads/prerequisites_v26_indexes')
     entries={n:dict(bytes=e.file_size,compressed=e.compress_size,crc32=e.CRC,offset=e.header_offset,method=e.compress_type) for n,e in z.entries.items() if not e.is_dir()}
     row=dict(at=now(),archive=name,url=url,size=z.reader.size,etag=z.reader.etag,network_bytes_known=z.reader.network_bytes,entries=entries,failed_sources=errors);once(p,row);break
    except Exception as exc:
     errors.append(dict(url=url,error=repr(exc)));once(LOG/'discovery_failures'/(name+'_'+str(len(errors))+'.json'),errors[-1])
   else:raise RuntimeError('No index source '+name)
  selected={}
  for n,e in row['entries'].items():
   if name.startswith('train_'):
    if re.search(r'^scene_\d{4}/object_id_list\.txt$',n) or '/realsense/' in n:selected[n]=e
   elif name=='models.zip':
    if n.endswith(('/nontextured.ply','/textured.obj','/textured.sdf')):selected[n]=e
   else:selected[n]=e
  summaries[name]=dict(selected_members=len(selected),compressed_bytes=sum(e['compressed'] for e in selected.values()),uncompressed_bytes=sum(e['bytes'] for e in selected.values()),largest_member=max(e['bytes'] for e in selected.values()),index=stat(p))
  all_entries[name]=selected
  print(name,json.dumps(summaries[name]),flush=True)
 once(LOG/'asset_inventory.json',dict(at=now(),selection='all100 original-train scenes, RealSense full256 views and geometry/labels; no official test content',archives=summaries,total_compressed_bytes=sum(v['compressed_bytes'] for v in summaries.values()),total_uncompressed_bytes=sum(v['uncompressed_bytes'] for v in summaries.values()),selected=all_entries))
if __name__=='__main__':
 with project_lock(ROOT):main()

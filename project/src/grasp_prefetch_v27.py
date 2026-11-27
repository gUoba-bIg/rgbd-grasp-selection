"""Two bounded CPU producers, preserving serial per-batch RNG and sample order."""
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from src.grasp_prerequisite_data_v27 import collate_fn

class TrainingBatches(Dataset):
 def __init__(self,frames,order,seed,epoch,start=0):self.frames=frames;self.order=order;self.seed=seed;self.epoch=epoch;self.start=start
 def __len__(self):return len(self.order)//2-self.start
 def __getitem__(self,index):
  batch=index+self.start;np.random.seed(self.seed+self.epoch*10000+batch)
  return batch,collate_fn([self.frames[i] for i in self.order[batch*2:batch*2+2]])

def worker_init(worker):torch.set_num_threads(1)

def loader(dataset):
 generator=torch.Generator();generator.manual_seed(dataset.seed+dataset.epoch)
 return DataLoader(dataset,batch_size=None,num_workers=2,prefetch_factor=1,pin_memory=True,worker_init_fn=worker_init,generator=generator,persistent_workers=False)

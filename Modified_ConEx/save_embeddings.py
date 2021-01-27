import numpy as np, pandas as pd, torch
def save_ConEx_emb(d, experiment, data_dir):
  emb = dict()
  for e in d.entities:
    emb[e] = np.array(experiment.Model.cpu().emb_e_real(torch.tensor(experiment.entity_idxs[e])).detach().tolist()+experiment.Model.cpu().emb_e_img(torch.tensor(experiment.entity_idxs[e])).detach().tolist())
  for r in d.relations:
    emb[r] = np.array(experiment.Model.cpu().emb_rel_real(torch.tensor(experiment.relation_idxs[r])).detach().tolist()+experiment.Model.cpu().emb_rel_img(torch.tensor(experiment.relation_idxs[r])).detach().tolist())
  pd.DataFrame(emb.values(), index=emb.keys()).to_csv("./"+("/").join(data_dir.split("/")[1:-2])+"/"+"ConEx_emb.csv")
  print("Finished saving ConEx embeddings for %s"%data_dir.split("/")[-3])
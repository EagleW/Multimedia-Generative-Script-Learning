from summac.model_summac import SummaCZS
import json
from tqdm import tqdm
from statistics import mean


model = SummaCZS(granularity="sentence", model_name="vitc")
cur_dir = 'wikihow_contrastive_crafts_stp/'
fname = cur_dir + 'submit.json'

with open(fname, 'r') as f:
    j = 0
    scores = []
    for line in tqdm(f):
        text = line.rstrip()
        
        data = json.loads(text)
        cands = data['pred']
        refs = data['ground']
        score = model.score_one(refs, cands)
        j += 1
        scores.append(score["score"])
print(mean(scores))
print()
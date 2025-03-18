import json
from tqdm import tqdm
from statistics import mean
from bart_score import BARTScorer


bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
bart_scorer.load(path='bart.pth')

cur_dir = '../wikihow_contrastive_crafts_stp/'
batch_size = 4

fname = cur_dir + 'submit.json'

with open(fname, 'r') as f:

    count = 0
    scores = []
    tgts = []
    srcs = []
    for line in tqdm(f):
        text = line.rstrip()
        
        data = json.loads(text)
        cand = data['pred']
        ref = data['ground']
        srcs.append(cand)
        tgts.append(ref)
    scores = bart_scorer.score(srcs, tgts, batch_size=batch_size)
print(mean(scores))
print()
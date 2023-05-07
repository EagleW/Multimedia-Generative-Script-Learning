
from torch.utils.data import DataLoader, Dataset
import json
import os
from tqdm import tqdm
import torch
from bleu.bleu import Bleu
from rouge.rouge import Rouge
from torch.utils.data.distributed import DistributedSampler
import random

# max history: 9; max title: 23; max step: 32, max caption 18, max target
def get_input(title, captions, steps, target, retrieves, tokenizer, title_neg, max_h=9, max_t=23, max_s=32, max_c=18, max_tgt=40, max_r=30):
    max_l = max(max_t, max_s, max_c)
    source = []
    source_id = []
    source_attention = []
    self_neg = title_neg

    new_captions = []
    for c in captions[-max_h - 1:]:
        if '<caption> ' not in c:
            new_captions.append('<caption> ' + c)
        else:
            new_captions.append(c)
    new_steps = []
    for s in step[-max_h - 1:]:
        if '<caption> ' not in s:
            new_steps.append('<caption> ' + s)
        else:
            new_steps.append(s)
    self_neg.extend(new_captions)
    self_neg.extend(new_steps)

    new_history = list(zip(new_captions, new_steps))
    
    tokenized = tokenizer('<cls> ' + title, padding='max_length', truncation=True, max_length=max_l + 1)
    source.append(title)

    source_id.extend(tokenized.input_ids[:-1])
    source_attention.extend(tokenized.attention_mask[:-1])

    for step, caption in new_history[:-1]:

        tokenized = tokenizer(step, padding='max_length', truncation=True, max_length=max_l)
        source.append(step)
        self_neg.append(step[len('<step> '):].strip())

        source_id.extend(tokenized.input_ids[1:-1])
        source_attention.extend(tokenized.attention_mask[1:-1])
        # print(len(source_id))

        tokenized = tokenizer(caption, padding='max_length', truncation=True, max_length=max_l)
        source.append(caption)
        self_neg.append(caption[len('<caption> '):].strip())

        source_id.extend(tokenized.input_ids[1:-1])
        source_attention.extend(tokenized.attention_mask[1:-1])
        # print(len(source_id))

    step, caption = new_history[-1]

    tokenized = tokenizer(step, padding='max_length', truncation=True, max_length=max_l)
    source.append(step)

    source_id.extend(tokenized.input_ids[1:-1])
    source_attention.extend(tokenized.attention_mask[1:-1])
    # print(len(source_id))
    self_neg.append(step[len('<step> '):].strip())

    tokenized = tokenizer(caption, padding='max_length', truncation=True, max_length=max_l)
    source.append(caption)
    self_neg.append(caption[len('<caption> '):].strip())

    source_id.extend(tokenized.input_ids[1:])
    source_attention.extend(tokenized.attention_mask[1:])
    # print(len(source_id))

    target_id = tokenizer.encode(target, max_length=max_tgt, truncation=True)
    hist_l = len(new_history)
    # input()

    retrieve_id = []
    retrieve_attention = []


    tokenized = tokenizer('<template> ' + retrieves[0], padding='max_length', truncation=True, max_length=max_r)
    retrieve_id.extend(tokenized.input_ids[:-1])
    retrieve_attention.extend(tokenized.attention_mask[:-1])
    for retrieve in retrieves[1:-1]:
        tokenized = tokenizer('<template> ' + retrieve, padding='max_length', truncation=True, max_length=max_r)
        retrieve_id.extend(tokenized.input_ids[1:-1])
        retrieve_attention.extend(tokenized.attention_mask[1:-1])

    tokenized = tokenizer('<template> ' + retrieves[-1], padding='max_length', truncation=True, max_length=max_r)
    retrieve_id.extend(tokenized.input_ids[1:])
    retrieve_attention.extend(tokenized.attention_mask[1:])

    source = ' '.join(source)
    return source, source_id, torch.LongTensor(source_attention), torch.LongTensor(target_id), hist_l, retrieve_id, torch.LongTensor(retrieve_attention), self_neg

class WikihowFineTuneDataset(Dataset):
    def __init__(self, split='train', topk=-1, args=None, tokenizer=None, 
                    mode='train', model='bart'):   
        super().__init__()

        self.topk = topk
        self.args = args
        self.neg_num_total = args.neg_num_total
        self.neg_num = args.neg_num

        self.dataset_dir = args.dataset_dir
        self.wikihow_dir = self.dataset_dir + '/'

        self.mode = mode
        self.split = split


        # Loading datasets to data
        self.tokenizer = tokenizer
        self.model = model


        self.data = []
        with open(self.wikihow_dir + split + '.json', 'r') as file_j:
            max_h=9
            max_t=23
            max_r=30
            max_s=32
            max_c=18
            max_tgt=40

            for line in tqdm(file_j, "Encoding Data"):
                record = json.loads(line)

                title = '<title> ' + record['title'].strip()
                method = record['method']
                title_neg = [record['title'].strip()]
                if len(method) > 0:
                    title += ' <method> ' + method.strip()
                    title_neg.append(method.strip())
                
                target = record['target'] 
                retrieve = record['retrieve'][:5]
                captions = record['captions'] 
                steps = record['steps']
                retrieve_neg = record['retrieve_neg']
                

                source, source_id, source_attention, target_id, hist_l, retrieve_id, retrieve_attention, self_neg = get_input(title, captions, steps, target, retrieve, self.tokenizer, title_neg, max_h=max_h, max_t=max_t, max_s=max_s, max_c=max_c, max_tgt=max_tgt, max_r=max_r)


                img_id = record['img']
                target_img = record['target_img']
                out_dict = {
                    'source': source,

                    'input_length': len(source_id),
                    'source_attention': source_attention,
                    'input_ids': torch.LongTensor(source_id),
                    'retrieve_ids': torch.LongTensor(retrieve_id),
                    'retrieve_attention': retrieve_attention,
                    'retrieve_length': len(retrieve_id),

                    'src_id': img_id,
                    'tgt_id':target_img,
                    'hist_l': hist_l,

                    'target_ids': target_id,
                    'target':target,
                    'target_length': len(target_id),

                    'retrieve_neg': retrieve_neg,
                    'self_neg': self_neg

                }

        
                self.data.append(out_dict)
                if self.topk != -1 and len(self.data) > self.topk:
                    break


        print("# all sentences:", len(self.data))
        self.max_tgt = max_tgt



    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        datum = self.data[idx]

        negs = []
        tmp = random.sample(datum['self_neg'], k= min(self.neg_num,len(datum['self_neg'])))
        negs.extend(tmp)

        cur_neg = len(tmp)
        s_negs = random.sample(datum['retrieve_neg'], k=self.neg_num_total -  cur_neg)

        negs.extend((s_negs))
        neg_length = []
        neg_ids = []

        for neg in negs:
            neg_id = self.tokenizer.encode(neg, max_length=self.max_tgt, truncation=True)
            neg_length.append(len(neg_id))
            neg_ids.append(torch.LongTensor(neg_id))
        
        datum['neg_length'] = neg_length
        datum['neg_ids'] = neg_ids

        return datum


    def collate_fn(self, batch):
        batch_entry = {}


        B = len(batch)

        S_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_L, dtype=torch.long) * self.tokenizer.pad_token_id
        attention_masks = torch.zeros(B, S_L, dtype=torch.long) * self.tokenizer.pad_token_id

        R_L = max(entry['retrieve_length'] for entry in batch)
        retrieve_ids = torch.ones(B, R_L, dtype=torch.long) * self.tokenizer.pad_token_id
        retrieve_attention_mask = torch.zeros(B, R_L, dtype=torch.long) * self.tokenizer.pad_token_id

        T_L = max(entry['target_length'] for entry in batch)
        target_ids = torch.ones(B, T_L, dtype=torch.long) * self.tokenizer.pad_token_id

        N_L = max(neg_length for entry in batch for neg_length in entry['neg_length'])
        neg_ids = torch.ones(B * self.neg_num_total, N_L, dtype=torch.long) * self.tokenizer.pad_token_id



        src_ids = []
        targets = []
        sources= []
        target_lengths= []
        hist_l = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']
            attention_masks[i, :entry['input_length']] = entry['source_attention']

            retrieve_ids[i, :entry['retrieve_length']] = entry['retrieve_ids']
            retrieve_attention_mask[i, :entry['retrieve_length']] = entry['retrieve_attention']

            for j in range(self.neg_num_total):
                index = i  * self.neg_num_total + j
                neg_ids[index, :entry['neg_length'][j]] = entry['neg_ids'][j]

            src_ids.append(entry['src_id'])
            sources.append(entry['source'])
            targets.append(entry['target'])
            target_lengths.append(entry['target_length'])
            hist_l.append(entry['hist_l'])

        batch_entry['input_ids'] = input_ids
        batch_entry['attention_masks'] = attention_masks

        batch_entry['retrieve_ids'] = retrieve_ids
        batch_entry['retrieve_attention_mask'] = retrieve_attention_mask

        batch_entry['neg_ids'] = neg_ids

        batch_entry['target_length'] = target_lengths
        batch_entry['max_hist'] = max(hist_l)
        batch_entry['hist_l'] = torch.LongTensor(hist_l)

        batch_entry['src_ids'] = src_ids
        batch_entry['sources'] = sources
        batch_entry['targets'] = targets
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100
        batch_entry['target_ids'] = target_ids
        batch_entry['neg_num_total'] = self.neg_num_total

        return batch_entry


def get_loader(args, split='train', mode='train', tokenizer=None,
               batch_size=32, workers=4, topk=-1, model='bart'):


    dataset = WikihowFineTuneDataset(
        split,
        topk=topk,
        args=args,
        tokenizer=tokenizer,
        mode=mode,
        model=model)
    
    if args.distributed:
        if mode == 'train':
            sampler = DistributedSampler(dataset)
        else:
            sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=(sampler is None),
            num_workers=workers, 
            pin_memory=True, 
            sampler=sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, 
            pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)

    loader.evaluator = WikihowEvaluator()


    return loader, sampler


class WikihowEvaluator:
    def __init__(self):
        self.scorers = [
            (Bleu(4),  ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L")
            ]

    

    def score(self, ref, hypo):
        final_scores = {}
        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score

        return final_scores
    def evaluate(self, quesid2ans):
        hypo = {}
        ref = {}
        i = 0
        for k in quesid2ans:
            ans, tgt = quesid2ans[k]
            hypo[i] = [ans]
            ref[i] = [tgt]
            i += 1

        score = self.score(ref, hypo)
        print(score)
        
        return {'score':2*score['ROUGE_L']*score['Bleu_4']/(score['Bleu_4']+ score['ROUGE_L']), 'bleu':score['Bleu_4'], 'rogue':score['ROUGE_L']}

    def dump_result(self, quesid2ans: dict, path):

        with open(path, 'w') as f:
            for k in quesid2ans:
                ans, tgt = quesid2ans[k]
                result = {'img_id':k, 'pred':ans, 'ground': tgt}
                f.write(json.dumps(result) + '\n')
        
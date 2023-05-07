import torch
from modules.modeling_bart import ContrastiveBartForConditionalGeneration

class BartWikiHow(ContrastiveBartForConditionalGeneration):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)


    def train_step(self, batch):

        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_masks'].to(device)
        max_hist = batch['max_hist']
        hist_l = batch['hist_l']
        retrieve_ids = batch['retrieve_ids'].to(device)
        retrieve_attention_mask = batch['retrieve_attention_mask'].to(device)
        neg_ids = batch['neg_ids'].to(device)

        lm_labels = batch["target_ids"].to(device)
        neg_num_total = batch["neg_num_total"]

        output = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            hist_l=hist_l,
            max_hist=max_hist,
            retrieve_ids=retrieve_ids,
            retrieve_attention_mask=retrieve_attention_mask,
            neg_ids=neg_ids,
            neg_num_total=neg_num_total,
            labels=lm_labels,
        )

        loss = output['loss']        
        cl_loss = output['cl_loss']

        result = {
            'loss': loss + cl_loss * 0.5,
            'cl_loss': cl_loss,
            'lm_loss': loss
        }

        return result

    @torch.no_grad()
    def test_step(self, batch, **kwargs):
        self.eval()
        device = next(self.parameters()).device
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_masks'].to(device)
        max_hist = batch['max_hist']
        hist_l = batch['hist_l']
        retrieve_ids = batch['retrieve_ids'].to(device)
        retrieve_attention_mask = batch['retrieve_attention_mask'].to(device)

        result = {}

        output = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            hist_l=hist_l,
            max_hist=max_hist,
            retrieve_ids=retrieve_ids,
            retrieve_attention_mask=retrieve_attention_mask,
            **kwargs
        )
        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        result['token_ids'] = output
        result['pred_ans'] = generated_sents

        return result


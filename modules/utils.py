import torch

def AttnClsGate(
    max_l, 
    mask_id, 
    device, 
    hidden_states,
    modified_attention,
    hist_l,
    embed_tokens, 
    gatew,
    cls,
    seq_l,
    attention_cls
    ):
    
    mask_token_id = torch.LongTensor([[mask_id]]).to(device)
    mask_e = embed_tokens(mask_token_id)
    concentrate_part = torch.clone(hidden_states[:,1:-1,:])

    modified_attention[:,(hist_l * 2 + 1) * max_l] = 0
    modified_attention = modified_attention[:,:-1].unsqueeze(-1)
    modified_attention_inverse = torch.ones_like(modified_attention).to(device) - modified_attention
    batch_size, _, hidden_dim = concentrate_part.shape
    modified_part = concentrate_part[:,1:,:].reshape(-1, max_l, hidden_dim)


    encoder_cls = torch.repeat_interleave(cls, seq_l, dim=0).unsqueeze(1)

    encoder_context, _ = attention_cls(encoder_cls, modified_part, modified_part)
    encoder_context = modified_part[:,0,:].unsqueeze(1)


    coef = torch.sigmoid(gatew(torch.cat([encoder_cls, encoder_context], dim=-1)))

    modified_update = coef * mask_e.expand(-1,max_l,-1) + (1 - coef) * modified_part
    modified_update = modified_update.view(batch_size, -1, hidden_dim)

    hidden_states[:,2:-1,:] = modified_update * modified_attention + hidden_states[:,2:-1,:] * modified_attention_inverse

    return hidden_states, batch_size, hidden_dim, mask_e




def AttnGate(
    cls,
    batch_size,
    hidden_dim,
    mask_e,
    max_l, 
    device, 
    hidden_states,
    modified_attention,
    gatew,
    attention_cls
):
    modified_attention = modified_attention.unsqueeze(-1)
    modified_attention_inverse = torch.ones_like(modified_attention).to(device) - modified_attention

    modified_part = torch.clone(hidden_states[:,1:-1,:]).reshape(-1, max_l, hidden_dim)
    encoder_cls = torch.repeat_interleave(cls, 5, dim=0).unsqueeze(1)
    encoder_context, _ = attention_cls(encoder_cls, modified_part, modified_part)
    encoder_context = modified_part[:,0,:].unsqueeze(1)


    coef = torch.sigmoid(gatew(torch.cat([encoder_cls, encoder_context], dim=-1)))
    modified_update = coef * mask_e.expand(-1,max_l,-1) + (1 - coef) * modified_part
    modified_update = modified_update.view(batch_size, -1, hidden_dim)

    hidden_states[:,1:-1,:] = modified_update * modified_attention + hidden_states[:,1:-1,:] * modified_attention_inverse

    return hidden_states
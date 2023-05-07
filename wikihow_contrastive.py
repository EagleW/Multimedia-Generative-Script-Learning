import torch.backends.cudnn as cudnn
import os
from pathlib import Path

from transformers import BartTokenizer
from tqdm import tqdm
import torch
import logging
import torch.distributed as dist
from torch.distributed import ReduceOp
from torch.nn.parallel import DistributedDataParallel as DDP

from param import parse_args

from wikihow_contrastive_data import get_loader
from utils_ import LossMeter, set_global_logging_level, reduce_dict
import wandb

set_global_logging_level(logging.ERROR, ["transformers"])

proj_dir = Path(__file__).resolve().parent.parent

from torch.cuda.amp import autocast

from trainer_base import TrainerBase
from wikihow_contrastive_model import BartWikiHow

class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, tokenizer=None, num_added_toks=0, sampler=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            tokenizer=tokenizer,
            num_added_toks=num_added_toks,
            train=train)

        model_kwargs = {}
        self.sampler = sampler
        
        config = self.create_config()

        model_class = BartWikiHow
        self.model = self.create_model(model_class, config, **model_kwargs)

        self.model.resize_token_embeddings(self.model.model.shared.num_embeddings + self.num_added_toks)

        self.model.tokenizer = self.tokenizer

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            ckpt_path = args.load
            self.load_checkpoint(ckpt_path)
            print('Load pretrained model')
        
        # GPU Options
        if self.verbose:
            from time import time
            start = time()
        print(f'Model Launching at GPU {self.args.gpu}')
        self.model = self.model.to(args.gpu)
        if args.distributed:
            self.model = DDP(self.model, device_ids=[args.gpu],
                                output_device=args.gpu
                                )

        # Optimizer
        if train:
            if self.args.fp16:
                print('Run in half precision')
                self.scaler = torch.cuda.amp.GradScaler()
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()
        
        print(self.model.num_parameters())

        if self.verbose:
            print(f'It took {time() - start:.1f}s')
            if args.wandb:
                wandb.watch(self.model)


    def train(self):
        if self.verbose:
            # loss_meter = LossMeter()
            LOSSES_NAME = ['loss', 'cl_loss', 'lm_loss']
            loss_meters = [LossMeter() for _ in range(3)]

            best_valid = 0.
            best_epoch = 0

        if self.args.distributed:
            dist.barrier()
        global_step = 0
        update_epoch = 0

        for epoch in range(self.args.epochs):
            flag_tensor = torch.zeros(1).to(self.model.device)
            if self.args.distributed:
                self.sampler.set_epoch(epoch)

            if self.start_epoch is not None:
                epoch += self.start_epoch
            self.model.train()
            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=150)

            epoch_results = {
                'loss': 0.,
                'cl_loss': 0.,
                'lm_loss': 0.,

            }

            for step_i, batch in enumerate(self.train_loader):
                self.model.zero_grad(set_to_none=True)

                if self.args.fp16:
                    with autocast():
                        if self.args.distributed:
                            results = self.model.module.train_step(batch)
                        else:
                            results = self.model.train_step(batch)

                        loss = results['loss']
                        self.scaler.scale(loss).backward()
                else:
                    if self.args.distributed:
                        results = self.model.module.train_step(batch)
                    else:
                        results = self.model.train_step(batch)
                    loss = results['loss']
                    loss.backward()
                loss = loss.detach()

                # Update Parameters
                if self.args.clip_grad_norm > 0:
                    if self.args.fp16:
                        # https://github.com/openai/CLIP/issues/83
                        # https://github.com/openai/CLIP/issues/57

                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

                        self.scaler.step(self.optim)

                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.clip_grad_norm)
                        self.optim.step()
                else:

                    if self.args.fp16:
                        self.scaler.step(self.optim)
                        self.scaler.update()
                    else:
                        self.optim.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step()
                for param in self.model.parameters():
                    param.grad = None

                global_step += 1

                for k, v in results.items():
                    epoch_results[k] += v.item()

                lr=self.optim.param_groups[0]["lr"] 

                if self.verbose:
                    desc_str = f'Epoch {epoch} | LR {lr:.10f}'

                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):
                            
                        loss_meter.update(results[f'{loss_name}'].item())
                        desc_str += f' {loss_name} {loss_meter.val:.6f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)


            if self.verbose:
                pbar.close()
            if self.args.distributed:
                dist.barrier()
                epoch_results = reduce_dict(epoch_results)

            # Validation
            score_dict = self.evaluate(self.val_loader)
            if self.args.distributed:
                dist.barrier()
                score_dict = reduce_dict(score_dict)

            if self.verbose:
                valid_score = score_dict['score'] * 100
                if valid_score > best_valid or epoch == 0:
                    best_valid = valid_score
                    best_epoch = epoch
                    self.save("BEST")
                    update_epoch  = epoch

                log_str = ''
                log_str += "\nEpoch %d: Best Score %0.2f\n" % (best_epoch, best_valid)

                wandb_log_dict = {}
                len_train_loader = len(self.train_loader)
                wandb_log_dict['Train/Loss'] = epoch_results['loss'] / len_train_loader
                wandb_log_dict['Train/CL_Loss'] = epoch_results['cl_loss'] / len_train_loader
                wandb_log_dict['Train/LM_Loss'] = epoch_results['lm_loss'] / len_train_loader

                wandb_log_dict['Valid/bleu'] = score_dict['bleu']
                wandb_log_dict['Valid/rogue'] = score_dict['rogue']
                wandb_log_dict['Valid/score'] = score_dict['score']

                if self.args.wandb:
                    wandb.log(wandb_log_dict, step=epoch)
                print("\nEpoch %d: Acc %0.4f bleu %0.4f rogue %0.4f loss %0.4f cl_loss %0.4f lm_loss %0.4f \n" % (epoch, valid_score, score_dict['bleu'], score_dict['rogue'], wandb_log_dict['Train/Loss'], wandb_log_dict['Train/CL_Loss'], wandb_log_dict['Train/LM_Loss']))
                print(log_str)
                print()
            if self.args.distributed:
                dist.barrier()
        
                if self.verbose:
                    if epoch - update_epoch > self.args.patient:
                        flag_tensor += 1
                dist.all_reduce(flag_tensor,op=ReduceOp.SUM)
                if flag_tensor > 0:
                    break
            else:
                if epoch - update_epoch > self.args.patient:
                    break

        if self.args.distributed:
            dist.barrier()

        if self.verbose:
            self.save("LAST")
            # Test Set
            best_path = os.path.join(self.args.output, 'BEST')
            self.load(best_path)

            quesid2ans = self.predict(self.test_loader)
            evaluator = self.test_loader.evaluator
            score_dict = evaluator.evaluate(quesid2ans)
            wandb_log_dict = {}
            wandb_log_dict['Test/bleu'] = score_dict['bleu']
            wandb_log_dict['Test/rogue'] = score_dict['rogue']
            evaluator.dump_result(quesid2ans, self.args.output + '/submit.json')

            print(wandb_log_dict)
            if self.args.wandb:
                wandb.log(wandb_log_dict)
                wandb.log({'finished': True})

            print('save prediction file')

        if self.args.distributed:
            dist.barrier()
            exit()

    def predict(self, loader, dump_path=None):
        if not os.path.isdir(self.args.output):
            os.makedirs(self.args.output, exist_ok=True)
        self.model.eval()
        with torch.no_grad():
            quesid2ans = {}
            gen_kwargs = {}
            if self.args.num_beams > 1:
                gen_kwargs['num_beams'] = self.args.num_beams
            if self.verbose:
                pbar = tqdm(total=len(loader), ncols=120, desc="Prediction")
            for i, batch in enumerate(loader):
                if self.args.distributed:
                    results = self.model.module.test_step(batch, **gen_kwargs)
                else:
                    results = self.model.test_step(batch, **gen_kwargs)

                pred_ans = results['pred_ans']
                ques_ids = batch['src_ids']
                tgt = batch['targets']

                for qid, ans, tgt in zip(ques_ids, pred_ans, tgt):
                    quesid2ans[qid] = (ans, tgt)

                if self.verbose:
                    pbar.update(1)

            if self.verbose:
                pbar.close()
                print('\n sample: '+ qid + '\n ans: ' + ans + '\n tgt: '+ tgt + '\n')

            if self.verbose:
                if dump_path is not None:
                    evaluator = loader.evaluator
                    evaluator.dump_result(quesid2ans, dump_path)
            if self.args.distributed:
                dist.barrier()


        return quesid2ans

    def evaluate(self, loader, dump_path=None):
        evaluator = loader.evaluator
        quesid2ans = self.predict(loader, dump_path)

        topk_score = evaluator.evaluate(quesid2ans)

        return topk_score

    def save(self, name):
        if not os.path.isdir(self.args.output):
            os.makedirs(self.args.output, exist_ok=True)
        if self.args.distributed:     
            torch.save(self.model.module.state_dict(),
                    os.path.join(self.args.output, "%s.pth" % name))
        else:
            torch.save(self.model.state_dict(),
                    os.path.join(self.args.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        if self.args.distributed:     
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)


def main_worker(gpu, args):
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl',
                                     init_method='env://', rank=args.rank, world_size=args.world_size)

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    num_added_toks = 0
    
    additional_special_tokens = ['<method>', '<title>', '<step>', '<caption>', '<cls>', '<template>']

    special_tokens_dict = {
        'additional_special_tokens': additional_special_tokens, 
        'mask_token': '[MASK]'
        }
    
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)


    if args.test_only:
    
        print(f'Building submit test loader at GPU {gpu}')

        split = f'submit_{gpu}'
        print('Loading', split)

        test_loader, sampler = get_loader(
            args,
            split='test', 
            mode='test', 
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            workers=4,
            topk=args.valid_topk,
            model=args.model
        )
        train_loader = None
        val_loader = None

        trainer = Trainer(args, train_loader, val_loader, test_loader, tokenizer, num_added_toks, train=False)
        dump_path = os.path.join(args.output, f'submit.json')
        trainer.predict(test_loader, dump_path=dump_path)

    else:

        print(f'Building train loader at GPU {gpu}')
        train_loader, sampler = get_loader(
            args,
            split='train', 
            mode='train', 
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            workers=4,
            topk=args.train_topk,
            model=args.model
        )

        if args.valid_batch_size is not None:
            valid_batch_size = args.valid_batch_size
        else:
            valid_batch_size = args.batch_size
        print(f'Building val loader at GPU {gpu}')
        val_loader, sampler = get_loader(
            args,
            split='valid', 
            mode='val', 
            tokenizer=tokenizer,
            batch_size=valid_batch_size,
            workers=4,
            topk=args.valid_topk,
            model=args.model
        )

        print(f'Building test loader at GPU {gpu}')
        test_loader, sampler = get_loader(
            args,
            split='test', 
            mode='test', 
            tokenizer=tokenizer,
            batch_size=valid_batch_size,
            workers=4,
            topk=args.valid_topk,
            model=args.model
        )

        trainer = Trainer(args, train_loader, val_loader, test_loader, tokenizer, num_added_toks, sampler, train=True)

        trainer.train()

if __name__ == "__main__":
    args = parse_args()
    if torch.cuda.is_available() and args.distributed:
        ngpus_per_node = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        cudnn.benchmark = True
        args.distributed = args.distributed and ngpus_per_node>1
        args.world_size =  ngpus_per_node
        args.rank = int(os.environ["RANK"])
    else:
        args.world_size = 0
        args.local_rank = -1
        args.distributed = False
    project_name = "Wikihow_contrastive"

    if args.local_rank in [0, -1]:
        comments = []
        if args.load is not None:
            ckpt_str = "_".join(args.load.split('/')[-3:])
            comments.append(ckpt_str)
        comment = '_'.join(comments)

        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M')

        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'

        args.run_name = run_name

        if args.wandb:
            wandb.init(project=project_name,  resume="allow")
            wandb.config.update(args)
            config = wandb.config
        else:
            config=args

    if args.distributed:
        main_worker(args.local_rank, args)
    else:
        main_worker(0, args)

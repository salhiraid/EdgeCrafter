import datetime
import json
import logging
import time

import torch

from ..misc import dist_utils, stats
from ..misc.metrics import BestMetricHolder
from ..optim.lr_scheduler import FlatCosineLRScheduler
from ._solver import BaseSolver
from .pose_engine import evaluate, train_one_epoch


def safe_barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    else:
        pass

def safe_get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0

class PoseSolver(BaseSolver):
    def _build_training_logger(self):
        logger = logging.getLogger(f'ecpose.training.rank{safe_get_rank()}')
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers and self.output_dir and dist_utils.is_main_process():
            handler = logging.FileHandler(self.output_dir / 'training.log')
            handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
            logger.addHandler(handler)
        return logger

    def train(self,):
        self._setup()
        self.criterion = self.cfg.criterion
        self.optimizer = self.cfg.optimizer
        self.lr_scheduler = self.cfg.lr_scheduler
        self.lr_warmup_scheduler = self.cfg.lr_warmup_scheduler

        # Load datasets
        self.train_dataloader = dist_utils.warp_loader(
            self.cfg.train_dataloader, shuffle=self.cfg.train_dataloader.shuffle
        )
        self.val_dataloader = dist_utils.warp_loader(
            self.cfg.val_dataloader, shuffle=self.cfg.val_dataloader.shuffle
        )

        self.evaluator = self.cfg.evaluator

        # Enable self-defined flat-cosine scheduler for pose if requested
        self.self_lr_scheduler = False
        if hasattr(self.cfg, "lrsheduler") and self.cfg.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} (pose) ## ".format(self.cfg.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(
                self.optimizer,
                self.cfg.lr_gamma,
                iter_per_epoch,
                total_epochs=self.cfg.epoches,
                warmup_iter=self.cfg.warmup_iter,
                flat_epochs=self.cfg.flat_epoch,
                no_aug_epochs=self.cfg.no_aug_epoch,
            )
            self.self_lr_scheduler = True

        self.best_map_holder = BestMetricHolder(use_ema=self.cfg.use_ema)
        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.load_resume_state(self.cfg.resume)

    def fit(self,):
        self.train()
        args = self.cfg
        training_logger = self._build_training_logger()
        n_parameters, model_stats = stats(self.cfg)
        
        print(model_stats)
        # print("-" * 42 + "Model Structrue" + "-" * 43)
        # print(self.model)
        
        # print("-" * 42 + "Check Shape of feats" + "-" * 43)
        # model = self.model.module if hasattr(self.model, 'module') else self.model
        # device = next(model.parameters()).device  
        # with torch.no_grad():
        #     feats = model.backbone(torch.randn(1, 3, 640, 640).to(device))
        #     for i, f in enumerate(feats):
        #         print(i, f.shape)

        print("-" * 42 + "Start training" + "-" * 43)
        
        
        best_scores = {}
        best_epochs = {}
        # evaluate again before resume training
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats = evaluate(
                module,
                self.postprocessor,
                self.evaluator,
                self.val_dataloader,
                self.device
            )
            for name, values in test_stats.items():
                if values:
                    best_scores[name] = values[0]
                    best_epochs[name] = self.last_epoch
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):
            epoch_start_time = time.time()

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_avail_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.cfg.train_dataloader.batch_size,
                args.grad_accum_steps,
                self.device, 
                epoch,
                args.clip_max_norm, 
                writer=self.writer, 
                warmup_scheduler=self.lr_warmup_scheduler, 
                ema=self.ema,
                args=args
                )

            if not self.self_lr_scheduler:
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    # weights = {
                    #     'model': self.state_dict(),
                    #     'ema': self.ema.state_dict() if self.ema is not None else None,
                    #     'optimizer': self.optimizer.state_dict(),
                    #     'lr_scheduler': self.lr_scheduler.state_dict(),
                    #     'warmup_scheduler': self.lr_warmup_scheduler.state_dict() if self.lr_warmup_scheduler is not None else None,
                    #     'epoch': epoch,
                    #     'args': args,
                    # }
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)


            
            # eval ema model if exists
            if self.ema is not None:
                test_stats = evaluate(
                    self.ema.module, 
                    self.postprocessor, 
                    self.evaluator,
                    self.val_dataloader, 
                    self.device, 
                    self.writer
                )
                for k in test_stats:
                    if self.writer and dist_utils.is_main_process():
                        for i, v in enumerate(test_stats[k]):
                            self.writer.add_scalar(f'Test/ema_{k}_{i}'.format(k), v, epoch)
                eval_stats = test_stats
            else:
                # eval regular model
                test_stats = evaluate(
                    self.model, 
                    self.postprocessor, 
                    self.evaluator,
                    self.val_dataloader, 
                    self.device, 
                    self.writer
                )
                # Log regular model results
                for k in test_stats:
                    if self.writer and dist_utils.is_main_process():
                        for i, v in enumerate(test_stats[k]):
                            self.writer.add_scalar(f'Test/regular_{k}_{i}'.format(k), v, epoch)
                eval_stats = test_stats
            
            improved_metrics = set()
            for name, values in eval_stats.items():
                if not values:
                    continue
                score = values[0]
                if score > best_scores.get(name, float('-inf')):
                    best_scores[name] = score
                    best_epochs[name] = epoch
                    improved_metrics.add(name)

            primary_metric = (
                'coco_eval_keypoints' if 'coco_eval_keypoints' in eval_stats
                else 'coco_eval_bbox'
            )
            if self.output_dir and primary_metric in improved_metrics:
                best_name = (
                    'best_stg2.pth'
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch
                    else 'best_stg1.pth'
                )
                dist_utils.save_on_master(self.state_dict(), self.output_dir / best_name)

            best_summary = {
                name: {'score': score, 'epoch': best_epochs[name]}
                for name, score in best_scores.items()
            }
            print(f'best_stats: {best_summary}')


            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in eval_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }
            
            # Add EMA results to log if available
            if self.ema is not None:
                log_stats.update({f'test_{k}': v for k, v in test_stats.items()})

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                training_logger.info(json.dumps(log_stats, sort_keys=True))

            if self.writer and dist_utils.is_main_process():
                for name, value in train_stats.items():
                    self.writer.add_scalar(f'Epoch/train_{name}', value, epoch)
                self.writer.flush()
                      
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def val(self, ):
        self.eval()
        module = self.ema.module if self.ema else self.model
        test_stats = evaluate(
                module,
                self.postprocessor,
                self.evaluator,
                self.val_dataloader,
                self.device,
            )

        # if self.output_dir:
        #     dist_utils.save_on_master(coco_evaluator.coco_eval["keypoints"].eval, self.output_dir / "eval.pth")

        return

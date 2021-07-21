import utils
import argparser
import os
from utils.logger import Logger

# from torch.nn.parallel import DistributedDataParallel
import torch.cuda.amp as amp
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import random
import torch
from torch.utils import data
from torch import distributed

from dataset import VOCSegmentationIncrementalWeak
from dataset import transform
from metrics import StreamSegMetrics

from segmentation_module import make_model

from train import Trainer
import tasks


def save_ckpt(path, model, trainer, optimizer, scheduler, epoch, best_score):
    """ save current model
    """
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": best_score,
        "trainer_state": trainer.state_dict()
    }
    torch.save(state, path)


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    train_transform = transform.Compose([
        transform.RandomResizedCrop(opts.crop_size, (0.5, 2.0)),
        # transform.RandomHorizontalFlip(), no augmentation if weak supervision
        # transform.RandomRotation(15),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    if opts.crop_val:
        val_transform = transform.Compose([
            transform.PadCenterCrop(512), # as in BiSeNet code
            # transform.Resize(size=opts.crop_size),
            # transform.CenterCrop(size=opts.crop_size),
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    else:
        # no crop, batch size = 1
        val_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

    labels, labels_old, path_base = tasks.get_task_labels(opts.dataset, opts.task, opts.step)
    labels_cum = labels_old + labels

    if opts.dataset == 'voc':
        dataset = VOCSegmentationIncrementalWeak
    elif opts.dataset == 'ade':
        raise NotImplementedError
    else:
        raise NotImplementedError

    if opts.overlap:
        path_base += "-ov"

    if not os.path.exists(path_base):
        os.makedirs(path_base, exist_ok=True)

    train_dst = dataset(root=opts.data_root, train=True, transform=train_transform,
                        labels=list(labels), labels_old=list(labels_old),
                        idxs_path=path_base + f"/train-{opts.step}.npy",
                        masking=not opts.no_mask, overlap=opts.overlap)

    if not opts.no_cross_val:  # if opts.cross_val:
        train_len = int(0.8 * len(train_dst))
        val_len = len(train_dst)-train_len
        train_dst, val_dst = torch.utils.data.random_split(train_dst, [train_len, val_len])
    else:  # don't use cross_val
        val_dst = dataset(root=opts.data_root, train=False, transform=val_transform,
                          labels=list(labels), labels_old=list(labels_old),
                          idxs_path=path_base + f"/val-{opts.step}.npy",
                          masking=not opts.no_mask, overlap=True)

    image_set = 'train' if opts.val_on_trainset else 'val'
    test_dst = dataset(root=opts.data_root, train=opts.val_on_trainset, transform=val_transform,
                       labels=list(labels_cum),
                       idxs_path=path_base + f"/test_on_{image_set}-{opts.step}.npy")

    return train_dst, val_dst, test_dst, len(labels_cum)


def main(opts):
    distributed.init_process_group(backend='nccl', init_method='env://')
    device_id, device = opts.local_rank, torch.device(opts.local_rank)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(device_id)

    # Initialize logging
    task_name = f"{opts.task}-{opts.dataset}"
    logdir_full = f"{opts.logdir}/{task_name}/{opts.name}/"
    if rank == 0:
        logger = Logger(logdir_full, rank=rank, debug=opts.debug, summary=opts.visualize, step=opts.step)
    else:
        logger = Logger(logdir_full, rank=rank, debug=opts.debug, summary=False)

    logger.print(f"Device: {device}")

    # Set up random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # xxx Set up dataloader
    train_dst, val_dst, test_dst, n_classes = get_dataset(opts)
    
    # reset the seed, this revert changes in random seed
    random.seed(opts.random_seed)

    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size,
                                   sampler=DistributedSampler(train_dst, num_replicas=world_size, rank=rank),
                                   num_workers=opts.num_workers, drop_last=True)
    val_loader = data.DataLoader(val_dst, batch_size=opts.batch_size if opts.crop_val else 1,
                                 sampler=DistributedSampler(val_dst, num_replicas=world_size, rank=rank),
                                 num_workers=opts.num_workers)
    logger.info(f"Dataset: {opts.dataset}, Train set: {len(train_dst)}, Val set: {len(val_dst)},"
                f" Test set: {len(test_dst)}, n_classes {n_classes}")
    logger.info(f"Total batch size is {opts.batch_size * world_size}")

    # xxx Set up model
    logger.info(f"Context Path Backbone: {opts.context_path}")
    logger.info(f"Method: {opts.method}")

    step_checkpoint = None
    model = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))

    if opts.step == 0:  # if step 0, we don't need to instance the model_old
        model_old = None
    else:  # instance model_old
        model_old = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step - 1), is_old=True)

    if opts.fix_bn:
        model.fix_bn()

    logger.debug(model)

    # xxx Set up optimizer
    params = []

    params.append({"params": filter(lambda p: p.requires_grad, model.bisenet.parameters()),
                   'weight_decay': opts.weight_decay})
    params.append({"params": filter(lambda p: p.requires_grad, model.sup1.parameters()),
                   'weight_decay': opts.weight_decay})
    params.append({"params": filter(lambda p: p.requires_grad, model.sup2.parameters()),
                   'weight_decay': opts.weight_decay})
    params.append({"params": filter(lambda p: p.requires_grad, model.cls.parameters()),
                   'weight_decay': opts.weight_decay})

    optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=True)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, max_iters=opts.epochs * len(train_loader), power=opts.lr_power)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    else:
        raise NotImplementedError
    logger.debug("Optimizer:\n%s" % optimizer)

    if model_old is not None:
        model_old = model_old.cuda()
    
    # Put the model on GPU
    model = model.cuda()

    # xxx Load old model from old weights if step > 0!
    if opts.step > 0:
        # get model path
        if opts.step_ckpt is not None:
            path = opts.step_ckpt
        else:
            path = f"checkpoints/step/{task_name}_{opts.name}_{opts.step - 1}_sec.pth"

        # generate model from path
        if os.path.exists(path):
            step_checkpoint = torch.load(path, map_location="cpu")

            hasModule = False
            if list(step_checkpoint['model_state'].keys())[0].startswith("module"):
                hasModule = True
                from collections import OrderedDict
                temp_step_checkpoint = OrderedDict()
                for k, v in step_checkpoint['model_state'].items():
                    new_name = k[7:] # remove 'module'
                    temp_step_checkpoint[new_name] = v

            if hasModule:
                model.load_state_dict(temp_step_checkpoint, strict=False)  # False because of incr. classifiers
            else:
                model.load_state_dict(step_checkpoint['model_state'], strict=False)  # False because of incr. classifiers
            
            if opts.init_balanced:
                # implement the balanced initialization (new cls has weight of background and bias = bias_bkg - log(N+1)
                model.init_new_supervision(device)
                model.init_new_classifier(device)
            # Load state dict from the model state dict, that contains the old model parameters
            if hasModule:
                model_old.load_state_dict(temp_step_checkpoint, strict=True)  # Load also here old parameters
                del temp_step_checkpoint
            else:
                model_old.load_state_dict(step_checkpoint['model_state'], strict=True)  # Load also here old parameters
            
            logger.info(f"[!] Previous model loaded from {path}")
            # clean memory
            del step_checkpoint['model_state']
        elif opts.debug:
            logger.info(f"[!] WARNING: Unable to find of step {opts.step - 1}! Do you really want to do from scratch?")
        else:
            raise FileNotFoundError(path)
        # put the old model into distributed memory and freeze it
        for par in model_old.parameters():
            par.requires_grad = False
        model_old.eval()

    # xxx Set up Trainer
    trainer_state = None
    # if not first step, then instance trainer from step_checkpoint
    if opts.step > 0 and step_checkpoint is not None:
        if 'trainer_state' in step_checkpoint:
            trainer_state = step_checkpoint['trainer_state']

    # instance trainer (model must have already the previous step weights)
    trainer = Trainer(model, model_old, device=device, opts=opts, trainer_state=trainer_state,
                      classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))

    # xxx Handle checkpoint for current model (model old will always be as previous step or None)
    best_score = 0.0
    cur_epoch = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        cur_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint['best_score']
        logger.info("[!] Model restored from %s" % opts.ckpt)
        # if we want to resume training, resume trainer from checkpoint
        if 'trainer_state' in checkpoint:
            trainer.load_state_dict(checkpoint['trainer_state'])
        del checkpoint
    else:
        if opts.step == 0:
            logger.info("[!] Train from scratch")

    # xxx Train procedure
    # print opts before starting training to log all parameters
    logger.add_table("Opts", vars(opts))

    if rank == 0 and opts.sample_num > 0:
        sample_ids = np.random.choice(len(val_loader), opts.sample_num, replace=False)  # sample idxs for visualization
        logger.info(f"The samples id are {sample_ids}")
    else:
        sample_ids = None

    label2color = utils.Label2Color(cmap=utils.color_map(opts.dataset))  # convert labels to images
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])  # de-normalization for original images

    TRAIN = not opts.test
    val_metrics = StreamSegMetrics(n_classes)
    results = {}

    # check if random is equal here.
    logger.print(torch.randint(0,100, (1,1)))
    # train/val here

    scaler = amp.GradScaler()

    while cur_epoch < opts.epochs and TRAIN:
        # =====  Train  =====
        model.train()

        epoch_loss = trainer.train(cur_epoch=cur_epoch, optim=optimizer,
                                   train_loader=train_loader, scheduler=scheduler, scaler=scaler, print_int=opts.print_interval, logger=logger)

        logger.info(f"End of Epoch {cur_epoch}/{opts.epochs}, Average Loss={epoch_loss[0]+epoch_loss[1]},"
                    f" Class Loss={epoch_loss[0]}, Reg Loss={epoch_loss[1]}")

        # =====  Log metrics on Tensorboard =====
        logger.add_scalar("E-Loss", epoch_loss[0]+epoch_loss[1], cur_epoch)
        logger.add_scalar("E-Loss-reg", epoch_loss[1], cur_epoch)
        logger.add_scalar("E-Loss-cls", epoch_loss[0], cur_epoch)
        logger.add_scalar("E-Loss-SEC", epoch_loss[2][0]+epoch_loss[2][1]+epoch_loss[2][2], cur_epoch)
        logger.add_scalar("E-Loss-Seed", epoch_loss[2][0], cur_epoch)
        logger.add_scalar("E-Loss-Expand", epoch_loss[2][1], cur_epoch)
        logger.add_scalar("E-Loss-Constrain", epoch_loss[2][2], cur_epoch)

        # =====  Validation  =====
        if (cur_epoch + 1) % opts.val_interval == 0 or cur_epoch == 0 or cur_epoch == opts.epochs - 2:
            logger.info("validate on val set...")
            model.eval()
            val_loss, val_score, ret_samples = trainer.validate(loader=val_loader, metrics=val_metrics,
                                                                ret_samples_ids=sample_ids, logger=logger)

            logger.print("Done validation")
            logger.info(f"End of Validation {cur_epoch}/{opts.epochs}, Validation Loss={val_loss[0]+val_loss[1]},"
                        f" Class Loss={val_loss[0]}, Reg Loss={val_loss[1]}")

            logger.info(val_metrics.to_str(val_score))

            # =====  Save Best Model  =====
            if rank == 0:  # save best model at the last iteration
                score = val_score['Mean IoU']
                # best model to build incremental steps
                save_ckpt(f"checkpoints/step/{task_name}_{opts.name}_{opts.step}_sec.pth",
                          model, trainer, optimizer, scheduler, cur_epoch, score)
                logger.info("[!] Checkpoint saved.")

            # =====  Log metrics on Tensorboard =====
            # visualize validation score and samples
            logger.add_scalar("V-Loss", val_loss[0]+val_loss[1], cur_epoch)
            logger.add_scalar("V-Loss-reg", val_loss[1], cur_epoch)
            logger.add_scalar("V-Loss-cls", val_loss[0], cur_epoch)
            logger.add_scalar("Val_Overall_Acc", val_score['Overall Acc'], cur_epoch)
            logger.add_scalar("Val_MeanIoU", val_score['Mean IoU'], cur_epoch)
            logger.add_table("Val_Class_IoU", val_score['Class IoU'], cur_epoch)
            logger.add_table("Val_Class_Acc", val_score['Class Acc'], cur_epoch)
            logger.add_figure("Val_Confusion_Matrix", val_score['Confusion Matrix'], cur_epoch)

            # keep the metric to print them at the end of training
            results["V-IoU"] = val_score['Class IoU']
            results["V-Acc"] = val_score['Class Acc']

            if opts.val_interval > 1 or cur_epoch in list(range(0, 5))+[10, 15, 20, 25]+list(range(30, 40)):
                for k, (img, target, lbl) in enumerate(ret_samples):
                    img = (denorm(img) * 255).astype(np.uint8)
                    target = label2color(target).transpose(2, 0, 1).astype(np.uint8)
                    lbl = label2color(lbl).transpose(2, 0, 1).astype(np.uint8)

                    concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                    logger.add_image(f'Sample_{k}', concat_img, cur_epoch)

        cur_epoch += 1

    # =====  Save Best Model at the end of training =====
    if rank == 0 and TRAIN:  # save best model at the last iteration
        # best model to build incremental steps
        save_ckpt(f"checkpoints/step/{task_name}_{opts.name}_{opts.step}_sec.pth",
                  model, trainer, optimizer, scheduler, cur_epoch, best_score)
        logger.info("[!] Checkpoint saved.")

    torch.distributed.barrier()

    # xxx From here starts the test code
    logger.info("*** Test the model on all seen classes...")
    # make data loader
    test_loader = data.DataLoader(test_dst, batch_size=opts.batch_size if opts.crop_val else 1,
                                  sampler=DistributedSampler(test_dst, num_replicas=world_size, rank=rank),
                                  num_workers=opts.num_workers)
    
    sample_ids_test = None
    if (opts.task, opts.step) in [('15-5', 1), ('15-5s', 5)]:
        sample_num_test = 20
        if rank == 0 and opts.sample_num > 0:
            sample_ids_test = np.random.choice(len(test_loader), sample_num_test, replace=False)  # sample idxs for visualization
            logger.info(f"The samples id for test are {sample_ids_test}")

    # load best model
    if TRAIN:
        model = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))
        # Put the model on GPU
        model = model.cuda()
        ckpt = f"checkpoints/step/{task_name}_{opts.name}_{opts.step}_sec.pth"
        checkpoint = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        logger.info(f"*** Model restored from {ckpt}")
        del checkpoint
        trainer = Trainer(model, None, device=device, opts=opts)

    model.eval()

    val_loss, val_score, ret_samples = trainer.validate(loader=test_loader, metrics=val_metrics, ret_samples_ids=sample_ids_test, logger=logger)
    logger.print("Done test")
    logger.info(f"*** End of Test, Total Loss={val_loss[0]+val_loss[1]},"
                f" Class Loss={val_loss[0]}, Reg Loss={val_loss[1]}")
    logger.info(val_metrics.to_str(val_score))

    if (opts.task, opts.step) in [('15-5', 1), ('15-5s', 5)]:
        miou115_list = list(filter(lambda p: p[0]>=1 and p[0]<=15, val_score['Class IoU'].items()))
        miou115 = np.sum(list(map(lambda p: p[1], miou115_list)))/15

        miou1620_list = list(filter(lambda p: p[0]>=16 and p[0]<=20, val_score['Class IoU'].items()))
        miou1620 = np.sum(list(map(lambda p: p[1], miou1620_list)))/5

        miouall_list = list(filter(lambda p: p[0]>=1 and p[0]<=20, val_score['Class IoU'].items()))
        miouall = np.sum(list(map(lambda p: p[1], miouall_list)))/20

        logger.info(f"mIoU  1-15: {miou115}\nmIoU 16-20: {miou1620}\nmIoU all (no bkg): {miouall}")

    logger.add_table("Test_Class_IoU", val_score['Class IoU'])
    logger.add_table("Test_Class_Acc", val_score['Class Acc'])
    logger.add_figure("Test_Confusion_Matrix", val_score['Confusion Matrix'])
    results["T-IoU"] = val_score['Class IoU']
    results["T-Acc"] = val_score['Class Acc']
    logger.add_results(results)

    logger.add_scalar("T_Overall_Acc", val_score['Overall Acc'], opts.step)
    logger.add_scalar("T_MeanIoU", val_score['Mean IoU'], opts.step)
    logger.add_scalar("T_MeanAcc", val_score['Mean Acc'], opts.step)

    for k, (img, target, lbl) in enumerate(ret_samples):
        img = (denorm(img) * 255).astype(np.uint8)
        target = label2color(target).transpose(2, 0, 1).astype(np.uint8)
        lbl = label2color(lbl).transpose(2, 0, 1).astype(np.uint8)

        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
        logger.add_image(f'Test_Sample_{k}', concat_img, cur_epoch)

    logger.close()


if __name__ == '__main__':
    parser = argparser.get_argparser()

    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)

    os.makedirs("checkpoints/step", exist_ok=True)

    cnt = 0
    for param, value in vars(opts).items():
      if param in ['task', 'overlap', 'step', 'batch_size', 'crop_size', 'context_path', 'lr', 'epochs', 'method', 'step_ckpt', 'val_interval', 'print_interval', 'sample_num', 'smooth']:
        print(f'{param}={value}  ', end='')
        cnt += 1
        if cnt % 5 == 0: print()
    print()

    main(opts)

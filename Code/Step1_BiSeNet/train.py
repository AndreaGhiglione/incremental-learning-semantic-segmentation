import sys
import argparse
from dataset.transform import *
from dataset.voc import VOCSegmentation as VOC
from torch.utils.data import DataLoader
import os
from model.build_BiSeNet import BiSeNet
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import random
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
from utils import Label2Color, Denormalize, color_map
from loss import DiceLoss


def get_transform():
    train_transform = Compose([
        RandomResizedCrop(480, (0.5, 2.0)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])
    val_transform = Compose([
        PadCenterCrop(size=512),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def val(args, model, dataloader, ret_samples_ids=None, phase='validation'):
    print('start val!')
    # label_info = get_label_info(csv_path)

    ret_samples = []
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda().long()

            output = model(data)

            # get RGB predict image
            _, prediction = output.max(dim=1)  # B, H, W
            label = label.cpu().numpy()
            prediction = prediction.cpu().numpy()

            # compute per pixel accuracy
            precision = compute_global_accuracy(prediction, label)
            hist += fast_hist(label.flatten(), prediction.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)

            # to have the same samples along the epochs, shuffle=False in dataloader_val
            if phase=='validation' and ret_samples_ids is not None and i in ret_samples_ids:  # get samples
                ret_samples.append((data[0].detach().cpu().numpy(),
                                    label[0],
                                    prediction[0]))
                ret_samples.append((data[1].detach().cpu().numpy(),
                                    label[1],
                                    prediction[1]))

        precision = np.mean(precision_record)
        # miou = np.mean(per_class_iu(hist))
        miou_list = per_class_iu(hist)[:-1]
        # miou_dict, miou = cal_miou(miou_list, csv_path)
        miou = np.mean(miou_list)
        print('precision per pixel for %s: %.3f' % (phase, precision))
        print('mIoU for %s: %.3f\n' % (phase, miou))

        miou_full_list = per_class_iu(hist)
        for i, m in enumerate(miou_full_list):
            print(f'class {i}: {m}')
        fixed_miou = np.mean(miou_full_list)
        print('mIoU for %s: %.3f' % (phase, fixed_miou))
        miou115 = np.mean(miou_full_list[1:16])
        miou1620 = np.mean(miou_full_list[16:])
        miouall = np.mean(miou_full_list[1:])
        print('mIoU  1-15 for %s: %.3f' % (phase, miou115))
        print('mIoU 16-20 for %s: %.3f\n' % (phase, miou1620))
        print('mIoU all (no bkg) for %s: %.3f\n' % (phase, miouall))

        # miou_str = ''
        # for key in miou_dict:
        #     miou_str += '{}:{},\n'.format(key, miou_dict[key])
        # print('mIoU for each class:')
        # print(miou_str)
        return precision, miou, ret_samples


def train(args, model, optimizer, dataloader_train, dataloader_val):
    writer = SummaryWriter(comment=''.format(args.optimizer, args.context_path))

    sample_ids = np.random.choice(len(dataloader_val), args.num_samples, replace=False)  # sample idxs for visualization
    print(f"The samples id are {sample_ids}")

    label2color = Label2Color(cmap=color_map('voc'))  # convert labels to images
    denorm = Denormalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])  # de-normalization for original images

    if args.loss == 'dice':
        loss_func = DiceLoss()
    elif args.loss == 'crossentropy':
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    max_miou = 0
    step = 0

    for epoch in range(args.epoch_start_i, args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        for i, (data, label) in enumerate(dataloader_train):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda().long()

            output, output_sup1, output_sup2 = model(data)
            loss1 = loss_func(output, label)
            loss2 = loss_func(output_sup1, label)
            loss3 = loss_func(output_sup2, label)
            loss = loss1 + loss2 + loss3
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % 100 == 0:
              writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            #torch.save(model.module.state_dict(),
            torch.save(model.state_dict(),
                       os.path.join(args.save_model_path, 'model.pth'))

        if (epoch % args.validation_step == 0 or epoch == args.num_epochs-1) and epoch != 0:
            precision, miou, ret_samples = val(args, model, dataloader_val, ret_samples_ids=sample_ids, phase='validation')
            if miou > max_miou:
                max_miou = miou
                import os
                os.makedirs(args.save_model_path, exist_ok=True)
                #torch.save(model.module.state_dict(),
                torch.save(model.state_dict(),
                           os.path.join(args.save_model_path, 'best_dice_loss.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou_val', miou, epoch)

            if epoch == args.num_epochs-1:
                for k, (img, target, lbl) in enumerate(ret_samples):
                    img = (denorm(img) * 255).astype(np.uint8)
                    target = label2color(target).transpose(2, 0, 1).astype(np.uint8)
                    lbl = label2color(lbl).transpose(2, 0, 1).astype(np.uint8)

                    concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                    
                    writer.add_image(f'Sample_{k}', concat_img, epoch)

            # precision and MIoU on training
            precision, miou, _ = val(args, model, dataloader_train, phase='train')
            writer.add_scalar('epoch/precision_train', precision, epoch)
            writer.add_scalar('epoch/miou_train', miou, epoch)


def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=10, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101",
                        help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='data', help='path of training data')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default="checkpoints", help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='crossentropy', help='loss function, dice or crossentropy')
    parser.add_argument('--num_samples', type=int, default=15, help='number of returned sample images')
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')

    args = parser.parse_args(params)
    print(args)

    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # create dataset and dataloader
    train_path = args.data
    train_transform, val_transform = get_transform()

    dataset_train = VOC(train_path, image_set="train", transform=train_transform)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    dataset_val = VOC(train_path, image_set="val", transform=val_transform)
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = model.cuda()

    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None

    # load pretrained model if exists
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        model.load_state_dict(torch.load(args.pretrained_model_path))
        #model.module.load_state_dict(torch.load(args.pretrained_model_path))
        print('Done!')

    # train
    train(args, model, optimizer, dataloader_train, dataloader_val)

    # val(args, model, dataloader_val, phase='validation')


if __name__ == '__main__':
    params = [
        '--num_epochs', '30',
        '--learning_rate', '1e-3',
        '--data', 'data',
        '--num_workers', '8',
        '--num_classes', '21',
        '--cuda', '0',
        '--batch_size', '16',
        '--save_model_path', './checkpoints_18_sgd',
        '--context_path', 'resnet18',  # set resnet18 or resnet101, only support resnet18 and resnet101
        '--optimizer', 'sgd',

    ]
    
    main(sys.argv[1:])

    # main(params)

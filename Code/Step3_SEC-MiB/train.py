import torch
from torch import distributed
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as functional
from functools import reduce

import numpy as np

from utils.loss import KnowledgeDistillationLoss, BCEWithLogitsLossWithIgnoreIndex, \
    UnbiasedKnowledgeDistillationLoss, UnbiasedCrossEntropy, IcarlLoss

from utils.sec_loss import softmax_layer, seed_loss_layer, expand_loss_layer, crf_layer, constrain_loss_layer
import krahenbuhl2013

from utils import get_regularizer


class Trainer:
    def __init__(self, model, model_old, device, opts, trainer_state=None, classes=None):

        self.model_old = model_old
        self.model = model
        self.device = device

        if classes is not None:
            new_classes = classes[-1]
            tot_classes = reduce(lambda a, b: a + b, classes)
            self.old_classes = tot_classes - new_classes
            self.tot_classes = tot_classes
        else:
            self.old_classes = 0

        # Select the Loss Type
        reduction = 'none'

        self.bce = opts.bce or opts.icarl
        if self.bce:
            self.criterion = BCEWithLogitsLossWithIgnoreIndex(reduction=reduction)
        elif opts.unce and self.old_classes != 0:
            self.criterion = UnbiasedCrossEntropy(old_cl=self.old_classes, ignore_index=255, reduction=reduction)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)

        # ILTSS
        self.lde = opts.loss_de
        self.lde_flag = self.lde > 0. and model_old is not None
        self.lde_loss = nn.MSELoss()

        self.lkd = opts.loss_kd
        self.lkd_flag = self.lkd > 0. and model_old is not None
        if opts.unkd:
            self.lkd_loss = UnbiasedKnowledgeDistillationLoss(alpha=opts.alpha)
        else:
            self.lkd_loss = KnowledgeDistillationLoss(alpha=opts.alpha)

        # ICARL
        self.icarl_combined = False
        self.icarl_only_dist = False
        if opts.icarl:
            self.icarl_combined = not opts.icarl_disjoint and model_old is not None
            self.icarl_only_dist = opts.icarl_disjoint and model_old is not None
            if self.icarl_combined:
                self.licarl = nn.BCEWithLogitsLoss(reduction='mean')
                self.icarl = opts.icarl_importance
            elif self.icarl_only_dist:
                self.licarl = IcarlLoss(reduction='mean', bkg=opts.icarl_bkg)
        self.icarl_dist_flag = self.icarl_only_dist or self.icarl_combined

        # Regularization
        regularizer_state = trainer_state['regularizer'] if trainer_state is not None else None
        self.regularizer = get_regularizer(model, model_old, device, opts, regularizer_state)
        self.regularizer_flag = self.regularizer is not None
        self.reg_importance = opts.reg_importance

        self.ret_intermediate = self.lde

        self.smooth = opts.smooth

    def train(self, cur_epoch, optim, train_loader, scheduler=None, scaler=None, print_int=10, logger=None):
        """Train and return epoch loss"""
        logger.info("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))

        device = self.device
        model = self.model
        # criterion = self.criterion

        epoch_loss = 0.0
        reg_loss = 0.0
        s_loss = 0.0
        e_loss = 0.0
        c_loss = 0.0
        interval_loss = 0.0
        lkd = torch.tensor(0.)
        lde = torch.tensor(0.)
        l_icarl = torch.tensor(0.)
        l_reg = torch.tensor(0.)

        train_loader.sampler.set_epoch(cur_epoch)

        model.train()
        for cur_step, (images, labels, classes, gt_map) in enumerate(train_loader):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            classes = classes.to(device, dtype=torch.float32)
            gt_map = gt_map.to(device, dtype=torch.float32)

            if (self.lde_flag or self.lkd_flag or self.icarl_dist_flag) and self.model_old is not None:
                with torch.no_grad():
                    with amp.autocast():
                        out_old, out_cx1_old, out_cx2_old, features_old = self.model_old(images, ret_intermediate=self.ret_intermediate, phase='train')

            optim.zero_grad()
            
            with amp.autocast():
                out, out_cx1, out_cx2, features = model(images, ret_intermediate=self.ret_intermediate)

            out_down = functional.interpolate(out, size=gt_map.shape[-2:], mode="bilinear")
            out_cx1_down = functional.interpolate(out_cx1, size=gt_map.shape[-2:], mode="bilinear")
            out_cx2_down = functional.interpolate(out_cx2, size=gt_map.shape[-2:], mode="bilinear")
            
            with amp.autocast():
                loss1_SEC_softmax = softmax_layer(out_down)
                loss1_s = seed_loss_layer(loss1_SEC_softmax, gt_map)
                loss1_e = expand_loss_layer(loss1_SEC_softmax, classes, self.tot_classes - 1)
                SEC_CRF_log1 = crf_layer(out_down, images, iternum=10)
                loss1_c = constrain_loss_layer(loss1_SEC_softmax, SEC_CRF_log1)
                loss1_tot = loss1_s + loss1_e + loss1_c

                loss2_SEC_softmax = softmax_layer(out_cx1_down)
                loss2_s = seed_loss_layer(loss2_SEC_softmax, gt_map)
                loss2_e = expand_loss_layer(loss2_SEC_softmax, classes, self.tot_classes - 1)
                SEC_CRF_log2 = crf_layer(out_cx1_down, images, iternum=10)
                loss2_c = constrain_loss_layer(loss2_SEC_softmax, SEC_CRF_log2)
                loss2_tot = loss2_s + loss2_e + loss2_c

                loss3_SEC_softmax = softmax_layer(out_cx2_down)
                loss3_s = seed_loss_layer(loss3_SEC_softmax, gt_map)
                loss3_e = expand_loss_layer(loss3_SEC_softmax, classes, self.tot_classes - 1)
                SEC_CRF_log3 = crf_layer(out_cx2_down, images, iternum=10)
                loss3_c = constrain_loss_layer(loss3_SEC_softmax, SEC_CRF_log3)
                loss3_tot = loss3_s + loss3_e + loss3_c

                loss = loss1_tot + loss2_tot + loss3_tot
            
            loss = loss.mean()

            if self.icarl_combined:
                # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
                n_cl_old = out_old.shape[1]
                # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
                l_icarl = self.icarl * n_cl_old * self.licarl(out.narrow(1, 0, n_cl_old),
                                                              torch.sigmoid(out_old))

            with amp.autocast():
                # xxx ILTSS (distillation on features or logits)
                if self.lde_flag:
                    lde = self.lde * self.lde_loss(features['pre_logits'], features_old['pre_logits'])

                if self.lkd_flag:
                    # resize new output to remove new logits and keep only the old ones
                    lkd = self.lkd * self.lkd_loss(out, out_old)

                # xxx first backprop of previous loss (compute the gradients for regularization methods)
                loss_tot = loss + lkd + lde + l_icarl

            scaler.scale(loss_tot).backward()

            # xxx Regularizer (EWC, RW, PI)
            if self.regularizer_flag:
                if distributed.get_rank() == 0:
                    self.regularizer.update()
                l_reg = self.reg_importance * self.regularizer.penalty()
                if l_reg != 0.:
                    with amp.scale_loss(l_reg, optim) as scaled_loss:
                        scaled_loss.backward()

            scaler.step(optim)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            s_loss += (loss1_s + loss2_s + loss3_s).mean().item()
            e_loss += (loss1_e + loss2_e + loss3_e).mean().item()
            c_loss += (loss1_c + loss2_c + loss3_c).mean().item()

            epoch_loss += loss.item()
            reg_loss += l_reg.item() if l_reg != 0. else 0.
            reg_loss += lkd.item() + lde.item() + l_icarl.item()
            interval_loss += loss.item() + lkd.item() + lde.item() + l_icarl.item()
            interval_loss += l_reg.item() if l_reg != 0. else 0.

            if (cur_step + 1) % print_int == 0:
                interval_loss = interval_loss / print_int
                logger.info(f"Epoch {cur_epoch}, Batch {cur_step + 1}/{len(train_loader)},"
                            f" Image {(cur_step + 1) * len(images)}/{len(train_loader) * len(images)},"
                            f" Loss={interval_loss}")
                logger.info(f"Loss made of: CE {loss}, LKD {lkd}, LDE {lde}, LReg {l_reg}")
                # visualization
                if logger is not None:
                    x = cur_epoch * len(train_loader) + cur_step + 1
                    logger.add_scalar('Loss', interval_loss, x)
                interval_loss = 0.0

        # collect statistics from multiple processes
        epoch_loss = torch.tensor(epoch_loss).to(self.device)
        reg_loss = torch.tensor(reg_loss).to(self.device)
        s_loss = torch.tensor(s_loss).to(self.device)
        e_loss = torch.tensor(e_loss).to(self.device)
        c_loss = torch.tensor(c_loss).to(self.device)

        torch.distributed.reduce(epoch_loss, dst=0)
        torch.distributed.reduce(reg_loss, dst=0)

        if distributed.get_rank() == 0:
            epoch_loss = epoch_loss / distributed.get_world_size() / len(train_loader)
            reg_loss = reg_loss / distributed.get_world_size() / len(train_loader)
            s_loss = s_loss / distributed.get_world_size() / len(train_loader)
            e_loss = e_loss / distributed.get_world_size() / len(train_loader)
            c_loss = c_loss / distributed.get_world_size() / len(train_loader)

        logger.info(f"Epoch {cur_epoch}, Class Loss={epoch_loss}, Reg Loss={reg_loss}, Seed Loss={s_loss}, Expand Loss={e_loss}, Constrain Loss={c_loss}")

        return (epoch_loss, reg_loss, (s_loss, e_loss, c_loss))

    def validate(self, loader, metrics, ret_samples_ids=None, logger=None):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device
        criterion = self.criterion
        model.eval()

        class_loss = 0.0
        reg_loss = 0.0
        lkd = torch.tensor(0.)
        lde = torch.tensor(0.)
        l_icarl = torch.tensor(0.)
        l_reg = torch.tensor(0.)

        ret_samples = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                if (self.lde_flag or self.lkd_flag or self.icarl_dist_flag) and self.model_old is not None:
                    with torch.no_grad():
                        out_old, features_old = self.model_old(images, ret_intermediate=True, phase='valid')

                out, features = model(images, ret_intermediate=True)

                scores = out.detach().cpu().numpy().transpose(2, 3, 1, 0)
                # d1, d2 = float(images.shape[0]), float(images.shape[1])

                images_t = images.detach().cpu().numpy().transpose(2, 3, 1, 0)

                scores_exp = np.exp(scores - np.max(scores, axis=2, keepdims=True))
                probs = scores_exp / np.sum(scores_exp, axis=2, keepdims=True)
                # probs = nd.zoom(probs, (d1 / probs.shape[0], d2 / probs.shape[1], 1.0), order=1)

                eps = 0.00001
                probs[probs < eps] = eps

                if self.smooth:
                    crf_list = []
                    for i_split in range(0, images_t.shape[3]):
                        image_split = images_t[:, :, :, i_split]
                        prob_split = probs[:, :, :, i_split]
                        crf_out = krahenbuhl2013.CRF(image_split, np.log(prob_split), scale_factor=1.0)
                        crf_list.append(crf_out)
                    result = np.stack(crf_list, axis=3)
                    result = np.argmax(result, axis=2)
                else:
                    result = np.argmax(probs, axis=2)

                if self.icarl_combined:
                    # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
                    n_cl_old = out_old.shape[1]
                    # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
                    l_icarl = self.icarl * n_cl_old * self.licarl(out.narrow(1, 0, n_cl_old),
                                                                  torch.sigmoid(out_old))

                # xxx ILTSS (distillation on features or logits)
                if self.lde_flag:
                    lde = self.lde_loss(features['pre_logits'], features_old['pre_logits'])

                if self.lkd_flag:
                    lkd = self.lkd_loss(out, out_old)

                # xxx Regularizer (EWC, RW, PI)
                if self.regularizer_flag:
                    l_reg = self.regularizer.penalty()

                # class_loss += loss.item()
                reg_loss += l_reg.item() if l_reg != 0. else 0.
                reg_loss += lkd.item() + lde.item() + l_icarl.item()

                # _, prediction = out.max(dim=1)

                labels = labels.cpu().numpy()
                # prediction = prediction.cpu().numpy()
                prediction = result.transpose(2, 0, 1)

                metrics.update(labels, prediction)

                if ret_samples_ids is not None and i in ret_samples_ids:  # get samples
                    ret_samples.append((images[0].detach().cpu().numpy(),
                                        labels[0],
                                        prediction[0]))
                    ret_samples.append((images[1].detach().cpu().numpy(),
                                        labels[1],
                                        prediction[1]))

            # collect statistics from multiple processes
            metrics.synch(device)
            score = metrics.get_results()

            class_loss = torch.tensor(class_loss).to(self.device)
            reg_loss = torch.tensor(reg_loss).to(self.device)

            torch.distributed.reduce(class_loss, dst=0)
            torch.distributed.reduce(reg_loss, dst=0)

            if distributed.get_rank() == 0:
                class_loss = class_loss / distributed.get_world_size() / len(loader)
                reg_loss = reg_loss / distributed.get_world_size() / len(loader)

            if logger is not None:
                logger.info(f"Validation, Class Loss={class_loss}, Reg Loss={reg_loss} (without scaling)")

        return (class_loss, reg_loss), score, ret_samples

    def state_dict(self):
        state = {"regularizer": self.regularizer.state_dict() if self.regularizer_flag else None}

        return state

    def load_state_dict(self, state):
        if state["regularizer"] is not None and self.regularizer is not None:
            self.regularizer.load_state_dict(state["regularizer"])

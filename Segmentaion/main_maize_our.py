"""
Author: Benny
Date: Nov 2019
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
from pathlib import Path
from tqdm import tqdm
from dataset import PartNormalDataset,PartPlants
from util import transform
from torch.nn.parallel import DataParallel


sys.path.append("./")
# seg_classes = {'Tomato_seed': [0,1,2]}
# seg_classes = {"Maize":[0,1,2]}
# seg_classes = {"Corn":[0,1,2,3,4,5]}
seg_classes = {"Rapeseed":[0,1,2,3]}

# seg_classes = {"Soybean":[0,1]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='Point_M2AE_SEG_at', help='model name')
    parser.add_argument('--batch_size', type=int, default=10, help='batch Size during training')
    parser.add_argument('--epoch', default=300, type=int, help='epoch to run')
    parser.add_argument('--warmup_epoch', default=10, type=int, help='warmup epoch')
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='initial learning rate')
    parser.add_argument('--log_dir', type=str, default='./plant_mae_finetune/0218/rapeseed_1W_loop10_256-64', help='log path')
    parser.add_argument('--npoint', type=int, default=10000, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--ckpts', type=str, default="../experiments/Plant/Plant_pretrain/pre_8-16-16/ckpt-best.pth", help='ckpts')
    parser.add_argument('--root', type=str, default='/public/xiek/data/perpare_data_pretrain/Plant_MAE_data/Segmentation', help='data root')#/public/xiek/data/Corn
    parser.add_argument('--class_choice', type=str, default="Rapeseed", help='choose a class to train or test')
    parser.add_argument('--sample_num', type=int, default=1000, help='choose a sample number')
    parser.add_argument('--aug', type=bool, default=True, help='data augmentation')

    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)
        
    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = args.root

    if args.aug:
        train_transform = transform.Compose([
            # transform.RandomRotate(along_z=True),
            transform.RandomScale(scale_low=0.8, scale_high=1.2),
            transform.RandomJitter(sigma=0.01, clip=0.05),
            transform.RandomShift()
        ])

        test_transform = transform.Compose([
            transform.RandomRotate(along_z=True)])
    else:
        train_transform = None
        test_transform = None


    #TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='trainval', normal_channel=args.normal)
    TRAIN_DATASET = PartPlants(root=args.root,npoints=args.npoint, split='trainval',class_choice=args.class_choice,normal_channel = False,sample_num=args.sample_num,loop=10,transform=train_transform)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=48, drop_last=True)
    #TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)
    TEST_DATASET = PartPlants(root=args.root,npoints=args.npoint, split='test',class_choice=args.class_choice,normal_channel = False,sample_num=1000,loop=1,transform=None)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=48)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    num_classes = 10
    num_part = 10

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('./%s.py' % args.model, str(exp_dir))
    # shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    # classifier = MODEL.Point_M2AE_SEG(num_part).cuda()

    classifier = MODEL.Point_M2AE_SEG_at(num_part).cuda()
    classifier = DataParallel(classifier)
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)
    print('# generator parameters:', sum(param.numel() for param in classifier.parameters()))
    start_epoch = 0

    if args.ckpts is not None:
        # classifier.load_model_from_ckpt(args.ckpts)
        # checkpoint = torch.load(args.ckpts)
        # classifier.load_state_dict(checkpoint['base_model'], strict=False)
        classifier.module.load_model_from_ckpt(args.ckpts)
## we use adamw and cosine scheduler
    def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                        # print(name)
                no_decay.append(param)
            else:
                decay.append(param)
        return [
                    {'params': no_decay, 'weight_decay': 0.},
                    {'params': decay, 'weight_decay': weight_decay}]

    param_groups = add_weight_decay(classifier, weight_decay=0.05)
    optimizer = optim.AdamW(param_groups, lr= args.learning_rate, weight_decay=0.05 )

    scheduler = CosineLRScheduler(optimizer,
                                  t_initial=args.epoch,
                                  t_mul=1,
                                  lr_min=1e-6,
                                  decay_rate=0.1,
                                  warmup_lr_init=1e-6,
                                  warmup_t=args.warmup_epoch,
                                  cycle_limit=1,
                                  t_in_epochs=True)

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0
    
    classifier.zero_grad()
    for epoch in range(start_epoch, args.epoch):
        mean_correct = []

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''

        classifier = classifier.train()
        loss_batch = []
        num_iter = 0
        '''learning one epoch'''
        for i, (points, label, target,_) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            num_iter += 1
            points = points.data.numpy()
            # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])#data aug
            # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred = classifier(points, to_categorical(label, num_classes))
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]

            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))
            loss = criterion(seg_pred, target)
            loss.backward()
            optimizer.step()
            loss_batch.append(loss.detach().cpu())

            if num_iter == 1:

                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 10, norm_type=2)
                num_iter = 0
                optimizer.step()
                classifier.zero_grad()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)

        train_instance_acc = np.mean(mean_correct)
        loss1 = np.mean(loss_batch)
        log_string('Train accuracy is: %.5f' % train_instance_acc)
        log_string('Train loss: %.5f' % loss1)
        log_string('lr: %.6f' % optimizer.param_groups[0]['lr'])

        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            classifier = classifier.eval()

            for batch_id, (points, label, target,cloud_names) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
                seg_pred = classifier(points, to_categorical(label, num_classes))
                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                target = target.cpu().data.numpy()

                for i in range(cur_batch_size):
                    cat = seg_label_to_cat[target[i, 0]]
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                correct = np.sum(cur_pred_val == target)
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)

                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    cat = seg_label_to_cat[segl[0]]
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (
                                np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        else:
                            part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(np.mean(part_ious))

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            mean_shape_ious = np.mean(list(shape_ious.values()))
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
            for cat in sorted(shape_ious.keys()):
                log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

        log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
            epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))
        if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        log_string('Best accuracy is: %.5f' % best_acc)
        log_string('Best class avg mIOU is: %.5f' % best_class_avg_iou)
        log_string('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
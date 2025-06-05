import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
# import os
from dataset import PartNormalDataset,PartPlants
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
from all_tools import read_ply2np
import torch.nn.functional as F
import shutil
from torch.nn.parallel import DataParallel
import datetime
from pathlib import Path
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
from util.voxelize import voxelize
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from util import transform

sys.path.append("./")
# seg_classes = {'Tomato_seed': [0,1,2]}
seg_classes = {"Corn":[0,1,2,3,4,5]}  #maize collected by ourselves
# seg_classes = {"Maize":[0,1,2]} #maize included in  Pheno4D
# seg_classes = {"Soybean":[0,1]}
# seg_classes = {'ptomato': [0,1,2]} #tomato included in  Pheno4D
# seg_classes = {'Potato': [0,1]}
# seg_classes = {'Tomato': [0,1,2]} #tomato collected by ourselves
# seg_classes = {'Cabbage': [0,1,2,3,4]}
# seg_classes = {'Rapeseed': [0,1,2,3]}
# seg_classes = {'Rice': [0,1]}
# seg_classes = {'Cotton': [0,1,2]}
# seg_classes = {"Rose":[0,1,2]}




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
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='Plant-MAE_SEG', help='model name')#Point_M2AE_SEG_at
    parser.add_argument('--batch_size', type=int, default=4, help='batch Size during training')
    parser.add_argument('--warmup_epoch', default=10, type=int, help='warmup epoch')
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='initial learning rate')
    parser.add_argument('--log_dir', type=str, default='./log/Corn/test', help='log path')
    parser.add_argument('--npoint', type=int, default=10000, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--ckpts', type=str,
                        default="./log/Corn/checkpoints/best_model.pth",
                        help='ckpts')   #/public/xiek/data/Part_Plants  #/public/xiek/data/perpare_data_pretrain/Plant_MAE_data/Segmentation
    parser.add_argument('--root', type=str, default='/public/xiek/data/perpare_data_pretrain/Plant_MAE_data/Segmentation', help='data root')#/public/xiek/data/straw_10W_labeled   #/public/xiek/data/Corn
    parser.add_argument('--class_choice', type=str, default="Corn", help='choose a class to train or test')
    parser.add_argument('--sample_num', type=int, default=10000, help='choose a sample number')
    parser.add_argument('--aug', type=bool, default=False, help='data augmentation')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    parser.add_argument('--save_folder', type=str, default='./log/Corn/test/output',
                        help='save pred_data_path')
    parser.add_argument('--save_pred_label', type=bool, default=False, help=' ')

    return parser.parse_args()



def main(args):
    def log_string(str):
        logger.info(str)
        print(str)
    #experiment_dir = 'log/part_seg/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % args.log_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    traget_root = args.save_folder
    if not os.path.exists(traget_root):
        os.mkdir(traget_root)
    if args.aug:
        test_transform = transform.Compose([
            transform.RandomRotate(along_z=True),
        ])
    else:
        test_transform=None
    TEST_DATASET = PartPlants(root=args.root, npoints=args.npoint, split='test', class_choice=args.class_choice,
                              normal_channel=False, sample_num=1000, loop=1, transform=test_transform)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=10)
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    num_classes = 10
    num_part = 8
    # data, label, seg = testDataLoader

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('./%s.py' % args.model, str(args.log_dir))
    # shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    classifier = MODEL.Plant_MAE_SEG(num_part).cuda()
    # classifier = MODEL.Point_M2AE_SEG_at(num_part).cuda()

    classifier = DataParallel(classifier)

    # criterion = MODEL.get_loss().cuda()
    # classifier.apply(inplace_relu)
    # print('# generator parameters:', sum(param.numel() for param in classifier.parameters()))
    if args.ckpts is not None:
        classifier.module.load_model_from_ckpt(args.ckpts)
        # checkpoint = torch.load(args.ckpts)
        # classifier.load_state_dict(checkpoint['model_state_dict'], strict=False)

    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}


        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
        seg_pred_all = []
        label_all = []
        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat
        iou_per_class = []
        class_tp = {l: 0 for l in seg_classes[cat]}
        class_fp = {l: 0 for l in seg_classes[cat]}
        class_fn = {l: 0 for l in seg_classes[cat]}
        class_tn = {l: 0 for l in seg_classes[cat]}

        classifier = classifier.eval()
        for batch_id, (points, label, target, cloud_names) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                                   smoothing=0.9):
            batchsize, num_point, _ = points.size()
            cur_batch_size, NUM_POINT, _ = points.size()

            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()

            for _ in range(args.num_votes):
                seg_pred= classifier(points, to_categorical(label, num_classes))
                vote_pool += seg_pred

            seg_pred = vote_pool / args.num_votes
            # seg = seg_pred
            cur_pred_val = seg_pred.cpu().data.numpy()

            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()

            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

            correct = np.sum(cur_pred_val == target)
            seg_pred_all.append(cur_pred_val.reshape(-1))
            label_all.append(target.reshape(-1))
            if args.save_pred_label:
                points = points.permute(0, 2, 1).cpu()
                for i in range(len(cloud_names)):
                    os.makedirs(os.path.join(traget_root, 'pred_label'), exist_ok=True)
                    pred_cloud = np.concatenate([points[i], cur_pred_val[i].reshape(-1, 1), target[i].reshape(-1, 1)],
                                                axis=1)
                    np.savetxt(os.path.join(traget_root, f"pred_label/{cloud_names[i]}.txt"), pred_cloud)

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
                #Calculate the IoU for each sample, then compute the average IoU for the entire category.
                for l in seg_classes[cat]:
                    true_positives = np.sum((segp == l) & (segl == l))
                    false_positives = np.sum((segp == l) & (segl != l))
                    false_negatives = np.sum((segp != l) & (segl == l))
                    true_negatives = np.sum((segp != l) & (segl != l))

                    class_tp[l] += true_positives
                    class_fp[l] += false_positives
                    class_fn[l] += false_negatives
                    class_tn[l] += true_negatives

                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))


                iou_per_class.append(part_ious)
                shape_ious[cat].append(np.mean(part_ious))

        n = len(iou_per_class)
        K = len(iou_per_class[0]) if n > 0 else 0
        average_iou = [sum(sublist[i] for sublist in iou_per_class) / n for i in range(K)]

        for i in range(len(average_iou)):
            log_string('Class_{} Result: iou/accuracy {:.4f}.'.format(i, average_iou[i]))


        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        for cat in sorted(shape_ious.keys()):
            log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        log_string('Accuracy is: %.5f' % test_metrics['accuracy'])
        log_string('Class avg mIOU is: %.5f' % test_metrics['class_avg_iou'])

        # 计算每个类别的总的召回率、IoU、精度和 F1 分数
        class_metrics = {}
        for l in seg_classes[cat]:
            tp = class_tp[l]
            fp = class_fp[l]
            fn = class_fn[l]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * prec * recall) / (prec + recall) if (prec + recall) > 0 else 0
           # Calculate the category IoU  for all samples as a whole, then compute the average IoU for the category. Our work used this.
            if tp + fp + fn == 0:
                iou = 1
            else:
                iou = tp / (tp + fp + fn)
            class_metrics[l] = {
                'TP': tp,
                'FP': fp,
                'TN': class_tn[l],
                'FN': fn,
                'Prec': prec,
                'Recall': recall,
                'F1 Score': f1,
                'IoU': iou
            }
        for l, metrics in class_metrics.items():
            log_string('New-Cal-way!!!!')
            log_string('Class {}'.format(l))
            log_string('Val result: Prec/Recall/F1 Score/IoU {:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(metrics['Prec'], metrics['Recall'], metrics['F1 Score'],metrics['IoU']))

            print(f"Class {l}:")
            print(f"  TP: {metrics['TP']}, FP: {metrics['FP']}, TN: {metrics['TN']}, FN: {metrics['FN']}")
            print(
                f"  Prec: {metrics['Prec']:.4f}, Recall: {metrics['Recall']:.4f}, F1 Score: {metrics['F1 Score']:.4f},IoU: {metrics['IoU']:.4f}")


if __name__ == '__main__':
    args = parse_args()
    main(args)

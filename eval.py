# Credit: https://github.com/guanhuankang/Learning-Semantic-Associations-for-Mirror-Detection/blob/main/evaluation.py

import os
import argparse
from tqdm import tqdm
from numpy import *
from joblib import Parallel, delayed
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prediction", type=str, default='/home/wenxue/Code/GlassWizard/pred_512/RGBP')
parser.add_argument("-gt", "--groundtruth", type=str, default='/home/wenxue/Data/Transparent/RGBP/test/mask')
args = parser.parse_args()

class Metrics:
    def __init__(self):
        self.initial()

    def initial(self):
        self.tp = []
        self.tn = []
        self.fp = []
        self.fn = []
        self.precision = []
        self.recall = []
        self.cnt = 0
        self.mae = []
        self.tot = []

    def update(self, pred, target):
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        assert pred.all() >= 0.0 and pred.all() <= 1.0
        assert target.all() >= 0.0 and target.all() <= 1.0
        assert pred.shape == target.shape

        ## threshold = 0.5
        def TP(prediction, true): return sum(logical_and(prediction, true))

        def TN(prediction, true): return sum(logical_and(
            logical_not(prediction), logical_not(true)))

        def FP(prediction, true): return sum(
            logical_and(logical_not(true), prediction))
        def FN(prediction, true): return sum(
            logical_and(logical_not(prediction), true))

        trueThres = 0.5
        predThres = 0.5
        self.tp.append(TP(pred >= predThres, target > trueThres))
        self.tn.append(TN(pred >= predThres, target > trueThres))
        self.fp.append(FP(pred >= predThres, target > trueThres))
        self.fn.append(FN(pred >= predThres, target > trueThres))
        self.tot.append(target.shape[0])
        assert self.tot[-1] == (self.tp[-1]+self.tn[-1] +
                                self.fn[-1]+self.fp[-1])

        # 256 precision and recall
        tmp_prec = []
        tmp_recall = []
        eps = 1e-4
        trueHard = target > 0.5
        for threshold in range(256):
            threshold = threshold / 255.
            tp = TP(pred >= threshold, trueHard)+eps
            ppositive = sum(pred >= threshold)+eps
            tpositive = sum(trueHard)+eps
            tmp_prec.append(tp/ppositive)
            tmp_recall.append(tp/tpositive)
        self.precision.append(tmp_prec)
        self.recall.append(tmp_recall)

        # mae
        self.mae.append(mean(abs(pred-target)))

        self.cnt += 1

    def compute_iou(self):
        iou = []
        n = len(self.tp)
        for i in range(n):
            iou.append(self.tp[i]/(self.tp[i]+self.fp[i]+self.fn[i]))
        return mean(iou)

    def compute_fbeta(self, beta_square=0.3):
        precision = array(self.precision).mean(axis=0)
        recall = array(self.recall).mean(axis=0)
        max_fmeasure = max([(1 + beta_square) * p * r / (beta_square * p + r)
                           for p, r in zip(precision, recall)])
        return max_fmeasure

    def compute_mae(self):
        return mean(self.mae)

    def compute_ber(self):
        ber = []
        for i in range(len(self.tot)):
            subA = self.tp[i]/(self.tp[i]+self.fn[i]) if self.tp[i]+self.fn[i] else 1.0
            subB = self.tn[i]/(self.tn[i]+self.fp[i]) if self.tn[i]+self.fp[i] else 1.0
            cur_ber = 100*(1.0-0.5*(subA+subB))
            ber.append(cur_ber)
        return array(ber).mean()
    
    def report(self):
        report = "Count:"+str(self.cnt)+"\n"
        report += f"IOU: {self.compute_iou()} FB: {self.compute_fbeta()} MAE: {self.compute_mae()} BER: {self.compute_ber()}"
        return report

gt_img_name = [x for x in os.listdir(args.groundtruth) if x.endswith(".png")]
pred_img_name = [x for x in os.listdir(args.prediction) if x.endswith(".jpg")]
n = len(pred_img_name)
print("Num: "+str(n))

def func(idx):
    global gt_img_name, pred_img_name
    met = Metrics()
    name = gt_img_name[idx]
    gt = Image.open(os.path.join(args.groundtruth, name)).convert('L')
    pred = Image.open(os.path.join(args.prediction, name.split('.')[0]+'.jpg')).convert('L')
    
    mask_shape = gt.size
    pred_shape = pred.size
    
    if mask_shape != pred_shape:
        min_height = min(mask_shape[1], pred_shape[1])
        min_width = min(mask_shape[0], pred_shape[0])
        
        gt = gt.resize((min_width, min_height), Image.LANCZOS)
        pred = pred.resize((min_width, min_height), Image.LANCZOS)
    
    gt = array(gt)
    pred = array(pred).astype(uint8)
    gt_max = 255 if gt.max() > 127. else 1.0
    gt = gt / gt_max
    pred = pred.astype(float) / 255.

    met.update(pred=pred, target=gt)
    return met

def main():
    num_worker = 16

    with Parallel(n_jobs=num_worker) as parallel:
        metric_lst = parallel(delayed(func)(i) for i in tqdm(range(n)))
    
    merge_metrics = Metrics()
    
    for x in tqdm(metric_lst):
        merge_metrics.tp += x.tp
        merge_metrics.tn += x.tn
        merge_metrics.fp += x.fp
        merge_metrics.fn += x.fn
        merge_metrics.precision += x.precision
        merge_metrics.recall += x.recall
        merge_metrics.cnt += x.cnt
        merge_metrics.mae += x.mae
        merge_metrics.tot += x.tot

    print(merge_metrics.report())

if __name__ == '__main__':
    main()
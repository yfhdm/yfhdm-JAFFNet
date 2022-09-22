import time
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import glob
import timeit
import  os
from data_loader import SalObjDataset
from skimage import io, transform, color
from data_loader import RescaleT
from data_loader import ToTensorLab
import os
from sod_metrics import MAE, Smeasure, WeightedFmeasure
import numpy as np
import sys

from models import JAFFNet
import torch
import glob

from models import JAFFNet
def eval_adp_e(y_pred, y):

    th = y_pred.mean() * 2
    y_pred_th = (y_pred >= th).float()
    if torch.mean(y) == 0.0:  # the ground-truth is totally black
        y_pred_th = torch.mul(y_pred_th, -1)
        enhanced = torch.add(y_pred_th, 1)
    elif torch.mean(y) == 1.0:  # the ground-truth is totally white
        enhanced = y_pred_th
    else:  # normal cases
        fm = y_pred_th - y_pred_th.mean()
        msd = y - y.mean()
        align_matrix = 2 * msd * fm / (msd * msd + fm * fm + 1e-20)
        enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
    return torch.sum(enhanced) / (y.numel() - 1 + 1e-20)

def test(model, test_img_list,test_gt_list):


    test_salobj_dataset = SalObjDataset(img_name_list=test_img_list, lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(256), ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=0)

    WFM = WeightedFmeasure()
    SM = Smeasure()
    M = MAE()

    model.eval()

    with torch.no_grad():
        for i_test, data_test in enumerate(test_salobj_dataloader):
            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)
            inputs_test = inputs_test.cuda()

            res = model(inputs_test)
            res = res[0].data.cpu().numpy().squeeze()
            pred = (res - res.min()) / (res.max() - res.min() + 1e-8)

            pred = Image.fromarray(pred * 255).convert('RGB')

            msd = io.imread(test_img_list[i_test])
            pred = pred.resize((msd.shape[1], msd.shape[0]), resample=Image.BILINEAR)
            pred = np.array(pred)[:, :, 0]

            gt = io.imread(test_gt_list[i_test])

            WFM.step(pred=pred, gt=gt)
            SM.step(pred=pred, gt=gt)
            M.step(pred=pred, gt=gt)

    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    mae = M.get_results()["mae"]

    em=0
    trans = transforms.Compose([transforms.ToTensor()])
    for i_test, data_test in enumerate(test_salobj_dataloader):

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        inputs_test = inputs_test.cuda()
        res = model(inputs_test)
        res = res[0].data.cpu().numpy().squeeze()
        pred = (res - res.min()) / (res.max() - res.min() + 1e-8)

        # print(res.shape)
        pred = Image.fromarray(pred * 255).convert('RGB')
        msd = io.imread(test_img_list[i_test])
        pred = pred.resize((msd.shape[1], msd.shape[0]), resample=Image.BILINEAR)
        pred = np.array(pred)[:, :, 0]

        gt = io.imread(test_gt_list[i_test])
        pred = trans(pred).cuda()
        gt = trans(gt).cuda()
        em = em + eval_adp_e(pred, gt)

    em=em/len(test_img_list)

    return mae,wfm,sm,em

if __name__ == "__main__":

    dataset='dagm'
    if dataset == 'mt':
        test_dir = '.\SemanticData\Magnetic-tile-defect-datasets\Test\image\\'
        test_img_list = glob.glob(test_dir + '*.jpg')
        test_gt_dir = ".\SemanticData\Magnetic-tile-defect-datasets\Test\gt\\"
        test_gt_list = []
        for img_path in test_img_list:
            img_name = img_path.split("\\")[-1]
            imgIdx = img_name.split(".")[0]
            test_gt_list.append(test_gt_dir + imgIdx + '.png')
    elif dataset == 'dagm':
        test_dir = '.\SemanticData\DAGM\Test\image\\'
        test_img_list = glob.glob(test_dir + '*.PNG')
        test_gt_dir = ".\SemanticData\DAGM\Test\gt\\"
        test_gt_list = []
        for img_path in test_img_list:
            img_name = img_path.split("\\")[-1]
            imgIdx = img_name.split(".")[0]
            test_gt_list.append(test_gt_dir + imgIdx + '.PNG')
    elif dataset == 'neu':
        test_dir=".\SemanticData\SD-saliency-900\Test\image\\"
        test_img_list = glob.glob(test_dir + '*.bmp')
        test_gt_dir = ".\SemanticData\SD-saliency-900\Test\gt\\"
        test_gt_list = []
        for img_path in test_img_list:
            img_name = img_path.split("\\")[-1]
            imgIdx = img_name.split(".")[0]
            test_gt_list.append(test_gt_dir + imgIdx + '.png')

    net= JAFFNet()
    net = net.cuda()
    net.load_state_dict(torch.load("./dagm.pth"))
    mae,wfm,sm,em= test(net, test_img_list,test_gt_list)
    print(" %.4f & %.4f & %.4f & " % (mae,wfm,sm))
    print(em)


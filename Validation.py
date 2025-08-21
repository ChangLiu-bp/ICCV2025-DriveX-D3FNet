import torch
import os
import cv2
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


from networks.Diffdlinknet_v6 import DinkNet34 


BATCHSIZE_PER_CARD = 4

class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        
    def test_one_img_from_path(self, path, evalmode=True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        if batchsize >= 8:
            return self.test_one_img_from_path_8(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_4(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_2(path)

    def test_one_img_from_path_8(self, path):
        img = cv2.imread(path)
        img90 = np.array(np.rot90(img))  
        img1 = np.concatenate([img[None], img90[None]]) 
        img2 = np.array(img1)[:, ::-1] 
        img3 = np.array(img1)[:, :, ::-1] 
        img4 = np.array(img2)[:, :, ::-1] 
      
        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)
      
        img1 = torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda()
        img2 = torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda()
        img3 = torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda()
        img4 = torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda()
        
        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]
        
        return mask2

    def test_one_img_from_path_4(self, path):
        img = cv2.imread(path)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]
        
        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)
        
        img1 = torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda()
        img2 = torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda()
        img3 = torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda()
        img4 = torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda()
        
        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]
        
        return mask2
    
    def test_one_img_from_path_2(self, path):
        img = cv2.imread(path)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = img3.transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = torch.Tensor(img5).cuda()
        img6 = img4.transpose(0, 3, 1, 2)
        img6 = np.array(img6, np.float32) / 255.0 * 3.2 - 1.6
        img6 = torch.Tensor(img6).cuda()
        
        maska = self.net.forward(img5).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]
        
        return mask3
    
    def test_one_img_from_path_1(self, path):
        img = cv2.imread(path)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = torch.Tensor(img5).cuda()
        
        mask = self.net.forward(img5).squeeze().cpu().data.numpy()
        mask1 = mask[:4] + mask[4:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]
        
        return mask3

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

def calculate_metrics(preds, targets):
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()

    
    intersection = np.sum((preds_flat == 1) & (targets_flat == 1))
    union = np.sum(preds_flat) + np.sum(targets_flat) - intersection
    iou = intersection / union if union != 0 else 0

  
    precision = precision_score(targets_flat, preds_flat)
    recall = recall_score(targets_flat, preds_flat)
    f1 = f1_score(targets_flat, preds_flat)

    return iou * 100, f1 * 100, precision * 100, recall * 100

#path of test dateset
source = ' '
val = [name for name in os.listdir(source) if name.endswith('_sat.jpg')]
solver = TTAFrame(DinkNet34)
#path of model weight
solver.load(' ')

#path of output folder
output_folder = ' '
os.makedirs(output_folder, exist_ok=True)


all_iou = []
all_f1 = []
all_precision = []
all_recall = []

for name in val:
  
    mask = solver.test_one_img_from_path(os.path.join(source, name))
    mask[mask >= 4.0] = 255
    mask[mask < 4.0] = 0
    mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)

 
    output_mask_path = os.path.join(output_folder, name.replace('_sat.jpg', '_mask.png')) 
    cv2.imwrite(output_mask_path, mask.astype(np.uint8))
    print(f"Saved predicted mask to {output_mask_path}")

    
    true_mask_path = os.path.join(source, name.replace('_sat.jpg', '_mask.png')) 
    true_mask = cv2.imread(true_mask_path, cv2.IMREAD_GRAYSCALE)

    if true_mask is not None: 
        true_mask[true_mask > 0] = 1
        
        
        iou, f1, precision, recall = calculate_metrics(mask[:, :, 0] > 0, true_mask)

    
        all_iou.append(iou)
        all_f1.append(f1)
        all_precision.append(precision)
        all_recall.append(recall)
    else:
        print(f"Warning: True mask not found for {true_mask_path}.") 


if all_iou: 
    avg_iou = np.mean(all_iou)
    avg_f1 = np.mean(all_f1)
    avg_precision = np.mean(all_precision)
    avg_recall = np.mean(all_recall)


print(f"IoU: {avg_iou:.2f}%")
print(f"F1-score: {avg_f1:.2f}%")
print(f"Precision: {avg_precision:.2f}%")
print(f"Recall: {avg_recall:.2f}%")

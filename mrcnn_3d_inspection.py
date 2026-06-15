import torchvision.transforms as transforms
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from glob import glob
import platform
from MaskRCNNDataset import MySolderingDataset
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.amp import GradScaler
from tools.engine import train_one_epoch, evaluate, test_one_epoch
from torchvision.transforms import v2 as T
from PIL import Image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from tools.utils import collate_fn
from torchvision import tv_tensors
from shutil import copyfile

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomRotation(degrees=(-5,5)))
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2))

    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.Resize((512,512)))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def main(args):

    n_classes = 5

    train_dataset = MySolderingDataset(dataset_dir=os.path.join("dataset", "train"), transforms=get_transform(train=True))
    test_dataset = MySolderingDataset(dataset_dir=os.path.join("dataset", "test"), transforms=get_transform(train=False))

    batch_size = args.batch_size
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')

    # load an instance segmentation model pre-trained on COCO
    model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes+1) # num_classes + background
    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, n_classes+1)
    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0001)
    # optimizer = torch.optim.Adam(params, lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    num_epochs = args.num_epochs

    scaler = GradScaler()
    
    train_losses = []
    test_losses = []
    best_test_loss = np.inf
    if args.command=='train':
        for epoch in range(num_epochs):
            print(f"Epoch: {epoch} / {num_epochs-1}")
            current_lr = lr_scheduler.get_last_lr()[0]
            print("current_lr:", current_lr)

            train_logger, train_loss = train_one_epoch(model, optimizer, train_loader, device, scaler=scaler)
            test_logger, test_loss = test_one_epoch(model, test_loader, device)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            print(f"Train Loss: {train_loss:.4f}\tTest Loss: {test_loss:.4f}")

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                os.makedirs('output', exist_ok=True)
                torch.save(model.state_dict(), os.path.join('output', 'model_best.pth'))

            # update the learning rate
            lr_scheduler.step()

            # evaluate(model, test_loader, device=device)

            save_loss_curve(train_losses, test_losses)

    if args.command=='test':
        # load model
        state_dict = torch.load(os.path.join('output', 'model_best.pth'), weights_only=True, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

        image_paths = glob(os.path.join('data', 'test', '*.png'))
        idx = 0

        eval_transform = get_transform(train=False)

        while True:
            image_path = image_paths[idx]
            # image = cv2.imread(image_path)
            image = Image.open(image_path)
            image = tv_tensors.Image(image)            
            image = eval_transform(image)
            model.eval()
            with torch.no_grad():
                predictions = model([image.to(device)]) #mps에서 연산
            pred = predictions[0]
            #print(predictions)
            #best_score_idx = torch.argmax(pred['scores']).item()
            # print(pred['scores'],best_score_idx)
            # print(pred['masks'],pred['masks'].shape)
            
            #x_points, y_mins, y_means, y_maxs = find_out_points(mask)
            
            mask_on_image, instances = draw_mask_on_image(image_path,pred, score_threshold=0.8)
            size = mask_on_image.shape[:2]

            cv2.imshow('mask_on_image',mask_on_image)
            key = cv2.waitKey()
            if key==ord('q'):
                break
            elif key==ord('a'):
                idx -= 1
            elif key==ord('d'):
                idx += 1
            if idx < 0:
                idx = 0
            if idx > len(image_paths)-1:
                idx = len(image_paths)


def draw_mask_on_image(image_path,pred,score_threshold=0.8):
    image = cv2.imread(image_path)
    h,w,c = image.shape

    masks = (pred["masks"] > 0.2) #true와 false로 이루어짐, instance의 pred의 threshold
    masks = masks.cpu().numpy().astype(np.uint8)*255 #shape([마스크수,1,h,w])

    mask_list = []
    for mask in masks:
        mask = np.squeeze(mask) #어디든 있는 1만 억제 : shape을 맞추기 위해
        mask = cv2.resize(mask, (w,h)) #사이즈 원복
        mask_list.append(mask) #mask가 mask_list로 쌓여감
    
    masks = np.array(mask_list)
    scores = pred['scores'].cpu().numpy() #score 텐서 -> numpy로
    labels = pred['labels'].cpu().numpy() #score크기 큰것부터 작은거순으로 뜸
    
    valid_indicies = np.where(scores > score_threshold)
    masks = masks[valid_indicies]
    scores = scores[valid_indicies]
    labels = labels[valid_indicies]

    colors = {1: (153,148,47), #embo
              2: (79,219,247), #cell_tab
              3: (196,245,225), #al_bead
              4: (58,145,252), #cu_bead
              5: (80,78,255) #slit_hole
              }

    embos = []
    beads = []
    cell_tabs = []
    slit_holes = []

    for mask, score, label in zip(masks, scores, labels): #zip쓸려면 len 같아야함
        
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #마스크 윤곽선 따기 (이미지 위에 오버랩 했을 때 잘보일려고), ([ [x,y], [x,y] ...])
        image = cv2.drawContours(image, contours, -1, colors[label], 2) #윤곽선 좌표를 선으로 이어줌
        image = cv2.putText(image, f'{score:.2f}',contours[0][0][0], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=colors[label],thickness=1) #소수점 둘째자리까지
    
        polygon = contours[0].squeeze().flatten().tolist() #가독성을 위해서 flatten numpy -> list로 변환, contours = (array[..],assray[..]), 0번째면 numpy array
        if label==1 :
            embos.append(polygon)
        if label==2 :
            cell_tabs.append(polygon)
        if label==3 :
            beads.append(polygon)
        if label==4 :
            beads.append(polygon)
        if label==5 :
            slit_holes.append(polygon)

    
    #bead가 중복되는 혼란을 postprocessing = cu/al bead 병합
    instances = {'embos' : [], #{[x,y,x,y...],[x,y,x,y...]}
                 'beads' : [], #{[x,y,x,y...],[x,y,x,y...]}
                 'cell_tabs' : [], #{[x,y,x,y...],[x,y,x,y...]}
                 'slit_holes' : []} #{[x,y,x,y...],[x,y,x,y...]}
    
    instances['beads'] = beads
    instances['embos'] = embos
    instances['cell_tabs'] = cell_tabs
    instances['slit_holes'] = slit_holes

    return image, instances



def save_loss_curve(train_losses, test_losses):
    plt.plot(train_losses, label='Train Loss', marker='.')
    plt.plot(test_losses, label='Test Loss', marker='.')
    plt.grid()
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Loss Curve")
    plt.savefig("loss_curve.png")
    plt.close()



"""parsing and configuration"""
def argparse_args():  
    desc = "Pytorch implementation of 'Mask R-CNN Image Segmentation'"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('command', help="'train' or 'test' or 'labeling'")
    parser.add_argument('--num_epochs', default=300, type=int, help="The number of epochs to run")
    parser.add_argument('--batch_size', default=4, type=int, help="The number of mini-batchs for each epoch")
    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = argparse_args()    
    if args is None:
        exit()
    print(args)
    
    main(args)
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
from convert_data import get_overlay_image
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

            ruled_inspection(instances,mask_on_image,size)

            #ruled_inspection_length(instances, mask_on_image)
            #ruled_inspection_no_weld(instances, mask_on_image)
            
           
            #print(instances)

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

    beads = bead_refinement(beads, size=(h,w))
    
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

def bead_refinement(beads, size):
    mask = np.zeros(size, dtype=np.uint8)
    for bead in beads:
        polygon =np.array(bead).reshape((-1,2))
        mask = cv2.fillPoly(mask, [polygon],255)

        # cv2.imshow('mask',mask)
        # cv2.waitKey()
    
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #마스크 윤곽선 따기 (이미지 위에 오버랩 했을 때 잘보일려고), ([ [x,y], [x,y] ...])

    beads = []
    for contour in contours:
        polygon = contour.squeeze().flatten().tolist()
        beads.append(polygon)

    return beads

def ruled_inspection(instances,image,size):

    embos_lists      = instances['embos']
    beads_lists      = instances['beads']
    slit_holes_lists = instances['slit_holes']

    for embo_list in embos_lists:
        image, result, matched_bead,embo_xmin,embo_xmax,embo_ymin = ruled_inspection_no_weld(embo_list, beads_lists, image)
        if result == 'NO_WELD_PASS':
            image, result = ruled_inspection_length(matched_bead, image)
            if result == 'BEAD_LENGTH_PASS':
                image, result = ruled_inspection_onesided(matched_bead,embo_xmin,embo_xmax,embo_ymin,slit_holes_lists, image, size)

    return image

def ruled_inspection_no_weld (embo_list, beads_lists, image):

    embo_array = np.array(embo_list).reshape(-1,2)
    embo_xmin,embo_ymin = np.min(embo_array, axis=0)
    embo_xmax,embo_ymax = np.max(embo_array, axis=0)

    #bead 0개 기본 - update 방식
    result = 'NO_WELD_FAIL'
    color = (0,0,255)
   
    matched_bead = []
    for bead_list in beads_lists:
        bead_array = np.array(bead_list).reshape(-1,2)
        bead_xmean, bead_ymean = np.mean(bead_array, axis=0)

        if bead_xmean > embo_xmin and bead_xmean < embo_xmax :
            result = 'NO_WELD_PASS'
            color = (255,0,0)
            matched_bead = bead_list

    cv2.putText(image, f'{result}',(embo_xmax+10, embo_ymax+10),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color, thickness=1)

    return image,result,matched_bead,embo_xmin,embo_xmax,embo_ymin


def ruled_inspection_length (matched_bead,image):

    bead_array = np.array(matched_bead).reshape(-1,2)

    _, ymin = np.min(bead_array, axis=0)
    xmax, ymax = np.max(bead_array, axis=0)

    length_mm = (ymax - ymin) *2.5 *0.05 #resize원복(*2.5), 1픽셀당 50um 3D스캔이므로 *0.05

    if length_mm >= 35:
        result = 'BEAD_LENGTH_PASS'
        color = (255,0,0)
    else:
        result = 'BEAD_LENGTH_FAIL'
        color = (0,0,255)

    cv2.putText(image, f'{result} {length_mm:.2f}mm',(xmax+10, ymax+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color, thickness=1)

    return image,result

def ruled_inspection_onesided(matched_bead, embo_xmin, embo_xmax, embo_ymin, slit_holes_lists, image, size):

    slit_holes_mask = np.zeros(size, dtype=np.uint8)
    # offset_slit_holes_mask = np.zeros(size, dtype=np.uint8)

    bead_array = np.array(matched_bead).reshape(-1,2)
    _, ymin = np.min(bead_array, axis=0)
    _, ymax = np.max(bead_array, axis=0)
    print('ymin',ymin)
    print('ymax',ymax)
    H, W = slit_holes_mask.shape[:2] #(H,W,3)에서 2개 잘라오기
    # bead_upper_offset = np.array([[0,0],[W,0],[W,ymin+3],[0,ymin+3]]) #0.4로 resize 되었으니, 1px=50um -> 1px=125um, 4px = 0.5mm / 4px만큼 0으로 채울려면 3px 더하거나 빼줌
    # bead_lower_offset = np.array([[0,ymax-3],[W,ymax-3],[W,H],[0,H]])

    for slit_holes_list in slit_holes_lists:
        slit = np.array(slit_holes_list).reshape(-1,2)
        slit_xmean, _ = np.mean(slit, axis=0)

        if slit_xmean > embo_xmin and slit_xmean < embo_xmax :
            slit_holes_mask = cv2.fillPoly(slit_holes_mask, [slit], 255)
            # offset_slit_holes_mask = cv2.fillPoly(slit_holes_mask, [bead_upper_offset, bead_lower_offset], 0)
    
    slit_holes_mask[ymin:ymin+4, :] = 0
    slit_holes_mask[ymax-4:ymax, :] = 0
    
    contours,_ = cv2.findContours(slit_holes_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    total_length_mm = []
    for contour in contours:
        polygon = contour.squeeze() #shape (polygon갯수,1,[x,y]) -> 크기가 1인 차원 제거
        _, poly_ymin = np.min(polygon, axis=0)
        _, poly_ymax = np.max(polygon, axis=0)
        length_mm = (poly_ymax - poly_ymin) *2.5 *0.05
        total_length_mm.append(length_mm)
        print('poly_ymin', poly_ymin)
        print('poly_ymax', poly_ymax)

    if sum(total_length_mm) >= 10:
        result = 'ONE_SIDED FAIL'
        color = (0,0,255)
    else :
        result = 'ONE_SIDED PASS'
        color = (255,0,0)

    cv2.putText(image, f'{result} {sum(total_length_mm):.2f}mm',(embo_xmax+10,embo_ymin+50),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color, thickness=1)

    return image,result

def convert_tif2png(dataset_dir):
    height_paths = glob(os.path.join(dataset_dir, "*.tif"))
    height_paths = [height_path for height_path in height_paths if "Height" in height_path]

    #png를 train과 test 폴더로 만들어서 넣기위한 makedir 코드
    parent,child = os.path.split(dataset_dir)
    if 'train' in child :
        target_dir = os.path.join(parent, 'train')
    if 'test' in child :
        target_dir = os.path.join(parent, 'test')

    os.makedirs(target_dir, exist_ok=True)

    for height_path in tqdm(height_paths):
        overlay_image = get_overlay_image(height_path)
        overlay_image_path = os.path.basename(height_path.replace('.tif','.png'))
        json_path = height_path.replace('.tif', '.json')
        json_path_new = os.path.join(target_dir, os.path.basename(json_path))

        cv2.imwrite(os.path.join(target_dir,overlay_image_path), overlay_image)
        copyfile(json_path, json_path_new)

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
    parser.add_argument('--batch_size', default=1, type=int, help="The number of mini-batchs for each epoch")
    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = argparse_args()    
    if args is None:
        exit()
    print(args)
    
    main(args)
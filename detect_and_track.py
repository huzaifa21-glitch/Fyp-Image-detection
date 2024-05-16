import os
import cv2
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from numpy import random
from sklearn.svm import SVR
import csv
import tempfile
import os
import json
import shutil
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import argparse
from pathlib import Path
from numpy import random
from random import randint
import torch.backends.cudnn as cudnn
from roboflow import Roboflow
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
                increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, \
                time_synchronized, TracedModel
from utils.download_weights import download

#For SORT tracking
import skimage
from sort import *
folist=[]
total_buildings=0
avg_buildings=0
def cost(x_length):
    # Get the length of X from the user

    # Generate X array starting from 1 up to the provided length
    X = np.arange(1, x_length + 1).reshape(-1, 1)

    # Get the range for Y from the user
    y_min = 13000
    y_max = 35000

    Y = np.random.randint(y_min, y_max + 1, size=(x_length,))

    # Create and fit the SVR model
    model = SVR(kernel='rbf')  # 'rbf' kernel is commonly used for non-linear regression
    model.fit(X, Y)

    # Predict the cost of buildings
    X_test = np.array([x_length]).reshape(-1, 1)
    predicted_prices = model.predict(X_test)
    price = f' {predicted_prices.item() * x_length:.0f}'

def people(x_length):
    # Get the length of X from the user

    # Create X array starting from 1 up to the given length
    X = np.arange(1, x_length + 1).reshape(-1, 1)

    # Get the minimum and maximum values for Y from the user
    y_min = 3
    y_max = 7

    # Generate random numbers between the minimum and maximum values for Y
    Y = np.random.randint(y_min, y_max + 1, size=x_length)

    # Transform the features into polynomial terms
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    # Fit the polynomial regression model
    model = LinearRegression()
    model.fit(X_poly, Y)

    # Predict the total number of people for a given number of buildings
    buildings = np.array([[x_length]])  # Number of buildings for prediction
    buildings_poly = poly.transform(buildings)
    predicted_people = model.predict(buildings_poly)
    peoples = f' {predicted_people.item() * x_length:.0f}'


def fill(xyxy, im, label='', color=(0, 0, 139), line_thickness=4):
    # Extract the coordinates
    x1, y1, x2, y2 = map(int, xyxy)

    # Draw a bounding box
    cv2.rectangle(im, (x1, y1), (x2, y2), color, line_thickness)

    # Draw the label text
    if label:
        font_thickness = max(line_thickness - 1, 1)  # for better visualization of text
        label_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
        cv2.rectangle(im, (x1, y1 - label_size[1]), (x1 + label_size[0], y1), color, -1)  # filled background
        cv2.putText(im, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, line_thickness / 3, (0, 0, 0),
                    thickness=font_thickness, lineType=cv2.LINE_AA)
    return im


#............................... Bounding Boxes Drawing ............................
"""Function to Draw Bounding boxes"""
def draw_boxes(img, bbox, identities=None, categories=None, names=None, save_with_object_id=False, path=None, offset=(0, 0), delt=0, city=None):
    buildings = 0  # Initialize the count of buildings
    global total_buildings
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0

        area = (x2 - x1)

        # Check if the area is greater than 90
        if area < 100:
            continue

        # Check if the detected object does not have class_id equal to 0 (assuming class_id is the last element in the det array)
        if cat != 0:
            buildings += 1  # Increment the count of buildings

            # Your existing code for processing the detected object goes here...
            # For example, you can print the label and draw the box
            label = str(id) + ":" + names[cat]
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 20), 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 144, 30), -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, [255, 255, 255], 1)

            # Increment the count of boxes drawn
            total_buildings += 1

            txt_str = ""
            if save_with_object_id:
                txt_str += "%i %i %f %f %f %f %f %f" % (
                    id, cat, int(box[0]) / img.shape[1], int(box[1]) / img.shape[0], int(box[2]) / img.shape[1],
                    int(box[3]) / img.shape[0], int(box[0] + (box[2] * 0.5)) / img.shape[1],
                    int(box[1] + (box[3] * 0.5)) / img.shape[0])
                txt_str += "\n"
                with open(path + '.txt', 'a') as f:
                    f.write(txt_str)

    tempfolder_path = '..\\temp'
    # print('total buildings: ', buildings)

    if delt == 1:
        # print(delt)
        folder_path = tempfolder_path

        if os.path.exists(folder_path):
            # List all files and subdirectories in the folder
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isfile(item_path):
                    # Remove files
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    # Remove directories and their contents
                    shutil.rmtree(item_path)
                    print(f"The folder '{folder_path}' has been emptied.")
                else:
                    print(f"The folder '{folder_path}' does not exist.")

    # city = 'Lahore'


    return img, buildings,total_buildings


def detect(save_img=False):
    source, weights,city,delt, view_img, save_txt, imgsz, trace, colored_trk, save_bbox_dim, save_with_object_id= opt.source, opt.weights,opt.city,opt.delt, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.colored_trk, opt.save_bbox_dim, opt.save_with_object_id
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))


    #.... Initialize SORT ....
    #......................... 
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh)
    #......................... 
    
    
    #........Rand Color for every trk.......
    rand_color_list = []
    amount_rand_color_prime = 5003 # prime number
    for i in range(0,amount_rand_color_prime):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)
    #......................................
   

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt or save_with_object_id else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()


    for path, img, im0s, vid_cap in dataset:
        global total_buildings
        total_buildings = 0
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                #..................USE TRACK FUNCTION....................
                #pass an empty array to sort
                dets_to_sort = np.empty((0,6))
                
                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                np.array([x1, y1, x2, y2, conf, detclass])))
                
                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks =sort_tracker.getTrackers()

                txt_str = ""

                #loop over tracks
                for track in tracks:
                    # color = compute_color_for_labels(id)
                    #draw colored tracks
                    if colored_trk:
                        [cv2.line(im0, (int(track.centroidarr[i][0]),
                                    int(track.centroidarr[i][1])), 
                                    (int(track.centroidarr[i+1][0]),
                                    int(track.centroidarr[i+1][1])),
                                    rand_color_list[track.id % amount_rand_color_prime], thickness=2) 
                                    for i,_ in  enumerate(track.centroidarr) 
                                      if i < len(track.centroidarr)-1 ] 
                    #draw same color tracks
                    else:
                        [cv2.line(im0, (int(track.centroidarr[i][0]),
                                    int(track.centroidarr[i][1])), 
                                    (int(track.centroidarr[i+1][0]),
                                    int(track.centroidarr[i+1][1])),
                                    (255,0,0), thickness=2) 
                                    for i,_ in  enumerate(track.centroidarr) 
                                      if i < len(track.centroidarr)-1 ] 

                    if save_txt and not save_with_object_id:
                        # Normalize coordinates
                        txt_str += "%i %i %f %f" % (track.id, track.detclass, track.centroidarr[-1][0] / im0.shape[1], track.centroidarr[-1][1] / im0.shape[0])
                        if save_bbox_dim:
                            txt_str += " %f %f" % (np.abs(track.bbox_history[-1][0] - track.bbox_history[-1][2]) / im0.shape[0], np.abs(track.bbox_history[-1][1] - track.bbox_history[-1][3]) / im0.shape[1])
                        txt_str += "\n"
                
                if save_txt and not save_with_object_id:
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(txt_str)


                # draw boxes for visualization
                if len(tracked_dets)>0:
                    bbox_xyxy = tracked_dets[:,:4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    draw_boxes(im0, bbox_xyxy, identities, categories, names, save_with_object_id, txt_path, city=opt.city)

            else: #SORT should be updated even with no detections
                tracked_dets = sort_tracker.update()
            #........................................................
            
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            
        
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                  cv2.destroyAllWindows()
                  raise StopIteration

            # Save results (image with detections)

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                    relative_path = ('..\\temp\\tempEarth.jpg')
                    cv2.imwrite(relative_path, im0)

                else:  # 'video' or 'stream'

                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        # print('v',vid_path)
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                            # print('s',save_path)
                        vid_writer = cv2.VideoWriter('..\\temp\\video\\EarthVideo.mp4',cv2.VideoWriter_fourcc(*'avc1'), fps,(w,h))
                    vid_writer.write(im0)
        if (total_buildings != 0):
            folist.append(total_buildings)

    print('list:', folist)
    avg_buildings = sum(folist) / len(folist) if len(folist) > 0 else 0
    print('sur', avg_buildings)

    file_path = 'CitiesDensity1.csv'  # Update the file path

    # Get the current working directory
    current_directory = os.getcwd()

    with open(file_path, 'r') as file:
        # Create a CSV reader
        reader = csv.reader(file)

        # Initialize a variable to store the household value for Islamabad
        household = None
        # print(city);
        # Loop through the CSV data and find the household value for Islamabad
        for row in reader:
            if row[0] == city:
                household = row[-2]
                pricex = row[-1]
                # print("HOUSEHOLDS " + str(household))
                # pricex = round((float(pricex) * 70) / 100, 0)
                # print(pricex)
                pricex = float(pricex)
                price = avg_buildings * pricex
                people = round((float(household) * avg_buildings), 0)
                # print(price, buildings, people);
                data = {
                    "DestroyedBuildings": f'{round(avg_buildings,2)}',
                    "PeopleAffected": people,
                    "TotalDamageCost": price,
                    "Location": city
                }
                folder_path = 'temp'
                # Change this to your desired folder path
                os.makedirs(folder_path, exist_ok=True)
                filename = "earthquakeData.json"
                file_path = os.path.join(folder_path, filename)
                with open('..\\temp\\earthquakeData.json', "w") as json_file:
                    json.dump(data, json_file)
                # print(f"JSON data saved to: {file_path}")
                # Save results (image with detections)



    if save_txt or save_img or save_with_object_id:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--download', action='store_true', help='download model weights automatically')
    parser.add_argument('--no-download', dest='download', action='store_false',help='not download model weights if already exist')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='object_tracking', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--colored-trk', action='store_true', help='assign different color to every track')
    parser.add_argument('--save-bbox-dim', action='store_true', help='save bounding box dimensions with --save-txt tracks')
    parser.add_argument('--save-with-object-id', action='store_true', help='save results with object id to *.txt')
    parser.add_argument('--delt', type=int, help='don`t trace model', default=0, )
    parser.add_argument('--city', type=str, help='name of the city', default='')
    parser.set_defaults(download=True)
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))
    if opt.download and not os.path.exists(''.join(opt.weights)):
        print('Model weights not found. Attempting to download now...')
        download('./')

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()

import os
import cv2
import time
import torch
import argparse
from pathlib import Path
from numpy import random
from random import randint
import torch.backends.cudnn as cudnn
import json

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

# For SORT tracking
import skimage
from sort import *

survivors_data = []
folist = []

num_persons_detected=0
avg_survivors=0

tempfolder_path1 = os.path.join(os.getcwd(), 'survivor')

def allocate_resources(num_persons_detected, disaster_type):
    # Define resource allocation rules based on disaster type
    resource_rules = {
        "Flood": [
            {"threshold": 50, "ambulances": 5, "boats": 7, "helicopters": 3, "rescue_volunteers": 28},
            {"threshold": 20, "ambulances": 4, "boats": 5, "helicopters": 1, "rescue_volunteers": 22},
            {"threshold": 10, "ambulances": 2, "boats": 3, "rescue_volunteers": 15},
            {"threshold": 0, "ambulances": 1, "boats": 2, "rescue_volunteers": 8}
        ],
        # Add more disaster types and rules as needed
        "Default": [
            {"threshold": 50, "ambulances": 5, "helicopters": 2, "rescue_volunteers": 26},
            {"threshold": 20, "ambulances": 3, "rescue_volunteers": 16},
            {"threshold": 10, "ambulances": 2, "rescue_volunteers": 12},
            {"threshold": 0, "ambulances": 2, "rescue_volunteers": 8}
        ]
    }

    # Select the appropriate rules based on the disaster type
    rules = resource_rules.get(disaster_type, resource_rules["Default"])

    # Find the matching rule based on the number of persons detected
    for rule in rules:
        if num_persons_detected >= rule["threshold"]:
            return {
                "ambulances": rule.get("ambulances", 0),
                "boats": rule.get("boats", 0),
                "helicopters": rule.get("helicopters", 0),
                "rescue_volunteers": rule.get("rescue_volunteers", 0)
                # Add more resources as needed
            }

    return {}  # Return an empty dictionary if no matching rule is found



def read_coordinates_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            bounding_boxes = []
            if not lines:  # If file is empty
                return 0

            for line in lines:
                coords = line.strip().split(',')
                bounding_boxes.append(list(map(int, coords)))
            return bounding_boxes
    except FileNotFoundError:  # If file is not found
        return 0


# ............................... Bounding Boxes Drawing ............................
"""Function to Draw Bounding boxes"""
def draw_boxes(img, bbox, identities=None, categories=None, names=None, save_with_object_id=False, path=None,
               offset=(0, 0), type=None,delt=0):

    global num_persons_detected  # Initialize the count of boxes drawn
    tempfolder_path = os.path.join(os.getcwd(), 'survivor')
    json_file_path = os.path.join(tempfolder_path, 'survivors_data.json')

    if os.path.exists(json_file_path):
        os.remove(json_file_path)


    # List to store data of survivors

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # print(delt)
        if (delt == 2):
            incoming_coords = (x1, y1, x2, y2)
            bounding_boxes = read_coordinates_from_file('floodstats/floodcoord.txt')
            # print(bounding_boxes);
            if (bounding_boxes != 0):
                for x1, y1, x2, y2 in bounding_boxes:

                    if x1 <= incoming_coords[0] <= x2 and y1 <= incoming_coords[1] <= y2:

                        # Calculate the area of the bounding box
                        area = (x2 - x1)
                        # print(area)
                        # Check if the area is greater than 700
                        # if area > 30:
                        #     continue  # Skip drawing the box

                        cat = int(categories[i]) if categories is not None else 0
                        id = int(identities[i]) if identities is not None else 0
                        data = (int((box[0] + box[2]) / 2), (int((box[1] + box[3]) / 2)))

                        # Check if the detected object has class_id equal to 5 (assuming class_id is the last element in the det array)
                        if int(box[-1]) == 3:

                            label = "Survivor"
                            survivors_data.append({
                                "id": id,
                                "category": cat,
                                "position": data
                            })

                        else:

                            label = "Survivor"

                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(img, (incoming_coords[0], incoming_coords[1]), (incoming_coords[2], incoming_coords[3]), (0, 255, 0), 2)  # Change rectangle color to green
                        cv2.rectangle(img, (incoming_coords[0], incoming_coords[1] - 20), (incoming_coords[0] + w, incoming_coords[1]), (0, 255, 0), -1)  # Change rectangle color to green
                        cv2.putText(img, label, (incoming_coords[0], incoming_coords[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1) # text color to black


                        # Increment the count of boxes drawn
                        num_persons_detected += 1

                        # Rest of the code remains unchanged
                        txt_str = ""
                        if save_with_object_id:
                            txt_str += "%i %i %f %f %f %f %f %f" % (
                                id, cat, int(box[0]) / img.shape[1], int(box[1]) / img.shape[0], int(box[2]) / img.shape[1],
                                int(box[3]) / img.shape[0], int(box[0] + (box[2] * 0.5)) / img.shape[1],
                                int(box[1] + (box[3] * 0.5)) / img.shape[0])
                            txt_str += "\n"
                            with open(path + '.txt', 'a') as f:

                                f.write(txt_str)
        elif (delt == 1):
            print('delt 1 working')
            # Calculate the area of the bounding box
            area = (x2 - x1)
            # print(area)
            # Check if the area is greater than 700
            if area > 30:
                continue  # Skip drawing the box

            cat = int(categories[i]) if categories is not None else 0
            id = int(identities[i]) if identities is not None else 0
            data = (int((box[0] + box[2]) / 2), (int((box[1] + box[3]) / 2)))

            # Check if the detected object has class_id equal to 5 (assuming class_id is the last element in the det array)
            if int(box[-1]) == 3:

                label = "Survivor"
                survivors_data.append({
                    "id": id,
                    "category": cat,
                    "position": data
                })

            else:

                label = "Survivor"

            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Change rectangle color to green
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)  # Change rectangle color to green
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0),
                        1)  # text color to black

            # Increment the count of boxes drawn
            num_persons_detected += 1

            # Rest of the code remains unchanged
            txt_str = ""
            if save_with_object_id:
                txt_str += "%i %i %f %f %f %f %f %f" % (
                    id, cat, int(box[0]) / img.shape[1], int(box[1]) / img.shape[0], int(box[2]) / img.shape[1],
                    int(box[3]) / img.shape[0], int(box[0] + (box[2] * 0.5)) / img.shape[1],
                    int(box[1] + (box[3] * 0.5)) / img.shape[0])
                txt_str += "\n"
                with open(path + '.txt', 'a') as f:
                    f.write(txt_str)



    return img, num_persons_detected



# Example usage:
# img, detected_persons = draw_boxes(img, bbox, identities, categories, names, save_with_object_id, path, offset, disaster_type)




# ..............................................................................

def detect(save_img=False,):
    source, weights,type, view_img, save_txt, imgsz, trace, colored_trk, save_bbox_dim, save_with_object_id = opt.source, opt.weights,opt.type,opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.colored_trk, opt.save_bbox_dim, opt.save_with_object_id
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # .... Initialize SORT ....
    # .........................
    sort_max_age = 5
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)
    # .........................

    # ........Rand Color for every trk.......
    rand_color_list = []
    amount_rand_color_prime = 5003  # prime number
    for i in range(0, amount_rand_color_prime):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)
    # ......................................

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt or save_with_object_id else save_dir).mkdir(parents=True,
                                                                                 exist_ok=True)  # make dir

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
        global num_persons_detected
        num_persons_detected=0
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
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

                # ..................USE TRACK FUNCTION....................
                # pass an empty array to sort
                dets_to_sort = np.empty((0, 6))

                # NOTE: We send in detected object class too
                for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort,
                                              np.array([x1, y1, x2, y2, conf, detclass])))

                # Run SORT


                # ... (other code)

                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks = sort_tracker.getTrackers()

                txt_str = ""

                # loop over tracks
                for track in tracks:
                    # color = compute_color_for_labels(id)
                    # draw colored tracks
                    if colored_trk:
                        [cv2.line(im0, (int(track.centroidarr[i][0]),
                                        int(track.centroidarr[i][1])),
                                  (int(track.centroidarr[i + 1][0]),
                                   int(track.centroidarr[i + 1][1])),
                                  rand_color_list[track.id % amount_rand_color_prime], thickness=2)
                         for i, _ in enumerate(track.centroidarr)
                         if i < len(track.centroidarr) - 1]
                        # draw same color tracks
                    else:
                        [cv2.line(im0, (int(track.centroidarr[i][0]),
                                        int(track.centroidarr[i][1])),
                                  (int(track.centroidarr[i + 1][0]),
                                   int(track.centroidarr[i + 1][1])),
                                  (255, 0, 0), thickness=2)
                         for i, _ in enumerate(track.centroidarr)
                         if i < len(track.centroidarr) - 1]

                    if save_txt and not save_with_object_id:
                        # Normalize coordinates
                        txt_str += "%i %i %f %f" % (track.id, track.detclass, track.centroidarr[-1][0] / im0.shape[1],
                                                    track.centroidarr[-1][1] / im0.shape[0])
                        if save_bbox_dim:
                            txt_str += " %f %f" % (
                            np.abs(track.bbox_history[-1][0] - track.bbox_history[-1][2]) / im0.shape[0],
                            np.abs(track.bbox_history[-1][1] - track.bbox_history[-1][3]) / im0.shape[1])
                        txt_str += "\n"

                if save_txt and not save_with_object_id:
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(txt_str)

                # draw boxes for visualization
                if len(tracked_dets) > 0:
                    bbox_xyxy = tracked_dets[:, :4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    draw_boxes(im0, bbox_xyxy, identities, categories, names, save_with_object_id, txt_path,type=opt.type,delt=opt.delt)

            else:  # SORT should be updated even with no detections
                tracked_dets = sort_tracker.update()
            # ........................................................

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
                    cv2.imwrite(os.path.join(tempfolder_path1, "result.png"), im0)
                    print(f" The image with the result is saved in: {tempfolder_path1}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter('survivor/survivorVideo.mp4',cv2.VideoWriter_fourcc(*'avc1'), fps,(w,h))
                    vid_writer.write(im0)
        if (num_persons_detected != 0):
            folist.append(num_persons_detected)
    print('list:',folist)
    avg_survivors = sum(folist) / len(folist) if len(folist) > 0 else 0
    print('sur',avg_survivors)

    # Example usage:
    #  # Replace with the actual number of persons detected
    allocated_resources = allocate_resources(avg_survivors,type)
    human_result = {
        'Total survivors detected': avg_survivors,
        'Ambulances': allocated_resources["ambulances"],
        'boats': allocated_resources["boats"],
        'helicopters': allocated_resources["helicopters"],
        'rescue_volunteers': allocated_resources["rescue_volunteers"],
            # Add more resources as needed

    }
    tempfolder_path = os.path.join(os.getcwd(), 'survivor')
    if avg_survivors > 0:
        # print(f"Total survivors detected: {avg_survivors}")

        relative_folder_path = tempfolder_path

        # Construct the absolute folder path by joining it with the current working directory
        absolute_folder_path = os.path.join(os.getcwd(), relative_folder_path)

        # Create the folder if it doesn't exist
        os.makedirs(absolute_folder_path, exist_ok=True)

        # Specify the relative file path for the JSON file
        relative_filename = 'survivors_data.json'

        # Construct the absolute file path by joining it with the folder path
        absolute_file_path = os.path.join(absolute_folder_path, relative_filename)

        # Save JSON data to the absolute file path
        with open(absolute_file_path, "w") as json_file:
            json.dump(human_result, json_file)

        print(f"JSON data saved to: {absolute_file_path}")
    else :
        print("No survivors Found");
        human_result = {
            'Total survivors detected': round(num_persons_detected,0),
                'ambulances': '0',
                'boats': '0',
                'helicopters': '0',
                'rescue_volunteers': '0',

        }
        relative_folder_path = tempfolder_path

        # Construct the absolute folder path by joining it with the current working directory
        absolute_folder_path = os.path.join(os.getcwd(), relative_folder_path)

        # Create the folder if it doesn't exist
        os.makedirs(absolute_folder_path, exist_ok=True)

        # Specify the relative file path for the JSON file
        relative_filename = 'survivors_data.json'

        # Construct the absolute file path by joining it with the folder path
        absolute_file_path = os.path.join(absolute_folder_path, relative_filename)

        # Save JSON data to the absolute file path
        with open(absolute_file_path, "w") as json_file:
            json.dump(human_result, json_file)

        print(f"JSON data saved to: {absolute_file_path}")
    if save_txt or save_img or save_with_object_id:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--download', action='store_true', help='download model weights automatically')
    parser.add_argument('--no-download', dest='download', action='store_false',
                        help='not download model weights if already exist')
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
    parser.add_argument('--save-bbox-dim', action='store_true',
                        help='save bounding box dimensions with --save-txt tracks')
    parser.add_argument('--save-with-object-id', action='store_true', help='save results with object id to *.txt')
    parser.add_argument('--delt', type=int, help='don`t trace model', default=0, )
    parser.add_argument('--type', type=str, help='type of disaster (e.g., flood)', default='flood')


    parser.set_defaults(download=True)
    opt = parser.parse_args()
    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))
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
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


if (delta == 1):
    incoming_coords = (x1, y1, x2, y2)
    temp = 0

    bounding_boxes = read_coordinates_from_file('floodstats\\floodcoord.txt')
    if (bounding_boxes != 0):
        for x1, y1, x2, y2 in bounding_boxes:

            if x1 <= incoming_coords[0] <= x2 and y1 <= incoming_coords[1] <= y2:
                if (temp == 0):
                    # p=p+1
                    inframe = inframe + 1

                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(img, (incoming_coords[0], incoming_coords[1]),
                                  (incoming_coords[2], incoming_coords[3]), (0, 255, 0), 2)

                    cv2.rectangle(img, (incoming_coords[0], incoming_coords[1] - 20),
                                  (incoming_coords[0] + w, incoming_coords[1]), (0, 0, 0), -1)
                    cv2.putText(img, label, (incoming_coords[0], incoming_coords[1] - 5), cv2.FONT_HERSHEY_DUPLEX,
                                0.6, [255, 255, 255], 1)
                    temp = temp + 1
                    # cv2.circle(img, data, 6, color,-1)   #centroid of box
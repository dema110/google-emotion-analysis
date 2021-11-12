def draw_bbox(bounding_boxes, image):
    """Draws bounding boxes for each image"""
    import cv2

    for i in range(len(bounding_boxes)):

        x1 = bounding_boxes[i][0,0]
        x2 = bounding_boxes[i][1,0]
        y1 = bounding_boxes[i][0,1]
        y2 = bounding_boxes[i][1,1]

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)),
                     color_code(i), 2)
    
    return image

def color_code(i):
    """Defines tuple values for RGB color codes, black if error"""
    
    # Case statements new in Python 3.10
    # https://docs.python.org/3/whatsnew/3.10.html#pep-634-structural-pattern-matching
    match i:
        case 0:
            return (0, 0, 255)      # red
        case 1:
            return (0, 255, 0)      # green
        case 2:
            return (255, 0, 0)      # blue
        case _:
            return (0, 0, 0)        # black


def plot_landmarks(landmarks, image):
    """Plot the facial landmarks"""
    import cv2

    for i in range(len(landmarks)):
        for p in range(landmarks[i].shape[0]):
            cv2.circle(image, 
                      (int(landmarks[i][p, 0]), int(landmarks[i][p, 1])),
                      2, (0, 0, 255), -1, cv2.LINE_AA)
    return image

 
def folder_files(path):
    """Get full path of all files in folder"""
    from os import listdir
    from os.path import isfile, join    

    files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

    return files


def open_files(path):
    """Extract file names from excel workbook, specific for the excel file given"""
    from openpyxl import load_workbook

    wb = load_workbook(path)

    columns = wb['One Face']["B"] + wb['Two Faces']["B"] + wb['Three Faces']["B"]
    image_list = [columns[x].value for x in range(len(columns)) if (columns[x].value != None and columns[x].value != "image_name")]

    # image_paths = [os.path.join(dir_path, x.split('.')[0] + "." + "jpg") for x in image_list]

    return image_list


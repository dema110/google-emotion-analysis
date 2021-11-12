import io
import pandas as pd
import numpy as np
import cv2

from os.path import join
from PIL import Image

from google.cloud import vision

# utility methods in utils.py
import utils


def detect_faces(path):
    """Detects faces in an image."""


    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.face_detection(image=image)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return response.face_annotations


def print_debug(filtered_response, path):
    """Quick visual output when running code"""

    print('=' * 60)
    print('Image:', path.split('/')[-1])
    print('Faces found:', len(filtered_response))
    print('=' * 60)

    for face in filtered_response:

        likelihood_joy = vision.Likelihood(face.joy_likelihood)
        likelihood_sorrow = vision.Likelihood(face.sorrow_likelihood)
        likelihood_anger = vision.Likelihood(face.anger_likelihood)
        likelihood_surprise = vision.Likelihood(face.surprise_likelihood)
        vertices = ['(%s,%s)' % (v.x, v.y) for v in face.bounding_poly.vertices]

        print('Face joy:\t', likelihood_joy.name)
        print('Face sorrow:\t', likelihood_sorrow.name)
        print('Face anger:\t', likelihood_anger.name)
        print('Face surprise:\t', likelihood_surprise.name)
        print('Confidence:\t', face.detection_confidence)
        print('Face bounds:\t', ",".join(vertices))

        print() 

    print()
    print()


def log_face(face, color_name, num_faces, path, results):
    """Appends data to the results running list; one row is one face"""

    row = {
        'image_name': path.split('/')[-1],
        'faces_filtered': num_faces,
        'color': color_name,
        'likelihood_joy': vision.Likelihood(face.joy_likelihood).name,
        'likelihood_sorrow': vision.Likelihood(face.sorrow_likelihood).name,
        'likelihood_anger': vision.Likelihood(face.anger_likelihood).name,
        'likelihood_surprise': vision.Likelihood(face.surprise_likelihood).name,
        'vertices': ['(%s,%s)' % (v.x, v.y) for v in face.bounding_poly.vertices],
        'confidence': face.detection_confidence
    }

    results.append(row)


def log_face_not_found(num_faces, path, results):
    """Appends a blank row if no face found"""
        
    row = {
        'image_name': path.split('/')[-1],
        'faces_filtered': num_faces,
        'color': None,
        'likelihood_joy': None,
        'likelihood_sorrow': None,
        'likelihood_anger': None,
        'likelihood_surprise': None,
        'vertices': None,
        'confidence': None
    }

    results.append(row)


def save_csv(results, output_folder, output_csv):
    """Append rows to CSV file, append only no overwrite"""

    df = pd.DataFrame()

    for row in results:
        df = df.append(row, ignore_index=True)
    
    with open(join(output_folder, output_csv), 'a') as f:
        df.to_csv(f, sep=',', encoding='utf-8', header=False)


def process_image(path, min_detection_confidence):
    """Call detect_faces and get results from Vision API; filter by min detection confidence"""

    response = detect_faces(path)

    filtered_response = [face for face in response if face.detection_confidence >= min_detection_confidence]

    return filtered_response


def save_image(faces, path, output_folder):
    """Save image with bounding boxes drawn"""

    image = Image.open(path).convert('RGB')
    image_array = np.array(image, dtype=np.float32)
    # cv2 image color conversion
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    bounding_boxes = []

    for face in faces:
        a = [
            [face.bounding_poly.vertices[0].x, face.bounding_poly.vertices[0].y],
            [face.bounding_poly.vertices[2].x, face.bounding_poly.vertices[2].y]
        ]
        bounding_boxes.append(np.array(a))

    try: 
        # draw the bounding boxes around the faces
        image_array = utils.draw_bbox(bounding_boxes, image_array)
    except Exception as e:
        print(e)
    
    save_path = output_folder + path.split('/')[-1]
    cv2.imwrite(save_path, image_array)


def color_name(i):
    """Provides human readable color name, used in CSV file"""

    # needs improvement
    match i:
        case 0:
            return 'red'   
        case 1:
            return 'green'
        case 2:
            return 'blue' 
        case _:
            return 'black'


def main():
    """Code execution"""

    # set path to the folder holding the images for analysis
    dir_images = "/path/to/images/to/analyze"
    image_paths = utils.folder_files(dir_images)

    # path to output the annotated images
    # output folder needs "/" at end of path
    output_folder = "/path/to/output/folder/"
    # name of output data file
    output_csv = "output-1.csv"

    # set the minimum detection confidence, 0.0 - 1.0
    min_detection_confidence = 0.35

    for path in image_paths[0::1]:

        results = []
        
        # call Google Vision API, filter for confidence, and return found faces
        faces = process_image(path, min_detection_confidence)

        # print to console for quick debug
        print_debug(faces, path)

        # if no faces found
        if len(faces) == 0:
            log_face_not_found(len(faces), path, results)
        # if faces found
        else:
            for i in range(0, len(faces)):
                log_face(faces[i], color_name(i), len(faces), path, results)     

        # save image to output folder
        save_image(faces, path, output_folder)
        # save csv to output folder
        save_csv(results, output_folder, output_csv)


if __name__ == "__main__":
    main()



import cv2
import time
import dsfd  # Import only the DSFD detector
from utils import draw_predict

def detect_faces_in_image(image_path, save=False, blur_faces=False):
    """
    Function to detect faces in a single image using DSFD.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[Error] Could not read image from {image_path}")
        return

    # Initialize the DSFD detector
    detector = dsfd

    # Perform face detection
    start_time = time.time()
    faces = detector.detect_faces(image)
    detection_time = time.time() - start_time

    print(f"[INFO] Detection completed in {detection_time:.4f} seconds")
    print(f"[INFO] Number of faces detected: {len(faces)}")
    print('#' * 60)

    face_count = 0

    # Loop through each detected face and save the cropped face
    for face in faces:
        bbox = face.bbox
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Crop the face region from the image
        cropped_face = image[y1:y2, x1:x2]

        # Save the cropped face
        face_count += 1
        output_face_path = f"output_face_{face_count}.jpg"
        cv2.imwrite(output_face_path, cropped_face)
        print(f"[INFO] Cropped face saved as {output_face_path}")

        # Optionally, draw bounding boxes on the original image (if blur_faces is False)
        draw_predict(image, face.conf, x1, y1, x2, y2, blur_faces, face.name)

    # Display the original image with bounding boxes (if any)
    cv2.imshow('DSFD Face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the output image with bounding boxes if needed
    if save:
        output_path = image_path.replace('.jpg', '_output.jpg')
        cv2.imwrite(output_path, image)
        print(f"[INFO] Output image with bounding boxes saved as {output_path}")


if __name__ == '__main__':
    # Modify this line to the path of your image file
    image_path = 'unnamed.jpg'
    detect_faces_in_image(image_path, save=True, blur_faces=False)

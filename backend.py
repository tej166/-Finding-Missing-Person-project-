import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

def extract_embeddings(image1_path, image2_path):

    # Initialize the MTCNN face detector

    detector = MTCNN()


    # Initialize the FaceNet model

    model = InceptionResnetV1(pretrained='vggface2').eval()


    # Load the images using the MTCNN face detector

    image1 = cv2.imread(image1_path)

    image2 = cv2.imread(image2_path)


    # Convert the images to RGB

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)


    # Detect faces in the input images

    faces1 = detector.detect(image1)

    faces2 = detector.detect(image2)


    # Extract the embeddings for each face using the FaceNet model

    embeddings1 = []

    for face in faces1:

        face = np.array(face, dtype=np.uint8)

        face = np.float32(face) / 255.0

        face = cv2.resize(face, (96, 96))

        face_tensor = torch.from_numpy(face).float().unsqueeze(0)

        embedding = model(face_tensor).detach().numpy()

        embeddings1.append(embedding)


    embeddings1 = np.mean(embeddings1, axis=0)


    embeddings2 = []

    for face in faces2:

        face = np.array(face, dtype=np.uint8)

        face = np.float32(face) / 255.0

        face = cv2.resize(face, (96, 96))

        face_tensor = torch.from_numpy(face).float().unsqueeze(0)

        embedding = model(face_tensor).detach().numpy()

        embeddings2.append(embedding)


    embeddings2 = np.mean(embeddings2, axis=0)


    return embeddings1, embeddings2

def compare_faces(embedding1, embedding2, threshold=0.4):
    distance = np.linalg.norm(embedding1 - embedding2)
    if distance < threshold:
        return True
    else:
        return False

if __name__ == '__main__':
    # Load the images
    image1_path = 'C:\\Users\\Acer\\Desktop\\Folder\\Finding-Missing-Person-project-main\\static\\victim_images\\Muskan02.jpg'
    image2_path = 'C:\\Users\\Acer\\Desktop\\Folder\\Finding-Missing-Person-project-main\\static\\user_images\\Mark1.jpg'
    embedding1, embedding2 = extract_embeddings(image1_path, image2_path)
    result = compare_faces(embedding1, embedding2)
    print(f'Result: {result}')





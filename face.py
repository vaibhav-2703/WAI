import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from tkinter import Tk, simpledialog, messagebox
from PIL import Image
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# =======================
# Configuration Settings
# =======================
KNOWN_FACES_PATH = 'known_faces.pkl'
SCALER_PATH = 'scaler.pkl'
CLASSIFIER_PATH = 'svm_classifier.pkl'
LABEL_DICT_PATH = 'label_dict.pkl'
SIMILARITY_THRESHOLD = 0.95

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# =======================
# Initialize Models
# =======================
mtcnn = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# =======================
# Utility Functions
# =======================

def list_available_cameras(max_cameras=5):
    """Lists available camera indices."""
    available_cameras = []
    for index in range(max_cameras):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(index)
            cap.release()
    return available_cameras

def select_camera():
    """Prompts the user to select a camera."""
    Tk().withdraw()  # Hide the root window
    available_cameras = list_available_cameras()

    if not available_cameras:
        messagebox.showerror("No Cameras Found", "No webcams were detected on this system.")
        logging.error("No webcams found.")
        return None

    camera_selection = simpledialog.askinteger(
        "Select Camera",
        f"Available Cameras: {available_cameras}\nEnter camera index to use:"
    )

    if camera_selection not in available_cameras:
        messagebox.showerror("Invalid Selection", f"Camera index {camera_selection} is not available.")
        logging.error(f"Invalid camera index selected: {camera_selection}")
        return None

    return camera_selection

def load_file(file_path, description):
    """Load a file and handle errors."""
    if not os.path.exists(file_path):
        logging.error(f"{description} file not found: {file_path}")
        return None
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_file(data, file_path, description):
    """Save a file and handle errors."""
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"{description} saved successfully at {file_path}")
    except Exception as e:
        logging.error(f"Error saving {description}: {e}")

# =======================
# Core Functions
# =======================

def generate_embeddings(data_path):
    """Generate embeddings from face images."""
    logging.info("Generating embeddings...")
    embeddings, names = [], []
    for person in os.listdir(data_path):
        person_path = os.path.join(data_path, person)
        if not os.path.isdir(person_path):
            continue
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            try:
                image = Image.open(image_path).convert('RGB')
                face = mtcnn(image)
                if face is not None:
                    embedding = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()
                    embeddings.append(embedding.flatten())
                    names.append(person)
            except Exception as e:
                logging.warning(f"Error processing {image_path}: {e}")
    save_file({'embeddings': embeddings, 'names': names}, KNOWN_FACES_PATH, "Known faces")
    return np.array(embeddings), names

def train_model(data_path):
    """Train an SVM classifier on generated embeddings."""
    embeddings, names = generate_embeddings(data_path)
    if embeddings.size == 0:
        logging.error("No embeddings generated. Ensure images are correctly formatted.")
        return

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    save_file(scaler, SCALER_PATH, "Scaler")

    label_dict = {name: idx for idx, name in enumerate(set(names))}
    labels = np.array([label_dict[name] for name in names])

    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(embeddings_scaled, labels)
    save_file(classifier, CLASSIFIER_PATH, "SVM classifier")
    save_file(label_dict, LABEL_DICT_PATH, "Label dictionary")

def recognize_faces():
    """Perform real-time face recognition using webcam."""
    scaler = load_file(SCALER_PATH, "Scaler")
    classifier = load_file(CLASSIFIER_PATH, "SVM classifier")
    label_dict = load_file(LABEL_DICT_PATH, "Label dictionary")
    if not scaler or not classifier or not label_dict:
        logging.error("Failed to load required files. Ensure training is complete.")
        return

    label_dict_inv = {v: k for k, v in label_dict.items()}
    camera_index = select_camera()
    if camera_index is None:
        logging.info("No camera selected. Exiting...")
        return

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logging.error(f"Could not access the webcam at index {camera_index}.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to capture frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)
        boxes, _ = mtcnn.detect(pil_frame)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                face = pil_frame.crop((x1, y1, x2, y2)).resize((160, 160))
                face_tensor = mtcnn.face_detector.transform(face).to(device)
                embedding = resnet(face_tensor.unsqueeze(0)).detach().cpu().numpy()
                embedding_scaled = scaler.transform(embedding)
                prediction = classifier.predict_proba(embedding_scaled)[0]
                max_prob = np.max(prediction)
                predicted_label = np.argmax(prediction)

                name = label_dict_inv[predicted_label] if max_prob >= SIMILARITY_THRESHOLD else "Unknown"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({max_prob:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =======================
# Main Execution
# =======================
if __name__ == "__main__":
    action = input("Enter 'train' to train the model or 'recognize' to start recognition: ").strip().lower()
    if action == 'train':
        data_path = input("Enter the path to the dataset: ").strip()
        train_model(data_path)
    elif action == 'recognize':
        recognize_faces()
    else:
        logging.error("Invalid action. Use 'train' or 'recognize'.")

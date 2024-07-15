import cv2
import numpy as np
from skimage import measure, morphology
from skimage.measure import regionprops
from torchvision import transforms
from PIL import Image
import torch

# Check if GPU is available and use it; otherwise, fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def remove_black_border(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding
    _, binary_image = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)
    # Label connected components
    labels = measure.label(binary_image, connectivity=2, background=0)
    # Find the largest connected component
    largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
    # Remove small objects (this step might not be necessary as it's not correctly applied)
    binary_image = morphology.remove_small_objects(labels, min_size=2)
    # Isolate the largest connected component
    binary_image = (binary_image == largest_label).astype(np.uint8) * 255
    # Get properties of labeled regions
    props = regionprops(binary_image)
    # Extract bounding box of the largest component
    boundingBox = props[0].bbox  # Assuming the largest connected component is the first one
    # Crop the image to the bounding box
    image = image[boundingBox[0]:boundingBox[2], boundingBox[1]:boundingBox[3]]
    return image

# Define the image transformations for disc model
preprocess_disc = transforms.Compose([
    transforms.Resize((650, 650)),
    transforms.ToTensor(),
])

# Define the image transformations for fovea model
preprocess_fovea = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define a function to get the disc mask for an image
def get_mask_for_disc(image):
    # Ensure the image is a PIL Image
    if isinstance(image, np.ndarray):
        # Convert NumPy array to PIL Image
        image = Image.fromarray(image)
    # Ensure the image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Preprocess the image
    image_tensor = preprocess_disc(image).unsqueeze(0).to(device)  # Add batch dimension and send to device
    # Pass the image through the model
    with torch.no_grad():
        output = model_disc(image_tensor)
    # Get the mask from the output
    mask = output['out'].argmax(dim=1).squeeze().cpu().numpy()
    return mask

# Define a function to get the fovea mask for an image
def get_mask_for_fovea(image):
    # Ensure the image is a PIL Image
    if isinstance(image, np.ndarray):
        # Convert NumPy array to PIL Image
        image = Image.fromarray(image)
    # Ensure the image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Preprocess the image
    image_tensor = preprocess_fovea(image).unsqueeze(0).to(device)  # Add batch dimension and send to device
    # Pass the image through the model
    with torch.no_grad():
        output = model_fovea(image_tensor)
    # Get the mask from the output
    mask = output['out'].argmax(dim=1).squeeze().cpu().numpy()
    return mask

#  return angle_deg
def calculate_torsion_angle(ellipse_angle):

    angle_deg = ellipse_angle
    if angle_deg > 90:
        angle_deg -= 180

    return angle_deg

def display_torsion_angle(image, angle_deg):
    angle_text = f"Torsion: {angle_deg:.1f}"
    cv2.putText(image, angle_text, (image.shape[1] // 3, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

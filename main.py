# My imports
from utils import *

# Standard imports
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from PIL import Image 
import configparser

# Read the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Set paths
pathIn = config['paths']['pathIn']
pathOut = config['paths']['pathOut']

# Set the preprocessing parameters
writeResults = config.getboolean('options', 'writeResults')

# Display, set to 1 to display processed images in the notebook, else 0
display = False

# Sort the files in the image directory
files = os.listdir(pathIn); files.sort()

# Check if GPU is available and use it; otherwise, fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained MobileNetV2 model
mobilenet_v2 = mobilenet_v2(pretrained=True)

# Create a new Deeplabv3 model with a modified MobileNetV2 backbone
class DeepLabV3MobileNetV2(nn.Module):
    def __init__(self):
        super(DeepLabV3MobileNetV2, self).__init__()
        self.backbone = nn.Sequential(*list(mobilenet_v2.features.children()))
        self.classifier = DeepLabHead(1280, 2)

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        x = self.classifier(x)
        x = nn.functional.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return {'out': x}

# Load disc model
model_disc = torch.load('model_disc.pth')
model_disc = model_disc.to(device) 
model_disc.eval()

# Load fovea model
model_fovea = torch.load('model_fovea.pth')
model_fovea = model_fovea.to(device) 
model_fovea.eval()

# Define a function to get the disc mask for an image
def get_mask_for_disc(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_tensor = preprocess_disc(image).unsqueeze(0).to(device) 
    with torch.no_grad():
        output = model_disc(image_tensor)
    mask = output['out'].argmax(dim=1).squeeze().cpu().numpy()
    return mask

# Define a function to get the fovea mask for an image
def get_mask_for_fovea(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_tensor = preprocess_fovea(image).unsqueeze(0).to(device) 
    with torch.no_grad():
        output = model_fovea(image_tensor)
    mask = output['out'].argmax(dim=1).squeeze().cpu().numpy()
    return mask

# Prepare empty dataframe to store results
results = pd.DataFrame(columns=['Filename', 'Ovality', 'Torsion'])

# Process images  
for i in range(len(files)):

    try:

        path = os.path.join(pathIn, files[i])  
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (650, int(image.shape[0] * (650 / image.shape[1]))))
        image = remove_black_border(image)

        # Process masks
        disc = cv2.resize(get_mask_for_disc(image).astype(np.uint8), (image.shape[1], image.shape[0]))
        fovea = cv2.resize(get_mask_for_fovea(image).astype(np.uint8), (image.shape[1], image.shape[0]))

        # Calculate the centroid of the disc mask
        M_disc = cv2.moments(disc)
        if M_disc["m00"] != 0:
            disc_center_x = int(M_disc["m10"] / M_disc["m00"])
            disc_center_y = int(M_disc["m01"] / M_disc["m00"])
        else:
            disc_center_x = 0
            disc_center_y = 0

        # Calculate the centroid of the fovea mask
        M_fovea = cv2.moments(fovea)
        if M_fovea["m00"] != 0:
            fovea_center_x = int(M_fovea["m10"] / M_fovea["m00"])
            fovea_center_y = int(M_fovea["m01"] / M_fovea["m00"])
        else:
            fovea_center_x = 0
            fovea_center_y = 0

        # Calculate the angle to rotate
        dy = fovea_center_y - disc_center_y
        dx = fovea_center_x - disc_center_x
        angle = np.arctan(dy / dx) * (180 / np.pi)

        # Calculate the center of the image for rotation
        image_center = (image.shape[1]//2, image.shape[0]//2)

        # Rotate around the image center
        rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1)

        # Apply the rotation to the image
        image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

        # Apply the rotation to the masks
        disc = cv2.warpAffine(disc, rotation_matrix, (disc.shape[1], disc.shape[0]))
        fovea = cv2.warpAffine(fovea, rotation_matrix, (fovea.shape[1], fovea.shape[0]))

        # Recalculate the centroid of the disc mask
        M_disc = cv2.moments(disc)
        if M_disc["m00"] != 0:
            disc_center_x = int(M_disc["m10"] / M_disc["m00"])
            disc_center_y = int(M_disc["m01"] / M_disc["m00"])
        else:
            disc_center_x = 0
            disc_center_y = 0

        # Recalculate the centroid of the fovea mask
        M_fovea = cv2.moments(fovea)
        if M_fovea["m00"] != 0:
            fovea_center_x = int(M_fovea["m10"] / M_fovea["m00"])
            fovea_center_y = int(M_fovea["m01"] / M_fovea["m00"])
        else:
            fovea_center_x = 0
            fovea_center_y = 0

        # Create a copy of the image to annotate
        image_with_annotations = image.copy()

        # Process and annotate disc
        contours, _ = cv2.findContours(disc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if len(largest_contour) >= 5:
                ellipse = cv2.fitEllipse(largest_contour)
                cv2.ellipse(image_with_annotations, ellipse, (0, 255, 0), 2)
                ellipse_center = (int(ellipse[0][0]), int(ellipse[0][1]))

        # Process and annotate fovea
        contours, _ = cv2.findContours(fovea, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                fovea_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                cv2.drawMarker(image_with_annotations, fovea_center, (255, 0, 0), cv2.MARKER_TILTED_CROSS, 20, 2)

        # Draw line between ellipse center and fovea center
        if 'ellipse_center' in locals() and 'fovea_center' in locals():
            cv2.line(image_with_annotations, ellipse_center, fovea_center, (255, 255, 0), 2)

            # Calculate and display torsion angle
            angle_deg = calculate_torsion_angle(ellipse[2])
            if fovea_center_x > disc_center_x:
                angle_deg = angle_deg * -1
            display_torsion_angle(image_with_annotations, angle_deg)

        # Draw the major axis of the ellipse
        # Calculate the endpoints of the minor axis
        minor_axis_length = ellipse[1][1]
        angle_of_rotation_rad_perpendicular = np.radians(ellipse[2] + 90)  
        # Calculate the offsets from the center to the endpoints of the minor axis
        dx_minor = (minor_axis_length / 2) * np.cos(angle_of_rotation_rad_perpendicular)
        dy_minor = (minor_axis_length / 2) * np.sin(angle_of_rotation_rad_perpendicular)
        # Calculate the endpoints for the minor axis
        endpt1_minor = (int(ellipse_center[0] - dx_minor), int(ellipse_center[1] - dy_minor))
        endpt2_minor = (int(ellipse_center[0] + dx_minor), int(ellipse_center[1] + dy_minor))
        # Draw the minor axis line
        cv2.line(image_with_annotations, endpt1_minor, endpt2_minor, (255, 255, 0), 2) 
        
        start_point_vertical = (disc_center_x, 0)
        end_point_vertical = (disc_center_x, image_with_annotations.shape[0])

        # Draw the vertical meridian line
        cv2.line(image_with_annotations, start_point_vertical, end_point_vertical, (255, 255, 0), 2) 

        # Ellipse ovality
        ovality = round(ellipse[1][0] / ellipse[1][1], 2)
        cv2.putText(image_with_annotations, f"Ovality: {ovality}", (image.shape[1] // 3, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # Display the final image
        if display == True:
            plt.imshow(image_with_annotations)
            plt.title(files[i])
            plt.axis('off')
            plt.show()

        # Store results
        filename = files[i]
        new_row = pd.DataFrame({'Filename': [filename], 'Ovality': [ovality], 'Torsion': [angle_deg]})
        results = pd.concat([results, new_row], ignore_index=True)
        print(f"Processed image {i+1}: {filename}")

        # Save the images with annotations
        #pathOut2 = f'{pathOut}{filename}.png'
        filename_wo_ext = os.path.splitext(filename)[0]  # Remove the existing extension
        pathOut2 = f'{pathOut}{filename_wo_ext}.png'
        
        # Convert BGR to RGB
        image_with_annotations = cv2.cvtColor(image_with_annotations, cv2.COLOR_BGR2RGB)

        # Save the image in RGB format
        if writeResults == True:
            cv2.imwrite(pathOut2, image_with_annotations)

    except Exception as e:
        print(f"Error processing image {i+1}: {files[i]}")
        print(e)

        # Store results
        filename = files[i]
        new_row = pd.DataFrame({'Filename': [filename], 'Ovality': 'NA', 'Torsion': 'NA'})
        results = pd.concat([results, new_row], ignore_index=True)
        
        continue

# Save results to a CSV file
if writeResults == True:    
    results.to_csv('./results/results.csv', index=False)

# Code is taken from https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
# This uses a k-means approach to identify the dominant colors of a frame in the video stream

# The rest of the code is based on the previous camera task, which cites https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
# The change in surrounding brightness affected the non-phone object more significantly than the phone
# This is likely due to the ambient light affecting the object, which changes its color when the brightness changes
# The phone, however, as a source of light is less prone to these ambient light changes



import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def find_histogram(clt):
    """
    Create a histogram with k clusters
    :param: clt (KMeans object)
    :return: hist (normalized histogram)
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()  # Normalize the histogram
    return hist

def plot_colors2(hist, centroids):
    """
    Create a bar showing dominant colors.
    :param: hist (color distribution from histogram)
    :param: centroids (dominant color centers)
    :return: bar (image representing the color bar)
    """
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    return bar

# Function to crop the image to focus on the center region
def crop_center(img, crop_size=20):
    """
    Crop the center of the image for analysis.
    :param: img (original frame from camera)
    :param: crop_size (size of the cropped region)
    :return: cropped image
    """
    h, w = img.shape[:2]
    center_y, center_x = h // 2, w // 2
    cropped = img[center_y - crop_size:center_y + crop_size, 
                  center_x - crop_size:center_x + crop_size]
    return cropped

# Function to process the image and extract dominant colors
def process_frame(frame, clt):
    """
    Process the frame to extract dominant colors.
    :param: frame (original frame)
    :param: clt (KMeans object)
    :return: bar (color bar image), HSV centers (dominant colors in HSV)
    """
    cropped_img = crop_center(frame)
    img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    img_reshaped = img_rgb.reshape((img_rgb.shape[0] * img_rgb.shape[1], 3))
    
    # Fit KMeans to the reshaped image
    clt.fit(img_reshaped)

    # Get histogram and dominant color centers
    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)
    
    # Convert cluster centers to HSV
    normalized_rgb = clt.cluster_centers_.astype('uint8')
    hsv_centers = cv2.cvtColor(np.array([normalized_rgb]), cv2.COLOR_RGB2HSV)
    
    return bar, hsv_centers

# Initialize video capture and KMeans clustering
cap = cv2.VideoCapture(0)
clt = KMeans(n_clusters=3)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Process the frame and extract the color bar and HSV values
    bar, hsv_centers = process_frame(frame, clt)

    # Display the original and cropped images
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Cropped Region', crop_center(frame))

    # Print cluster centers in both RGB and HSV
    print("RGB values of the 3 cluster centers:\n", clt.cluster_centers_)
    print("HSV values of the cluster centers:\n", hsv_centers)

    # Plot the dominant colors bar
    plt.axis("off")
    plt.imshow(bar)
    plt.show()

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup resources
cap.release()
cv2.destroyAllWindows()

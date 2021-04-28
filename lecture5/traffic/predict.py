import tensorflow as tf
import numpy as np
import cv2
import sys

from traffic import IMG_WIDTH, IMG_HEIGHT


def main():
    # Check command-line arguments
    if len(sys.argv) != 3:
        sys.exit("Usage: python traffic.py image model.h5")

    # Read, resize and store resized image as numpy array
    img = np.array([cv2.resize(cv2.imread(sys.argv[1]), (IMG_WIDTH, IMG_HEIGHT))])

    # Get a pretrained neural network
    model = tf.keras.models.load_model(sys.argv[2])
    
    # Make prediction and assign sign a type
    prediction = model.predict(img)
    sign_index = np.argmax(prediction, axis=-1)[0]
    sign_type = SIGNS_INDEXES[sign_index]
    
    # Print the result
    print(f"Sign '{sign_type}'. Sign index: {sign_index}")


# Match signs indexes with their types
SIGNS_INDEXES = {
    0: "20km/h speed limit",
    1: "30km/h speed limit",
    2: "50km/h speed limit",
    3: "60km/h speed limit",
    4: "70km/h speed limit",
    5: "80km/h speed limit",
    6: "End of 80km/h speed limit",
    7: "100km/h speed limit",
    8: "120km/h speed limit",
    9: "Overtaking prohibited",
    10: "Truck overtaking prohibited",
    11: "Intersection with a minor road",
    12: "Main road",
    13: "Give way",
    14: "Stop",
    15: "Traffic prohibited",
    16: "Truck traffic prohibited",
    17: "Entry prohibited",
    18: "Hazardous area",
    19: "Dangerous left turn",
    20: "Dangerous right turn",
    21: "Multiple turns",
    22: "Rough road",
    23: "Slippery road",
    24: "Road narrowing",
    25: "Road maintenance",
    26: "Traffic light regulation",
    27: "Pedestrian traffic prohibited",
    28: "Children",
    29: "Cyclists entry",
    30: "Snowfall",
    31: "Wild animals",
    32: "End of all prohibitions and restrictions",
    33: "Traffic to the right",
    34: "Traffic to the left",
    35: "Traffic straight ahead",
    36: "Traffic straight ahead or to the right",
    37: "Traffic straight ahead or to the left",
    38: "Avoid obstacles on the right side",
    39: "Avoid obstacles on the left side",
    40: "Roundabout movement",
    41: "End of overtaking prohibition",
    42: "End of truck overtaking prohibition"
}


if __name__ == "__main__":
    main()

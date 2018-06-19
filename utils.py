import cv2, os
import numpy as np
import matplotlib.image as mpimg


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 120,160, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    return mpimg.imread(os.path.join(data_dir, image_file))


def crop(image):
    
    return image[60:, :, :] 

def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to BGR
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image


def choose_image(data_dir, object_image):
    
    return load_image(data_dir, object_image)


def random_flip(image, steering_angle):
    """
    Randomly flip the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        
        steering_angle = flip(steering_angle)
    return image, steering_angle

def flip(steering_angle):
    steering_angle[0]=steering_angle[0]
    steering_angle[1]=-(steering_angle[1])

    return steering_angle



def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle = translate(steering_angle,trans_x,trans_y)
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle

def translate(steers, tx,ty):
    if tx >= 0:
        steers[0] = steers[0]
        steers[1] = steers[1] + tx*0.004

    else:
        steers[0] = steers[0] 
        steers[1] = steers[1] - tx*0.004

    if ty >= 0:
        steers[0] = steers[0] + ty*0.002
        steers[1] = steers[1]
    else:
        steers[0] = steers[0] - ty*0.002
        steers[1] = steers[1]
        

    for i in range(len(steers)):
        if steers[i]>1:
            steers[i] = 1
        if steers[i]<-1:
            steers[i] = -1

    return steers


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augument(data_dir, object_image,steering_angle, range_x=320, range_y=100):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image = choose_image(data_dir, object_image)
    #image1 = preprocess(image)
    image,steering_angle = random_flip(image,steering_angle)
    #image,steering_angle = random_translate(image,steering_angle, range_x, range_y)
    image = random_brightness(image)
    return image,steering_angle


def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    """
    Generate training image give image paths and associated steering angles
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty([batch_size,2])
    while True:
       # print("E")
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            #print(index)
            object_image = image_paths[index]
            steering_angle = steering_angles[index]
            # argumentation
            if is_training and np.random.rand()<0.5:
                
                image, steering_angle = augument(data_dir, object_image,steering_angle)
                #print("P_T")
            else:
                image = load_image(data_dir, object_image)
                #print("P_V") 
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers


#!/usr/bin/env python
# coding: utf-8

# In[3]:

import streamlit as st
import requests
import numpy as np
import cv2
from PIL import Image
import glob
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import segmentation_models as sm

st.title("Future Vision Transport 🚘")
st.header("Prédiction d'une image via appel API")
st.info("""
On prédit le modèle grâce à une liste d'images de test.
Il suffit de sélectionner l'image grâce au menu glissant.
Il faut ensuite cliquer sur le bouton "Prédire" pour afficher le masque
prédit.
""")

IMG_WIDTH_HEIGHT = (256, 512)

val_imgs = glob.glob("./data/test/images/*.png")
val_masks = glob.glob("./data/test/masks/*.png")

val_imgs = sorted(val_imgs, key=str)
val_masks = sorted(val_masks, key=str)

image_id = st.slider("Choisissez l'id de l'image", 0, 400, 0)

color_image = Image.open(val_imgs[image_id])
mask_image = Image.open(val_masks[image_id])

col1, col2 = st.columns(2)
with col1:
    st.subheader("Image d'entrée en couleur")
    st.image(color_image)

with col2:
    st.subheader("Masque de l'image")
    st.image(mask_image)
    
# Function to compute dice loss
def dice_loss(y_true, y_pred):
    y_true = tensorflow.cast(y_true, tensorflow.float32)
    y_pred = tensorflow.math.sigmoid(y_pred)
    numerator = 2 * tensorflow.reduce_sum(y_true * y_pred)
    denominator = tensorflow.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator

def images_file():
    val_input_dir = './data/test/images/'
    val_target_dir = './data/test/masks/'

    val_input_path = sorted(glob.glob(val_input_dir + '/*.png'))
    val_target_path = sorted(glob.glob(val_target_dir +'/*.png'))
    
    return val_input_path, val_target_path

def image_quantites():
    images, masks = images_file()
    images_quantite = len(images)
    
    return images_quantite

# Function to compute dice coefficient
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice
# Function to compute balanced cross entropy
def balanced_cross_entropy(beta):
        def loss(y_true, y_pred):
            weight_a = beta * tensorflow.cast(y_true, tensorflow.float32)
            weight_b = (1 - beta) * tensorflow.cast(1 - y_true, tensorflow.float32)

            o = (
                        tensorflow.math.log1p(
                            tensorflow.exp(-tensorflow.abs(y_pred))
                        ) + tensorflow.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b
            return tensorflow.reduce_mean(o)

        return loss
    
def iou(y_true, y_pred):
        iou_metric = tensorflow.numpy_function(raw_iou, [y_true, y_pred], tensorflow.float32)
        return iou_metric

model = keras.models.load_model(
        './model/model_unet_dice_aug.h5', 
    custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef,}
    )
    
content_type = 'image/png'
headers = {'content-type': content_type}


def rgb_seg_img(seg_arr, n_classes):
    
    class_colors = {
        0:(0,0,0),        # void
        1:(128, 64, 128), # flat
        2:(102,102,156),  # construction
        3:(153,153,153),  # object
        4:(107, 142, 35), # nature
        5:(70,130,180),   # sky
        6:(255, 0, 0),    # human
        7:(0, 0, 142)     # vehicle
    }
    
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += ((seg_arr_c) * (class_colors[c][0])).astype('uint8') # R
        seg_img[:, :, 1] += ((seg_arr_c) * (class_colors[c][1])).astype('uint8') # G
        seg_img[:, :, 2] += ((seg_arr_c) * (class_colors[c][2])).astype('uint8') # B

    return seg_img.astype('uint8')

def model_predict(img):
    model = sm.Unet('vgg16', classes=8)
    model.load_weights('C:/Users/folyb/Documents/IA/P8/unet_vgg16_aug_weights.h5')
    img = img_to_array(load_img(f'{img}', target_size=(IMG_WIDTH_HEIGHT)))/255
    img = np.expand_dims(img, 0)
    preds = model.predict(img)

    return preds


def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


local_css("style.css")
legend = """<div>
<span class='highlight bold void'>        </span>
<span>void</span>
</div>
<div>
<span class='highlight bold flat'>        </span>
<span>flat</span>
</div>
<div>
<span class='highlight bold construction'>        </span>
<span>construction</span>
</div>
<div>
<span class='highlight bold object'>        </span>
<span>object</span>
</div>
<div>
<span class='highlight bold nature'>        </span>
<span>nature</span>
</div>
<div>
<span class='highlight bold sky'>        </span>
<span>sky</span>
</div>
<div>
<span class='highlight bold human'>        </span>
<span>human</span>
</div>
<div>
<span class='highlight bold vehicle'>        </span>
<span>vehicle</span>
</div>
"""

if st.button("Prédire"):
    with st.spinner("Prédiction en cours..."):
                
        images, masks = images_file()
        #json = request.json

        print("CECI EST LE PRINT DU JSON FINAL")
        print(image_id)
        preds = model_predict(images[image_id])
    
        pred_mask = np.argmax(preds, axis=-1)
   
        pred_mask = np.squeeze(pred_mask)
        
        #pred_data = np.array(pred_mask)
        pred_mask = rgb_seg_img(pred_mask, 8)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Masque prédit")
            st.text(" ")
            #plt.imsave('./static/outputs/colorized_mask.png',  pred_mask, cmap='nipy_spectral_r')
            st.image(pred_mask)
        with col4:
            st.subheader("Légende")
            st.markdown(legend, unsafe_allow_html=True)
            
                               
st.markdown("made by **FOLY BENOIT KUEVIAKOE**")
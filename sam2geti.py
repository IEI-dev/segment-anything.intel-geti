#muti-image dataset
#https://blog.roboflow.com/how-to-use-segment-anything-model-sam/

#default json
json_default = {
    "infos":{
       
    },
    "categories":{
       "label":{
          "labels":[
             {
                "name":"vitb",
                "parent":"",
                "attributes":[
                   
                ]
             }
          ],
          "label_groups":[
             {
                "name":"Instance segmentation labels",
                "group_type":"exclusive",
                "labels":[
                   "vitb"
                ]
             }
          ],
          "attributes":[
             
          ]
       },
       "mask":{
          "colormap":[
             {
                "label_id":0,
                "r":215,
                "g":188,
                "b":94
             }
          ]
       }
    },
    "items":[
       
    ]
 }

#load model
import torch
from segment_anything import sam_model_registry

DEVICE = torch.device('cpu')#'cuda:0'
MODEL_TYPE = "vit_b"
CHECKPOINT_PATH = "./checkpoint/sam_vit_b_01ec64.pth"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
print("load model...")


#mask to polygon
import numpy as np
def convert_mask_to_polygon(mask):
    contours = None
    if int(cv2.__version__.split('.')[0]) > 3:
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[0]
    else:
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[1]
    
    contours = max(contours, key=lambda arr: arr.size)
    if contours.shape.count(1):
        contours = np.squeeze(contours)
    if contours.size < 3 * 2:
        raise Exception('Less then three point have been detected. Can not build a polygon.')
    
    polygon = []
    for point in contours:
        polygon.append(int(point[0]))
        polygon.append(int(point[1]))
    
    print("mask2polygon...")
    return polygon



#replace json_default(items)
import os
import cv2
Image_folder = 'input/dog2'
files = os.listdir(Image_folder)

for file in files:
   items = {
          "id":"dog",
          "annotations":[
             
          ],
          "attr":{
             "has_empty_label":False
          },
          "image":{
             "path":"dog.jpg",
             "size":[
                1365,
                2048
             ]
          },
          "media":{
             "path":""
          }
       }

   IMAGE_PATH = Image_folder+"/"+file
   Image_Name = file.split(".")[0]
   items["id"] = Image_Name
   items["image"]["path"] = file
   im = cv2.imread(IMAGE_PATH)
   h, w, c = im.shape
   items["image"]["size"] = [h, w]
   print("load image["+file+"] ,and generate item(json)...")






   # inference
   import cv2
   from segment_anything import SamAutomaticMaskGenerator

   mask_generator = SamAutomaticMaskGenerator(sam)
   IMAGE_PATH = IMAGE_PATH

   image_bgr = cv2.imread(IMAGE_PATH)
   image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
   results = mask_generator.generate(image_rgb)
   print("inference...")





   #replace json_default(annotation)
   for result in results:
      try:
         masks = result["segmentation"]
         object_mask = np.array(masks, dtype=np.uint8)
         object_mask = cv2.normalize(object_mask, None, 0, 255, cv2.NORM_MINMAX)
         polygon = convert_mask_to_polygon(object_mask)
         annotation = {
                     "id":0,
                     "type":"polygon",
                     "attributes":{
                        
                     },
                     "group":0,
                     "label_id":0,
                     "points":[
                        
                     ],
                     "z_order":0
                  }
         annotation["points"]=polygon
         items["annotations"].append(annotation)
         #json_default["items"].append(items)
      except:
         pass
   json_default["items"].append(items)
   print("generate mask(json)...")






#create datumaro/annotations and datumaro/images
import os
path_annotations = './output/dataset/auto-datumaro/auto-datumaro/annotations/'
path_images = './output/dataset/auto-datumaro/auto-datumaro/images/default/'
os.makedirs(path_annotations, exist_ok=True)
os.makedirs(path_images, exist_ok=True)
print("mkdir dataset...")

#create datumaro/annotations/default.json 
import json
f = open(path_annotations+"default.json","w")
json_default = json.dumps(json_default, indent = 4) 
f.writelines(json_default)
f.close()
print("generate annotations...")

#create datumaro/annotations/default/image.jpg and compress it to zip file 
import shutil
Image_folder = 'input/dog2'
files = os.listdir(Image_folder)
for file in files:
   IMAGE_PATH = Image_folder+"/"+file
   image_name = IMAGE_PATH.split("/")[-1]
   shutil.copyfile(IMAGE_PATH, path_images+image_name)
shutil.make_archive("./output/dataset/auto-datumaro/auto-datumaro-temp", 'zip', "./output/dataset/auto-datumaro/auto-datumaro")
print("generate images and compress zip...")

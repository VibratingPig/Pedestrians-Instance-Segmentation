import pandas as pd
import ast

image_name = 'e8c95d9d060f.png'

df = pd.read_csv('train_image_level.csv')

image_id = image_name.split('.')[0] + "_image"
row = df[df.id == image_id]
# boxes is a pandas series
boxes = row.boxes
for box in boxes:
    list_of_dictionaries = ast.literal_eval(box)
    for boundary_boxes in list_of_dictionaries:
        # box is a string representation of the bounds
        # its a list of dictionaries

        x = boundary_boxes['x']
        y = boundary_boxes['y']
        x_width = boundary_boxes['width']
        y_height = boundary_boxes['height']

df.head()
# Import the function from our utility file
from bbox import draw_bounding_box

# 1. Define your inputs
# Replace this with your actual image path 'x'
image_path_x = "data/color/test/images/color_010_source.png" 

# Set your bounding box variables
box_x = 320
box_y = 356
box_w = 383
box_h = 67

# 2. Call the function
result_image = draw_bounding_box(
    image_path=image_path_x, 
    x=box_x, 
    y=box_y, 
    w=box_w, 
    h=box_h,
    output_path="tests/test_output.png", # Saves a copy to your current folder
    color="red",
    thickness=3
)

# 3. Display the result if the image loaded successfully
if result_image:
    result_image.show()
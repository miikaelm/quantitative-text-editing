from PIL import Image, ImageDraw

def draw_bounding_box(image_path, x, y, w, h, output_path=None, color="red", thickness=2):
    """
    Draws a bounding box on an image.

    Args:
        image_path (str): The path to the source image.
        x (int): The top-left x-coordinate of the bounding box.
        y (int): The top-left y-coordinate of the bounding box.
        w (int): The width of the bounding box.
        h (int): The height of the bounding box.
        output_path (str, optional): Where to save the new image. Defaults to None.
        color (str, optional): Color of the bounding box. Defaults to "red".
        thickness (int, optional): Line thickness. Defaults to 2.

    Returns:
        Image: The PIL Image object with the drawn bounding box.
    """
    try:
        # Load the image
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: The image at {image_path} could not be found.")
        return None

    # Initialize the drawing context
    draw = ImageDraw.Draw(img)

    # Calculate bottom-right coordinates
    x2 = x + w
    y2 = y + h

    # Draw the rectangle
    draw.rectangle([x, y, x2, y2], outline=color, width=thickness)

    # Save the image if an output path is provided
    if output_path:
        img.save(output_path)
        print(f"Saved image with bounding box to {output_path}")

    return img
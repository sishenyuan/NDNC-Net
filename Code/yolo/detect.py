import numpy as np
from ultralytics import YOLO

def detect_nurd(model_path, img):
    # Load a model
    model = YOLO(model_path)

    # Run the model on the image
    results = model(img)
    
    nurd_coordinates = []
    
    # Process results
    for result in results:
        for box in result.boxes.xyxy:
            nurd_coordinates.append((box[0].cpu(), box[2].cpu()))
        # # Save the result image
        # result.save(filename=output_path)
    
    nurd_coordinates = np.array(nurd_coordinates)  

    sorted_indices = np.argsort(nurd_coordinates[:, 0])
    nurd_coordinates = nurd_coordinates[sorted_indices]

    return nurd_coordinates

# Example usage
# nurd_coords = process_image("train/weights/best.pt", "1.png")

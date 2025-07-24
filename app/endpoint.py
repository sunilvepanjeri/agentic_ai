from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
import tempfile
import numpy as np
import cv2
from matplotlib import pyplot as plt


router = APIRouter(tags=["endpoint"], prefix="/api")



@router.post('/image')
async def image_endpoint(image: UploadFile = File(...)):

    with tempfile.TemporaryFile(delete = False) as image_file:
        image_file.write(await image.read())
        temp = image_file.name
    image = cv2.imread(temp, cv2.IMREAD_GRAYSCALE)

    assert image is not None, "image not found"

    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)

    edges = cv2.Canny(blurred_image, 100, 200)

    kernel = np.ones((5, 5), np.uint8)
    cleaned_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output_image, filtered_contours, -1, (0, 255, 0), 2)

    with tempfile.TemporaryFile(delete = False, suffix = ".jpg") as image_file:
        final_output = image_file.name
        cv2.imwrite(final_output, output_image)

    return FileResponse(final_output, media_type="image/jpg")

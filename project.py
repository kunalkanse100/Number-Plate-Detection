import cv2
from easyocr import Reader

# Initialize the camera
camera = cv2.VideoCapture(0)  # 0 indicates the default camera

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()

    # Preprocess the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 10, 200)

    # Find contours and extract the license plate
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            n_plate_cnt = approx
            break

    if len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(n_plate_cnt)
        license_plate = gray[y:y + h, x:x + w]

        # Perform OCR
        reader = Reader(['en'])
        detection = reader.readtext(license_plate)

        # Display results
        if len(detection) == 0:
            text = "Impossible to read the text from the license plate"
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)
        else:
            cv2.drawContours(frame, [n_plate_cnt], -1, (0, 255, 0), 3)
            text = f"{detection[0][1]} {detection[0][2] * 100:.2f}%"
            cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Camera Feed', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()

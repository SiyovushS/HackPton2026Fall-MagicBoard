import cv2
from PIL import Image
from pix2text import Pix2Text

p2t = Pix2Text.from_config(enable_table=False)   # load once, before the loop
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    cv2.imshow("Magic Board - live", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):                                   # space = OCR this frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # OpenCV is BGR, PIL wants RGB
        result = p2t.recognize_text_formula(Image.fromarray(rgb))
        print("OCR:", result)
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
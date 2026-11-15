import colorsys

import cv2
import numpy as np
import pyvirtualcam

# Open the default camera
cam = cv2.VideoCapture(0)
print(cam)
print(cam.read)

width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
dim = (height, width)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (width, height))

with pyvirtualcam.Camera(width=dim[1], height=dim[0], fps=20) as virt_cam:
    print(f"Using virtual camera: {virt_cam.device}")
    # frame = np.zeros((virt_cam.height, virt_cam.width, 3), np.uint8)  # RGB

    while True:
        ret, frame = cam.read()

        # Write the frame to the output file
        out.write(frame)

        # Display the captured frame
        cv2.imshow("Camera", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord("q"):
            break

        i = np.indices(dim)
        brightness = np.sqrt(i[0]**2 + i[1]**2)
        frame[:] += np.stack([brightness, brightness, brightness], axis=-1).astype(np.uint8)
        virt_cam.send(frame)
        virt_cam.sleep_until_next_frame()

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()

import colorsys

import cv2
import numpy as np
import pyvirtualcam

# Open the default camera
cam = cv2.VideoCapture(0)

img = cv2.imread("test.png")
img_height, img_width, img_ch = img.shape

width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
dim = (height, width)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (width, height))

with pyvirtualcam.Camera(width=dim[1], height=dim[0], fps=20) as virt_cam:
    print(f"Using virtual camera: {virt_cam.device}")
    # frame = np.zeros((virt_cam.height, virt_cam.width, 3), np.uint8)  # RGB

    thresh_point = 0

    while True:
        ret, frame = cam.read()

        # Write the frame to the output file
        out.write(frame)

        if cv2.waitKey(1) == ord("k"):
            thresh_point += 16

        if cv2.waitKey(1) == ord("j"):
            thresh_point -= 16

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        edges = cv2.Canny(blurred, 20, 80)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # _, thresh = cv2.threshold(gray, thresh_point, 255, cv2.THRESH_BINARY)
        thresh = dilated_edges
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(frame, contours, -1, (255, 0, 0), 1)
        for c in contours:
            if(len(c) < 4):
                continue

            min_size = pow(64, 2)
            area = cv2.contourArea(c)
            if(area < max(16, min_size)):
                continue

            area_factor = 1.2
            approx = None
            try:
                # This function just fails sometimes...
                approx = cv2.approxPolyN(c, epsilon_percentage=area_factor - 1, nsides=4)
            except cv2.error:
                print("Bad contour " + str(c))
                continue


            approx_area = cv2.contourArea(approx)
            if(4 < approx.size / 2):
                continue

            if(area_factor < approx_area / area):
                # why are you even here?
                continue
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 1)

            dst_pos = np.float32(approx)
            trans = cv2.getPerspectiveTransform(
                np.float32([
                    [0, 0],
                    [img_width, 0],
                    [img_width, img_height],
                    [0, img_height]
                ]),
                dst_pos,
            )

            warped_img = cv2.warpPerspective(img, trans, (width, height))

            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(mask, approx, 255)
            # cv2.fillConvexPoly(mask, approx.reshape(4, 2).astype(np.int32), 255)
            cv2.copyTo(warped_img, mask, frame)



        virt_cam.send(frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Camera", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord("q"):
            break

        virt_cam.sleep_until_next_frame()

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()

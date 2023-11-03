import cv2
import numpy as np
import copy
import os


# mouse callback function
def mouse_callback(event, x, y, flags, param):
    global corner_list
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(corner_list) < 4:
            corner_list.append((x, y))


def Find_Homography(world, camera):
    '''
    given corresponding point and return the homagraphic matrix
    '''
    return


if __name__ == "__main__":
    img_src = cv2.imread("assets/post.png")
    src_H, src_W, _ = img_src.shape
    # print(H,W)
    file_path = "./1/output"
    img_tar = cv2.imread("assets/display.jpg")

    cv2.namedWindow("Interative window")
    cv2.setMouseCallback("Interative window", mouse_callback)
    cv2.setMouseCallback("Interative window", mouse_callback)

    corner_list = []
    while True:
        fig = img_tar.copy()
        key = cv2.waitKey(1) & 0xFF

        if len(corner_list) == 4:
            # implement the inverse homography mapping and bi-linear interpolation
            pass

        # quit
        if key == ord("q"):
            break

        # reset the corner_list
        if key == ord("r"):
            corner_list = []
        # show the corner list
        if key == ord("p"):
            print(corner_list)
        cv2.imshow("Interative window", fig)

    cv2.imwrite(os.path.join(file_path, "homography.png"), fig)
    cv2.destroyAllWindows()

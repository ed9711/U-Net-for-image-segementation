import cv2 as cv
import os


def drawOutline(img, mask, batch):
    dst = cv.Canny(mask, 50, 200, None, 3)
    img_np = img.cpu().detach().numpy()[:, :, :, :]
    img_np = img_np.squeeze().transpose(1, 2, 0)
    result = cv.bitwise_and(img_np, img_np, mask=cv.bitwise_not(dst))
    # cv.imshow("asdf", result)
    # cv.waitKey()
    result = cv.convertScaleAbs(result, alpha=(255.0))
    # result = cv.resize(result, (500, 500), interpolation = cv.INTER_AREA)
    if not os.path.exists("./result1/"):
        os.makedirs("./result1/")
    cv.imwrite("./result1/{}.png".format(batch), result)

    return result

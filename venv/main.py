import cv2
import image_process as pic
import time

font = cv2.FONT_HERSHEY_SIMPLEX  # 设置字体
size = 0.5  # 设置大小
i = 4
cnt = 1
width, height = 300, 300  # 设置拍摄窗口大小
x0, y0 = 600, 100  # 设置选取位置

cap = cv2.VideoCapture(0)  # 开摄像头
count = 0
# image_path = 'image/0-0.png'
# image = cv2.imread(image_path)
# image = pic.imageProcess(image)
# path2 = './data/data_train/0-0_train.png'
# print(path2)
# cv2.imwrite(path2, image)
#
#
# for i in range(9):
#         for j in range(50):
#             path = image_path + str(j) + '-' + str(i) + '.png'
#             print(path)
#             image = cv2.imread(path)
#             image = pic.imageProcess(image)
#             path2 = 'data_train/' + str(i) + '-' + str(j) + '_train.png'
#             print(path2)
#             cv2.imwrite(path2, image)


if __name__ == "__main__":
    while (1):


        ret, frame = cap.read()  # 读取摄像头的内容
        frame = cv2.flip(frame, 2)
        cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0))  # 画出截取的手势框图
        roi = frame[y0:y0 + height, x0:x0 + width]  # 获取手势框图
        key = cv2.waitKey(50) & 0xFF  # 按键判断并进行一定的调整
        cv2.imshow('frame', frame)  # 播放摄像头的内容


        if key == ord('s'): # 开始录制
            imagePath = 'image/1-'+ str(count) + '.png'
            print("save:" + imagePath)
            cv2.imwrite(imagePath, roi)
            count += 1
            while(count == 50):
                break
        # 按'q'键退出录像
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()  # 关闭所有窗口







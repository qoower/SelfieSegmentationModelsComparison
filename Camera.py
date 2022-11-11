from PIL import Image
import cv2
import time
import mediapipe as mp
import numpy as np

import threading
import tensorflow as tf
from models.SINet import *
from keras.models import load_model

class SelfieSegPN:
    def __init__(self, width=320, height=240):
        # Initialize tflite-interpreter
        self.width = width
        self.height = height

        self.interpreter = tf.lite.Interpreter(model_path="models/portrait_video.tflite")  # Use 'tf.lite' on recent tf versions
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        h, w = self.input_details[0]['shape'][1:3]
        self.dim = w
        self.prev = np.zeros((self.dim, self.dim, 1))

    def normalize(self, imgOri, scale=1, mean=[103.94, 116.78, 123.68], val=[0.017, 0.017, 0.017]):
        img = np.array(imgOri.copy(), np.float32) / scale
        return (img - mean) * val

    def seg(self, frame):
        img = np.array(frame)
        img = cv2.resize(img, (self.dim, self.dim))
        img = img.astype(np.float32)

        img = self.normalize(img)

        # Add prior as fourth channel
        img = np.dstack([img, self.prev])
        img = img[np.newaxis, :, :, :]

        # Invoke interpreter for inference
        self.interpreter.set_tensor(self.input_details[0]['index'], np.array(img, dtype=np.float32))
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]['index'])
        out = out.reshape(self.dim, self.dim, 1)
        out = (255 * out).astype("uint8")
        _, out = cv2.threshold(out, 128, 255, cv2.THRESH_BINARY)
        self.prev = (out / 255.0).astype("float32")

        mask = cv2.resize(out, (self.width, self.height))
        _, mask = cv2.threshold(mask, 128, 255, 0)

        return mask

class SelfieSegMNV3:
    def __init__(self, width=320, height=240):
        self.width = width
        self.height = height
        self.dim = 224
        self.model = load_model("models/munet_mnv3_wm05.h5")

    def seg(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        image = image.resize((self.dim, self.dim), Image.ANTIALIAS)
        img = np.float32(np.array(image) / 255.0)
        img = img[:, :, 0:3]

        # Reshape input and threshold output
        out = self.model.predict(img.reshape(1, self.dim, self.dim, 3))
        out = np.float32((out > 0.5)).reshape(self.dim, self.dim)
        mask = (255 * out).astype("uint8")

        mask = cv2.resize(mask, (self.width, self.height))
        _, mask = cv2.threshold(mask, 128, 255, 0)

        return mask
        
class SelfieSegMNV2:
    def __init__(self, width=320, height=240):
        # Initialize tflite-interpreter
        self.width = width
        self.height = height
        self.interpreter = tf.lite.Interpreter(model_path="models/deconv_fin_munet.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape'][1:3]

        # Image overlay
        self.overlay = np.zeros((self.input_shape[0], self.input_shape[1], 3), np.uint8)
        self.overlay[:] = (127, 0, 0)

    def seg(self, frame):
        # BGR->RGB, CV2->PIL
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        # Resize image
        image = image.resize(self.input_shape, Image.ANTIALIAS)

        # Normalization
        image = np.asarray(image)
        prepimg = image / 255.0
        prepimg = prepimg[np.newaxis, :, :, :]

        # Segmentation
        self.interpreter.set_tensor(self.input_details[0]['index'], np.array(prepimg, dtype=np.float32))
        self.interpreter.invoke()
        outputs = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Process the output
        output = np.uint8(outputs[0] > 0.5)
        res = np.reshape(output, self.input_shape)
        mask = Image.fromarray(np.uint8(res), mode="P")
        mask = np.array(mask.convert("RGB")) * self.overlay
        mask = cv2.resize(np.asarray(mask), (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        # frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)

        return mask
        
class SelfieSegMP:
    def __init__(self, width=320, height=240):
        self.width = width
        self.height = height
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    def seg(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = self.selfie_segmentation.process(image)

        mask = cv2.resize(results.segmentation_mask, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        mask = (255 * mask).astype("uint8")
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

        return mask

class SelfieSegSN:
    def __init__(self, width=320, height=240):
        self.width = width
        self.height = height

        # Load the sinet pytorch model
        self.config = [[[3, 1], [5, 1]], [[3, 1], [3, 1]],
                  [[3, 1], [5, 1]], [[3, 1], [3, 1]], [[5, 1], [3, 2]], [[5, 2], [3, 4]],
                  [[3, 1], [3, 1]], [[5, 1], [5, 1]], [[3, 2], [3, 4]], [[3, 1], [5, 2]]]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = SINet(classes=2, p=2, q=8, config=self.config, chnn=1)
        self.model.load_state_dict(torch.load('./models/model_296.pth', map_location=self.device))
        self.model.eval()

        # Enable gpu mode, if cuda available
        self.model.to(self.device)

        # Mean and std. deviation for normalization
        self.mean = [102.890434, 111.25247, 126.91212]
        self.std = [62.93292, 62.82138, 66.355705]
        self.dim = 320

    def seg(self, frame):
        img = np.array(frame)
        img = cv2.resize(img, (self.dim, self.dim))
        img = img.astype(np.float32)

        # Normalize and add batch dimension
        img = (img - self.mean) / self.std
        img /= 255
        img = img.transpose((2, 0, 1))
        img = img[np.newaxis, ...]

        # Load the inputs into GPU
        inps = torch.from_numpy(img).float().to(self.device)

        # Perform prediction and plot results
        with torch.no_grad():
            torch_res = self.model(inps)
            _, mask = torch.max(torch_res, 1)

        # Alpha blending with background image
        mask = mask.view(self.dim, self.dim, 1).cpu().numpy()
        mask = mask * 255
        mask = mask.reshape(self.dim, self.dim).astype("uint8")
        mask = cv2.resize(mask, (self.width, self.height))
        _, mask = cv2.threshold(mask, 128, 255, 0)

        return mask

    
if __name__ == "__main__":
    #"""
    width = 320
    height = 240
    segMP = SelfieSegMP(width, height)
    segSN = SelfieSegSN(width, height)
    segPN = SelfieSegPN(width, height)
    segMNV2 = SelfieSegMNV2(width, height)
    segMNV3 = SelfieSegMNV3(width, height)
    
    seg_dict = {  "MP" : segMP,
                  "SN" : segSN,
                  "PN" : segPN,
                  "MNV2" : segMNV2,
                  "MNV3" : segMNV3
                }    
    mask_dict = { "MP" : np.empty((width, height)),
                  "SN" : np.empty((width, height)),
                  "PN" : np.empty((width, height)),
                  "MNV2" : np.empty((width, height)),
                  "MNV3" : np.empty((width, height))
                }
                
    def worker(seg_name, frame):        
        mask_dict[seg_name] = seg_dict[seg_name].seg(frame)
    
    
    # Capture video from camera
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    # Load and resize the background image
    bgd = cv2.imread('./img/background.jpg')
    bgd = cv2.resize(bgd, (width, height))

    elapsedTime = 0
    count = 0

    while cv2.waitKey(1) < 0:
        t1 = time.time()

        # Read input frames
        success, frame = cap.read()
        if not success:
           cap.release()
           break
           
           
        # Get segmentation masks
        threads = []
        for seg_name in seg_dict.keys():
            thread = threading.Thread(target=worker, args=(seg_name, frame))
            thread.start()
            threads.append(thread)
            
        for thread in threads:
            thread.join()

        # Merge with background
        fgMP = cv2.bitwise_or(frame, frame, mask=mask_dict["MP"])
        bgMP = cv2.bitwise_or(bgd, bgd, mask=~mask_dict["MP"])
        outMP = cv2.bitwise_or(fgMP, bgMP)
        # Merge with background
        fgSN = cv2.bitwise_or(frame, frame, mask=mask_dict["SN"])
        bgSN = cv2.bitwise_or(bgd, bgd, mask=~mask_dict["SN"])
        outSN = cv2.bitwise_or(fgSN, bgSN)
        # Merge with background
        fgPN = cv2.bitwise_or(frame, frame, mask=mask_dict["PN"])
        bgPN = cv2.bitwise_or(bgd, bgd, mask=~mask_dict["PN"])
        outPN = cv2.bitwise_or(fgPN, bgPN)
        # Merge with background
        fgMNV2 = cv2.bitwise_or(frame, frame, mask=mask_dict["MNV2"])
        bgMNV2 = cv2.bitwise_or(bgd, bgd, mask=~mask_dict["MNV2"])
        outMNV2 = cv2.bitwise_or(fgMNV2, bgMNV2)
        # Merge with background
        fgMNV3 = cv2.bitwise_or(frame, frame, mask=mask_dict["MNV3"])
        bgMNV3 = cv2.bitwise_or(bgd, bgd, mask=~mask_dict["MNV3"])
        outMNV3 = cv2.bitwise_or(fgMNV3, bgMNV3)

        cv2.imshow('Source', frame)
        cv2.imshow('Mediapipe', outMP)
        cv2.imshow('SINet', outSN)
        cv2.imshow('PortraitNet', outPN)
        cv2.imshow('MobileNetV2', outMNV2)
        cv2.imshow('MobileNetV3', outMNV3)

    cv2.destroyAllWindows()
    cap.release()
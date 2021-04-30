from argparse import ArgumentParser

import cv2
import imageio
import numpy as np
import torch
from minine.minine_decoder import MinineDecoder
from minine.minine_encoder import MinineEncoder
from minine.preprocessor import Preprocessor

# parameters
parser = ArgumentParser()
parser.add_argument("input", help="path to input video")
parser.add_argument("--padding", default=40, help="padding for the preprocessing")
opt = parser.parse_args()

# image
input_image_path = opt.input
input_image = imageio.imread(input_image_path)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Initialise Minine codec
encoder = MinineEncoder(identity_refresh_rate=99999999999)
decoder = MinineDecoder()

# Initialise VideoCapture
prep = Preprocessor(padding=opt.padding)

# Preprocessing the frame
rect = prep.detect_face(input_image)
if len(rect) != 1:
    # Either too many or no faces detected
    exit("No face found in image")

rect = rect[0]
frame = prep.cut_face(input_image, rect)
frame = prep.resize_image(frame)

# Constructing a GUI with sliders for the rotation & translation
def nothing(x):
    pass


cv2.namedWindow("nvidai-minine")
cv2.createTrackbar("angle", "nvidai-minine", 0, 360, nothing)
cv2.createTrackbar("tx+", "nvidai-minine", 0, 100, nothing)
cv2.createTrackbar("tx-", "nvidai-minine", 0, 100, nothing)
cv2.createTrackbar("ty+", "nvidai-minine", 0, 100, nothing)
cv2.createTrackbar("ty-", "nvidai-minine", 0, 100, nothing)
# cv2.createTrackbar("scale+", "nvidai-minine", 0, 100, nothing)
# cv2.createTrackbar("scale-", "nvidai-minine", 0, 100, nothing)

while True:
    # Encoding the frame
    encoded_frame = encoder.encode_frame(frame)

    # Manipulating the keypoints
    if encoded_frame.encPE != None:
        keypoints = encoded_frame.encPE.keypoints["value"]
        keypoints = keypoints.squeeze(dim=0).cpu()

        theta = np.radians(cv2.getTrackbarPos("angle", "nvidai-minine"))
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

        tx_pos = cv2.getTrackbarPos("tx+", "nvidai-minine") / 100
        tx_neg = cv2.getTrackbarPos("tx-", "nvidai-minine") / 100
        ty_pos = cv2.getTrackbarPos("ty+", "nvidai-minine") / 100
        ty_neg = cv2.getTrackbarPos("ty-", "nvidai-minine") / 100

        # scale_pos = 1 + (cv2.getTrackbarPos("scale+", "nvidai-minine") / 100)
        # scale_neg = 1 + (cv2.getTrackbarPos("scale-", "nvidai-minine") / 100)

        keypoints[:, 0] += tx_pos - tx_neg
        keypoints[:, 1] += ty_pos - ty_neg

        # keypoints *= scale_pos / scale_neg

        keypoints = keypoints.numpy().dot(rotation_matrix)
        keypoints = torch.tensor(keypoints).type(torch.float32)
        keypoints = keypoints.unsqueeze(dim=0).to("cuda")
        encoded_frame.encPE.keypoints["value"] = keypoints

    # Decoding the frame
    decoded_frame = decoder.decode_frame(encoded_frame)

    # Display the resulting frame
    resized_frame = prep.resize_image(decoded_frame, (512, 512))
    cv2.imshow("nvidai-minine", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        # When everything done, release the capture
        cv2.destroyAllWindows()
        break

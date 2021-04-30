import cv2
import numpy as np

from minine.minine_decoder import MinineDecoder
from minine.minine_encoder import MinineEncoder
from minine.preprocessor import Preprocessor

# Initialise Minine codec
encoder = MinineEncoder(identity_refresh_rate=100)
decoder = MinineDecoder()

# Initialise VideoCapture
prep = Preprocessor(detection_refresh_rate=10000)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocessing the frame
    rect = prep.detect_face(frame)
    if len(rect) != 1:
        # Either too many or no faces detected
        # TODO should not skip frame, but reuse previous coords
        print("frame skipped")
        continue

    rect = rect[0]
    frame = prep.cut_face(frame, rect)
    frame = prep.resize_image(frame)

    # Encoding the frame
    encoded_frame = encoder.encode_frame(frame)

    # Decoding the frame
    decoded_frame = decoder.decode_frame(encoded_frame)

    # Get the keypoints out of the encPE
    if encoded_frame.encPE != None:
        keypoints = (256 / 2) * (encoded_frame.encPE.keypoints["value"] + 1)
        keypoints = keypoints.squeeze(dim=0).cpu()
        jacobians = encoded_frame.encPE.keypoints["jacobian"].squeeze(dim=0).cpu()

        # plot the keypoints and jacobians on the decoded frames
        for kp, jac in zip(keypoints, jacobians):

            # extract keypoints from encoded frames
            x = int(round(kp[0].item()))
            y = int(round(kp[1].item()))

            # extract jacobian from encoded frames and make calculate ellipse shape
            xv = np.array(jac[:, 0].cpu())
            yv = np.array(jac[:, 1].cpu())

            axesLengthScalar = 12
            axesLength = (
                round(axesLengthScalar * np.linalg.norm(xv)),
                round(axesLengthScalar * np.linalg.norm(yv)),
            )

            unit_xv = xv / np.linalg.norm(xv)
            unit_yv = yv / np.linalg.norm(yv)
            dot_product = np.dot(unit_xv, unit_yv)
            angle = 90 + np.degrees(np.arccos(dot_product))

            decoded_frame = cv2.cvtColor(decoded_frame, cv2.COLOR_BGR2RGB)

            # Drawing the keypoints
            decoded_frame = cv2.circle(
                decoded_frame, (x, y), 2, (100, 0, 0), thickness=-1
            )

            # Drawing the jacobians
            overlay = decoded_frame.copy()
            overlay = cv2.ellipse(
                overlay,
                (x, y),
                axesLength,
                angle,
                startAngle=0,
                endAngle=360,
                color=(255, 0, 0),
                thickness=-1,
            )
            decoded_frame = cv2.addWeighted(overlay, 0.1, decoded_frame, 0.9, 0)

            decoded_frame = cv2.cvtColor(decoded_frame, cv2.COLOR_RGB2BGR)

    # Display the resulting frame
    resized_frame = prep.resize_image(decoded_frame, (512, 512))
    cv2.imshow("nvidai-minine", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
from minine.minine_decoder import MinineDecoder
from minine.minine_encoder import MinineEncoder
from minine.preprocessor import Preprocessor

# Initialise Minine codec
encoder = MinineEncoder(identity_refresh_rate=200)
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

    # Display the resulting frame
    resized_frame = prep.resize_image(decoded_frame, (512, 512))
    resized_frame_original = prep.resize_image(frame, (512, 512))

    cv2.imshow("nvidai-minine", np.hstack((resized_frame, resized_frame_original)))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

import cv2
import pyvirtualcam
from minine.minine_decoder import MinineDecoder
from minine.minine_encoder import MinineEncoder
from minine.preprocessor import Preprocessor

# Initialise Minine codec
encoder = MinineEncoder(identity_refresh_rate=100)
decoder = MinineDecoder()

# Initialise VideoCapture
prep = Preprocessor(detection_refresh_rate=50)
cap = cv2.VideoCapture(0)

# Initialise virtual cam
cam = pyvirtualcam.Camera(width=512, height=512, fps=20)
print(f"Using virtual camera: {cam.device}")

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

    resized_frame = prep.resize_image(decoded_frame, (512, 512))
    cam_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    cam.send(cam_frame)
    cam.sleep_until_next_frame()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

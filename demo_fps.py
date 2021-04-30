import cv2
import time
from minine.minine_decoder import MinineDecoder
from minine.minine_encoder import MinineEncoder
from minine.preprocessor import Preprocessor

# Initialise Minine codec
encoder = MinineEncoder(identity_refresh_rate=200)
decoder = MinineDecoder()

# Initialise VideoCapture
prep = Preprocessor(detection_refresh_rate=200)
cap = cv2.VideoCapture(0)

timestamp = time.time()
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

    # Calculate fps and reset counter
    fps = 1. / (time.time() - timestamp)
    timestamp = time.time()

    # Add fps counter to the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame = cv2.cvtColor(decoded_frame, cv2.COLOR_BGR2RGB)
    frame = cv2.putText(
        frame,
        "fps: " + str(int(fps)),
        (5, 20),
        font,
        0.4,
        (100, 255, 0),
        1,
        cv2.LINE_AA,
    )
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Display the resulting frame
    resized_frame = prep.resize_image(frame, (512, 512))
    cv2.imshow("nvidai-minine", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

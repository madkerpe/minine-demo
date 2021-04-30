import time

import cv2
import matplotlib.pyplot as plt

from minine.minine_decoder import MinineDecoder
from minine.minine_encoder import MinineEncoder
from minine.preprocessor import Preprocessor

fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

identity_refresh_rate = 10

t_start_setup = time.time()
# Initialise Minine codec
encoder = MinineEncoder(identity_refresh_rate=identity_refresh_rate)
decoder = MinineDecoder()

# Initialise VideoCapture
prep = Preprocessor(detection_refresh_rate=10 * identity_refresh_rate)
cap = cv2.VideoCapture(0)
t_end_setup = time.time()
setup_time = 1000 * (t_end_setup - t_start_setup)

# Buffers for frame data
capture_durations, preprocessing_durations, display_durations = [], [], []
encoding_durations, decoding_durations = [], []
frame_numbers, frame_number = [], 0
frame_durations, frame_time = [], 0
calculated_fps = 0

print(f"setup took {setup_time} ms")
fig.show()

while True:
    # Capture frame-by-frame
    t_start_capture = time.time()
    ret, frame = cap.read()
    t_end_capture = time.time()
    capture_durations.append(1000 * (t_end_capture - t_start_capture))
    frame_time += 1000 * (t_end_capture - t_start_capture)

    # Preprocessing the frame
    t_start_preprocessing = time.time()
    rect = prep.detect_face(frame)
    if len(rect) != 1:
        # Either too many or no faces detected
        # TODO should not skip frame, but reuse previous coords
        print("frame skipped")
        capture_durations.pop()
        frame_time = 0
        continue

    rect = rect[0]
    frame = prep.cut_face(frame, rect)
    frame = prep.resize_image(frame)
    t_end_preprocessing = time.time()
    preprocessing_durations.append(1000 * (t_end_preprocessing - t_start_preprocessing))
    frame_time += 1000 * (t_end_preprocessing - t_start_preprocessing)

    # Encoding the frame
    t_start_encoding = time.time()
    encoded_frame = encoder.encode_frame(frame)
    t_end_encoding = time.time()
    encoding_durations.append(1000 * (t_end_encoding - t_start_encoding))
    frame_time += 1000 * (t_end_encoding - t_start_encoding)

    # Decoding the frame
    t_start_decoding = time.time()
    decoded_frame = decoder.decode_frame(encoded_frame)
    t_end_decoding = time.time()
    decoding_durations.append(1000 * (t_end_decoding - t_start_decoding))
    frame_time += 1000 * (t_end_decoding - t_start_decoding)

    # Add fps counter to the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame = cv2.cvtColor(decoded_frame, cv2.COLOR_BGR2RGB)
    frame = cv2.putText(
        frame,
        "calculated fps: " + str(int(calculated_fps)),
        (5, 20),
        font,
        0.4,
        (100, 255, 0),
        1,
        cv2.LINE_AA,
    )
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Display the resulting frame
    t_start_display = time.time()
    resized_frame = prep.resize_image(frame, (512, 512))
    cv2.imshow("nvidai-minine", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        plt.close()
        break
    t_end_display = time.time()
    display_durations.append(1000 * (t_end_display - t_start_display))
    frame_time += 1000 * (t_end_display - t_start_display)
    calculated_fps = 1 / (frame_time / 1000)
    frame_durations.append(frame_time)
    frame_time = 0

    # Display the timings in real-time
    frame_numbers.append(frame_number)
    sliding_window_size = 5
    sliding_window_start = max(
        0, frame_number - sliding_window_size * identity_refresh_rate
    )

    ax1.clear()
    ax2.clear()
    ax3.clear()

    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 20)
    ax3.set_ylim(0, 100)

    ax1.plot(
        frame_numbers[sliding_window_start:],
        encoding_durations[sliding_window_start:],
        label="encoding time (ms)",
        color="g",
    )
    ax1.plot(
        frame_numbers[sliding_window_start:],
        decoding_durations[sliding_window_start:],
        label="decoding time (ms)",
        color="r",
    )

    ax2.plot(
        frame_numbers[sliding_window_start:],
        capture_durations[sliding_window_start:],
        label="capture time (ms)",
        color="g",
    )
    ax2.plot(
        frame_numbers[sliding_window_start:],
        preprocessing_durations[sliding_window_start:],
        label="preprocess time (ms)",
        color="b",
    )
    ax2.plot(
        frame_numbers[sliding_window_start:],
        display_durations[sliding_window_start:],
        label="display time (ms)",
        color="r",
    )

    ax3.plot(
        frame_numbers[sliding_window_start:],
        frame_durations[sliding_window_start:],
        label="total time (ms)",
        color="g",
    )

    # TODO: fix location of legend
    ax1.legend()
    ax2.legend()
    ax3.legend()

    fig.canvas.draw()
    time.sleep(0.1)
    frame_number += 1

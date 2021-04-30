import imageio
from minine.minine_decoder import MinineDecoder
from minine.minine_encoder import MinineEncoder
from minine.preprocessor import Preprocessor
from pympler import asizeof
from skimage import img_as_ubyte

# video
input_video_path = "./input_video.mp4"
output_video_path = "./output_video.mp4"

# Initialise Minine codec
encoder = MinineEncoder(identity_refresh_rate=100)
decoder = MinineDecoder()

prep = Preprocessor(detection_refresh_rate=10000)

# Read video from disk
video_reader = imageio.get_reader(input_video_path)
video_frames = [frame for frame in video_reader]
fps = video_reader.get_meta_data()["fps"]
decoded_frames = []

for frame in video_frames:
    # Encoding the frame
    encoded_frame = encoder.encode_frame(frame)
    print(f"size of 'transmitted' frame is: {asizeof.asizeof(encoded_frame)} bytes")

    # Decoding the frame
    decoded_frame = decoder.decode_frame(encoded_frame)
    decoded_frames.append(decoded_frame)

# Write video to disk
ubyte_frames = [img_as_ubyte(frame) for frame in decoded_frames]
imageio.mimsave(output_video_path, ubyte_frames, fps=30)

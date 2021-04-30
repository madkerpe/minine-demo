from argparse import ArgumentParser

import imageio
from minine.minine_decoder import MinineDecoder
from minine.minine_encoder import MinineEncoder
from minine.preprocessor import Preprocessor
from skimage import img_as_ubyte
from tqdm import tqdm

# parameters
parser = ArgumentParser()
parser.add_argument("input", help="path to input video")
parser.add_argument(
    "--output", default="./output/output_video.mp4", help="path the output video"
)
parser.add_argument(
    "--preprocessing", default=False, dest="preprocessing", action="store_true"
)
opt = parser.parse_args()

# video
input_video_path = opt.input
output_video_path = opt.output

# Initialise Minine codec
encoder = MinineEncoder(identity_refresh_rate=999999999)
decoder = MinineDecoder()

# Read video from disk
video_reader = imageio.get_reader(input_video_path)
video_frames = [frame for frame in video_reader]
fps = video_reader.get_meta_data()["fps"]
decoded_frames = []

if opt.preprocessing:
    prep = Preprocessor(detection_refresh_rate=999999999)
    rect = prep.detect_face(video_frames[0])
    if len(rect) != 1:
        exit("Either too many or no faces detected")
    rect = rect[0]

for frame in tqdm(video_frames):
    # Optionally preprocessing
    if opt.preprocessing:
        frame = prep.cut_face(frame, rect)
        frame = prep.resize_image(frame)

    # Encoding the frame
    encoded_frame = encoder.encode_frame(frame)

    # Decoding the frame
    decoded_frame = decoder.decode_frame(encoded_frame)
    decoded_frames.append(decoded_frame)

# Write video to disk
ubyte_frames = [img_as_ubyte(frame) for frame in decoded_frames]
imageio.mimsave(output_video_path, ubyte_frames, fps=30)

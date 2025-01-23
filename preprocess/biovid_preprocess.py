# parsing labels, segment and crop raw videos.
import argparse
import os
import sys

sys.path.append(os.getcwd())


def crop_face(root: str, ext: str):
    from util.face_sdk.face_crop import process_videos
    source_dir = os.path.join(root, "downloaded")
    target_dir = os.path.join(root, "cropped")
    process_videos(source_dir, target_dir, ext=ext)


def gen_split(root: str, ext: str):
    videos = list(filter(lambda x: x.endswith(ext), os.listdir(os.path.join(root, 'cropped'))))
    total_num = len(videos)

    with open(os.path.join(root, "train.txt"), "w") as f:
        for i in range(int(total_num * 0.8)):
            f.write(videos[i][:-len(ext)] + "\n")

    with open(os.path.join(root, "val.txt"), "w") as f:
        for i in range(int(total_num * 0.8), int(total_num * 0.9)):
            f.write(videos[i][:-len(ext)] + "\n")

    with open(os.path.join(root, "test.txt"), "w") as f:
        for i in range(int(total_num * 0.9), total_num):
            f.write(videos[i][:-len(ext)] + "\n")


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="Root directory of CelebV-HQ", required=True)
parser.add_argument("--ext", help="File extension of videos", default=".mp4")  # Default to .mp4
args = parser.parse_args()

if __name__ == '__main__':
    data_root = args.data_dir
    extension = args.ext

    crop_face(data_root, extension)

    # Uncomment the following lines if you want to check for existing files before generating splits
    # if not os.path.exists(os.path.join(data_root, "train.txt")) or \
    #    not os.path.exists(os.path.join(data_root, "val.txt")) or \
    #    not os.path.exists(os.path.join(data_root, "test.txt")):
    #     gen_split(data_root, extension)

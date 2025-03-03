import os.path
import sys
import argparse
import shutil

parser = argparse.ArgumentParser("Preprocess YTF dataset")
parser.add_argument("--data_dir", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--max_workers", type=int, default=8)

if __name__ == '__main__':

    args = parser.parse_args()

    # # copy the metadata (split) to the data_dir
    # shutil.copy(os.path.join(os.path.dirname(__file__), "..", "dataset", "misc", "youtube_face", "train_set.csv"),
    #     args.data_dir)
    # shutil.copy(os.path.join(os.path.dirname(__file__), "..", "dataset", "misc", "youtube_face", "val_set.csv"),
    #     args.data_dir)

    # Crop faces from videos
    sys.path.append(".")
    if not os.path.exists("logs"):
        os.mkdir("logs")

    from util.face_sdk.face_crop import process_images_multi_modal
    process_images_multi_modal(
        os.path.join(args.data_dir, "Texture_crop"),
        os.path.join(args.data_dir, "Depth_crop"),
        os.path.join(args.data_dir, "Thermal_crop"),
        os.path.join(args.data_dir)  # Save directory
    )

    # Face parsing based on these cropped faces (only for texture images)
    from util.face_sdk.face_parse import process_images as face_parse_process_images
    face_parse_process_images(
        os.path.join(args.data_dir, "Texture_crop_crop_images_DB"),
        os.path.join(args.data_dir, "Texture_crop_face_parsing_images_DB")
    )


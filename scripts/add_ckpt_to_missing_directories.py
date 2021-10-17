import os

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_dir",
    "weights/original_weights",
    help="Path to directories with original weights.",
)


def main(argv_):
    """Add `checkpoint` file to missing ckpt directories."""
    ckpt_directories = []
    for content in os.listdir(FLAGS.input_dir):
        if content.startswith("resnet-rs-"):
            full_path = os.path.join(FLAGS.input_dir, content)
            if os.path.isdir(full_path):
                ckpt_directories.append(full_path)

    for directory in ckpt_directories:
        if "checkpoint" not in os.listdir(directory):
            lines_to_add = [
                'model_checkpoint_path: "model.ckpt"',
                'all_model_checkpoint_paths: "model.ckpt"',
            ]
            with open(
                f"{os.path.join(FLAGS.input_dir, directory)}/checkpoint", "w"
            ) as f:
                f.writelines(lines_to_add)
            print(f"Added checkpoint to {directory}")


if __name__ == "__main__":
    app.run(main)

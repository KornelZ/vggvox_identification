from cnn import Cnn
from audio_generator import AudioGenerator
import sys
import os


def train(metadata_file, dataset_dir, checkpoint_file):
    generator = AudioGenerator(metadata_path=metadata_file,
                               dataset_path=dataset_dir,
                               batch_size=20)
    vggvox_net = Cnn()
    vggvox_net.fit_generator(
        model_checkpoint_path=checkpoint_file,
        train_generator=generator.train(),
        train_steps=generator.get_training_steps(),
        validation_generator=generator.validate(),
        validation_steps=generator.get_validation_steps())


if __name__ == "__main__":
    argc = len(sys.argv)
    if argc != 4:
        print("Usage: python vggvox_test.py metadata_path.csv dataset_dir checkpoint_path.hdf5")
        exit(1)
    else:
        if not os.path.exists(sys.argv[1]):
            raise IOError("Metadata csv file does not exist")
        if not os.path.exists(sys.argv[2]):
            raise IOError("Dataset directory does not exist")
        train(*sys.argv[1:])

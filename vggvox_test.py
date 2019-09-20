from cnn import Cnn
from audio_generator import AudioGenerator
import sys
import os


def test(metadata_file, dataset_dir, model_file):
    generator = AudioGenerator(metadata_path=metadata_file,
                               dataset_path=dataset_dir,
                               batch_size=1)
    vggvox_net = Cnn(model_file)
    metrics = vggvox_net.evaluate_generator(test_generator=generator.test(),
                                            test_steps=generator.get_test_steps())
    print(f"Loss {metrics[0]}, Top 1 Accuracy {metrics[1]}, Top 5 Accuracy {metrics[2]}")


if __name__ == "__main__":
    argc = len(sys.argv)
    if argc != 4:
        print("Usage: python vggvox_test.py metadata_path.csv dataset_dir model_path.hdf5")
        exit(1)
    else:
        if not os.path.exists(sys.argv[1]):
            raise IOError("Metadata csv file does not exist")
        if not os.path.exists(sys.argv[2]):
            raise IOError("Dataset directory does not exist")
        if not os.path.exists(sys.argv[3]):
            raise IOError("Model weight file does not exist")
        test(*sys.argv[1:])

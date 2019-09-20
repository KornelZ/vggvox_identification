from preprocessing import process_audio
import csv
import os
import numpy
import scipy.io.wavfile
import keras as K


class AudioGenerator:

    TRAIN_SET = 1
    VALID_SET = 2
    TEST_SET = 3

    def __init__(self,
                 metadata_path: str,
                 dataset_path: str,
                 batch_size: int):
        self.metadata_path = metadata_path
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.data_dict = None
        self.num_classes = 0
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self._read_metadata()

    def _read_metadata(self):
        """Reads csv file with columns:
            DATA_TYPE PATH
            DATA_TYPE: 1 if train, 2 if validation, 3 if test
        """
        self.data_dict = {"dataset": [], "paths": [], "labels": []}
        try:
            with open(self.metadata_path) as f:
                reader = csv.reader(f, delimiter=" ")
                for row in reader:
                    if len(row) != 0:
                        self.data_dict["dataset"].append(int(row[0]))
                        record_path = str(row[1])
                        index = record_path.index("id1")
                        record_path = record_path[index:]
                        if self.dataset_path in record_path:
                            self.data_dict["paths"].append(record_path)
                        else:
                            self.data_dict["paths"].append(os.path.join(self.dataset_path, record_path).replace('\\', '/'))
                        self.data_dict["labels"].append(int(record_path[index + 3: index + 7]) - 1)
        except IOError as e:
            print("Could not read dataset: " + str(e))
            exit(1)

        self.num_classes = len(numpy.unique(self.data_dict["labels"]))
        self.train_data = self._get_set(AudioGenerator.TRAIN_SET)
        self.valid_data = self._get_set(AudioGenerator.VALID_SET)
        self.test_data = self._get_set(AudioGenerator.TEST_SET)

    def _get_steps(self, set_type):
        """Number of steps per epoch for training/validation/testing"""
        return numpy.ceil(len(self._get_set(set_type)) / self.batch_size)

    def _get_set(self, set_type):
        if set_type == AudioGenerator.TRAIN_SET and self.train_data is not None:
            return self.train_data
        if set_type == AudioGenerator.VALID_SET and self.valid_data is not None:
            return self.valid_data
        if set_type == AudioGenerator.TEST_SET and self.test_data is not None:
            return self.test_data

        data = {"dataset": [], "paths": [], "labels": []}
        for i in range(len(self.data_dict["labels"])):
            if self.data_dict["dataset"][i] == set_type:
                data["dataset"].append(self.data_dict["dataset"][i])
                data["paths"].append(self.data_dict["paths"][i])
                data["labels"].append(self.data_dict["labels"][i])
        return numpy.array(list(zip(data["dataset"], data["labels"], data["paths"])))

    def _filter_dataset(self, column, value):
        return list(filter(lambda x: x == value, self.data_dict[column]))

    def _generate_batches(self, dataset, clip=True):
        while True:
            i = 0
            batch = []
            labels = []
            for x, y in self._generate(dataset, clip):
                batch.append(x)
                labels.append(y)
                i += 1
                if self.batch_size == len(batch) or i == len(dataset):
                    yield numpy.array(batch), numpy.array(labels)
                    batch = []
                    labels = []

    def _generate(self, dataset, clip):
        for i in range(len(dataset)):
            yield self._get_example(dataset, i, clip)

    def _get_example(self, dataset, index, clip):
        try:
            _, wave = scipy.io.wavfile.read(dataset[index][2], mmap=True)
            return process_audio(wave, clip), self._to_categorical(dataset, index)
        except IOError as e:
            print("Could not read wav file: " + str(e))
            exit(1)

    def _to_categorical(self, dataset, index):
        return numpy.array(K.utils.to_categorical(dataset[index][1], self.num_classes))

    def train(self):
        """Generator for training"""
        numpy.random.shuffle(self.train_data)
        for b, l in self._generate_batches(self.train_data):
            yield b, l

    def validate(self):
        """Generator for validation"""
        for b, l in self._generate_batches(self.valid_data):
            yield b, l

    def test(self):
        """Generator for testing"""
        for b, l in self._generate_batches(self.test_data, clip=False):
            yield b, l

    def get_training_steps(self):
        return self._get_steps(AudioGenerator.TRAIN_SET)

    def get_validation_steps(self):
        return self._get_steps(AudioGenerator.VALID_SET)

    def get_test_steps(self):
        return self._get_steps(AudioGenerator.TEST_SET)

    def print_dataset_info(self):
        print("Metadata path:", self.metadata_path)
        print("Dataset path:", self.dataset_path)
        print("Total examples:", len(self.data_dict["paths"]))
        print("Train examples:", len(self._filter_dataset("dataset", 1)))
        print("Valid examples:", len(self._filter_dataset("dataset", 2)))
        print("Test examples:", len(self._filter_dataset("dataset", 3)))
        print("Num classes:", self.num_classes)


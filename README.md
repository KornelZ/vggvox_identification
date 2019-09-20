# Vggvox Identification
Training and evaluation of VGGVox neural network for speaker identification.
Based on on Nagrani et al 2017, [VoxCeleb: a large-scale speaker identification dataset](https://arxiv.org/pdf/1706.08612.pdf) and https://github.com/linhdvu14/vggvox-speaker-identification

### Instructions
1. Install Python 3.6 and requirements.
2. Download dataset file (see Downloads).
3. To run training code:
```
python.py vggvox_train.py metadata.csv voxceleb/wav/ checkpoint.hdf5
```
3. To run testing code:
```
python.py vggvox_test.py metadata.csv voxceleb/wav/ with-augmentation.hdf5
```

### Downloads
Model weights trained on augmented dataset (~81.77% accuracy)
https://drive.google.com/open?id=1BMnrz6B6WYRfx7ssY6DacTeoIlFYxZd2

Model weights trained on original dataset (~74.60% accuracy)
https://drive.google.com/open?id=1YxCi6_BlOG8oV85g4Gu3B2T-I1hzC6aa

7z dataset file (see: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)
https://drive.google.com/open?id=1n1YAUHH84R7F19DofHiDFLKHheJC6nwO

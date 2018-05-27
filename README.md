# Cloze-Test
Implementation of model in 'A Simple and Effective Appraoch to the Story Cloze Test'

Paper:  https://arxiv.org/abs/1803.05547

Requirements:

- PyTorch
- numpy
- h5py
- matplotlib

# Data

Available at: https://drive.google.com/drive/folders/1d2WsD-Zvf8k_slDlJJyH9GVWAzo2Ium4?usp=sharing

Files: 

- ROC_encoded.hdf5 (not used): The skip-thought embeddings of the training stories in ROCStory corpus
- ROCval_encoded.hdf5:  The skip-thought embeddings of the validation  stories in ROCStory corpus
- ROCtest_encoded.hdf5: The skip-thought embeddings of the  test stories in the ROCStory corpus

# Instructions

Download data to same folder and run `skipthought-cloze-val-softmax.py`.
Uncomment `trainIters()` in the above file to train model.

# Other Links

- Skipthought embeddings: https://github.com/ryankiros/skip-thoughts
- ROCStory corpus/Cloze Test: http://cs.rochester.edu/nlp/rocstories/
# ReCo-Diff
ReCo-Diff: Residual-Conditioned Deterministic Sampling for Cold Diffusion in Sparse-View CT

## Requirements

- Create a new virtual environment and install the required libraries using `requirements.txt`.
  ```shell
  git clone https://github.com/choiyoungeunn/ReCo-Diff.git
  cd ReCo-Diff
  conda create -n recodiff python=3.7
  conda activate recodiff
  pip install -r ./requirements.txt
  ```
- torch-radon is required for simulating DRRs and geometry utils. Install torch-radon by:
1. Download torch-radon from [torch-radon](https://github.com/matteo-ronchetti/torch-radon)
     ```shell
     git clone https://github.com/matteo-ronchetti/torch-radon.git
     ```
2. Due to some out-dated Pytorch function in torch-radon, modify code by running
     ```shell
     cd torch-radon
     patch -p1 < path/to/ReCo-Diff/torch-radon_fix/torch-radon_fix.patch
     ```
3. Install torch-radon by running
     ```shell
     python setup.py install
     ```

## Dataset

- Download the [AAPM dataset](https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h/folder/144226105715).
- After downloading, run preprocessing with `./datasets/preprocess_aapm.py`.
- Provide dataset paths directly in each script before running.

## Train (ReCo-Diff)

- Script: `recodiff_train.sh`
- Set `res_dir` and `dataset_path` at the top of the file, then run:
  ```shell
  ./recodiff_train.sh
  ```

## Test (ReCo-Diff)

### Single checkpoint

- Script: `recodiff_test.sh`
- Set `res_dir`, `dataset_path`, and `net_checkpath_default` at the top of the file, then run:
  ```shell
  ./recodiff_test.sh
  ```

### All checkpoints in a directory

- Script: `recodiff_test_ALL_search.sh`
- Set `ckpt_dir` at the top of the file, then run:
  ```shell
  ./recodiff_test_ALL_search.sh
  ```

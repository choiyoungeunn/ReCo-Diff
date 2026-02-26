# ReCo-Diff
ReCo-Diff: Residual-Conditioned Deterministic Sampling for Cold Diffusion in Sparse-View CT

---

## 📦 Requirements

- Create a new virtual environment and install the required libraries using `requirements.txt`.

```shell
git clone https://github.com/choiyoungeunn/ReCo-Diff.git
cd ReCo-Diff
conda create -n recodiff python=3.7
conda activate recodiff
pip install -r ./requirements.txt
```

- `torch-radon` is required for simulating DRRs and geometry utilities.

### 🔧 Install torch-radon

1. Download torch-radon  
👉 https://github.com/matteo-ronchetti/torch-radon

```shell
git clone https://github.com/matteo-ronchetti/torch-radon.git
```

2. Apply patch (due to outdated PyTorch functions)

```shell
cd torch-radon
patch -p1 < path/to/ReCo-Diff/torch-radon_fix/torch-radon_fix.patch
```

3. Install

```shell
python setup.py install
```

---

## 📊 Dataset

- Download the AAPM dataset:  
  https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h/folder/144226105715

- After downloading, run preprocessing:

```shell
python ./datasets/preprocess_aapm.py
```

- Provide dataset paths directly in each script before running.

---

## 💾 Pretrained Checkpoint

Download the checkpoint from the link below and load it into the model to run the test.

👉 https://drive.google.com/drive/folders/17G5z6vLXAuA5GYvGJTbSP1kEAVBPt6mh?usp=sharing

---

## 🚀 Train (ReCo-Diff)

- Script: `recodiff_train.sh`
- Set `res_dir` and `dataset_path` at the top of the file, then run:

```shell
./recodiff_train.sh
```

---

## 🧪 Test (ReCo-Diff)

### 🔹 Single checkpoint

- Script: `recodiff_test.sh`
- Set `res_dir`, `dataset_path`, and `net_checkpath_default` at the top of the file, then run:

```shell
./recodiff_test.sh
```

### 🔹 All checkpoints in a directory

- Script: `recodiff_test_ALL_search.sh`
- Set `ckpt_dir` at the top of the file, then run:

```shell
./recodiff_test_ALL_search.sh
```

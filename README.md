# Universal Offline BBO

Implementation code on research of universal offline black-box optimization.

## Template 

We sincerely thank the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template).

## Requirements

For [Design-Bench](https://github.com/brandontrabucco/design-bench/) usage, we suggest using Python 3.8 and construct the environments as:

```shell
YOUR_PATH_TO_CONDA=~/.conda # Properly set it

# Create conda environment
conda create -n universal-offline python=3.8 -y
conda activate universal-offline

# Download MuJoCo package
wget https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210_linux.tar.gz
mkdir ~/.mujoco
tar -zxvf mujoco210_linux.tar.gz -C ~/.mujoco

# Mujoco_py installation
conda install gxx_linux-64 gcc_linux-64 -y
conda install --channel=conda-forge libxcrypt -y
pip install Cython==0.29.36 numpy==1.22.0 mujoco_py==2.1.2.14
# Set up the environment variable
conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
# Reactivate the conda environment to make the variable take effect
conda activate universal-offline
# Copy C++ dependency libraries
mkdir ${YOUR_PATH_TO_CONDA}/envs/universal-offline/include/X11
mkdir ${YOUR_PATH_TO_CONDA}/envs/universal-offline/include/GL
sudo cp /usr/include/X11/*.h ${YOUR_PATH_TO_CONDA}/envs/universal-offline/include/X11
sudo cp /usr/include/GL/*.h ${YOUR_PATH_TO_CONDA}/envs/universal-offline/include/GL
# Mujoco Compile
python -c "import mujoco_py"

# Torch Installation
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 torchmetrics==0.11.4

# Design-Bench Installation
pip install design-bench==2.0.12
pip install pip==24.0
pip install robel==0.1.2 morphing_agents==1.5.1 transforms3d --no-dependencies
pip install botorch==0.8.5 gpytorch==1.10

# Install other dependencies
pip install gym==0.13.1 params_proto==2.9.6 scikit-image==0.17.2 scikit-video==1.1.11 scikit-learn==0.23.1 wandb pypop7 pymoo==0.6.1.2

# Fix numpy version, otherwise it would raise environment error
pip install numpy==1.22.0

# Install lightning & hydra
pip install lightning==2.3.3 hydra-core==1.3.2 hydra-colorlog==1.2.0 hydra-optuna-sweeper==1.2.0

# Install sentence-piece
pip install -U sentencepiece

# Install other utils packages
pip install rootutils pre-commit rich pytest isort black

# Install SOO-bench
cd thirdparty_benchmark/
git clone https://github.com/zhuyiyi-123/SOO-Bench.git
cd SOO-Bench/
pip install -e ./revive_hybrid/
pip install -e .
cd ../../
```

then download data of Design-Bench following [this link](https://github.com/brandontrabucco/design-bench/issues/11#issuecomment-2067352331).

To check the stability of installation, you can run test demos in ``demo/`` by
```shell
python tests/test_design_bench.py
python tests/test_t5.py
python tests/test_tokenizer.py
```

## Usage
For UniSO-T, you can run by
```shell
python src/train_UniSO-T.py \
    experiment=UniSO-T-Improved \
    ++trainer.max_epochs=200
```
For UniSO-N, you should first fine-tune a pre-trained T5 via
```shell
python src/train_UniSO-N.py experiment=finetune_t5   
```
then, you can run by
```shell
# After specified the finetuned T5 ``.pt`` path,
PT_PATH=
python src/train_UniSO-N.py \
    experiment=UniSO-N \
    ++trainer.max_epochs=200 \
    ++pt_path=$PT_PATH
```

## Code Reference
+ For benchmark, we use [Design-Bench](https://github.com/brandontrabucco/design-bench) and [SOO-Bench](https://anonymous.4open.science/r/SOO-Bench-9025).
+ For language model, we use [Google T5](https://huggingface.co/docs/transformers/model_doc/t5).

## Citation
```bib
@inproceedings{tan2025towards,
  title={Towards Universal Offline Black-Box Optimization via Learning Language Model Embeddings},
  author={Tan, Rong-Xi and Chen, Ming and Xue, Ke and Wang, Yao and Wang, Yaoyuan and Fu, Sheng and Qian, Chao},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  address={Vancouver, Canada},
  year={2025},
}
```
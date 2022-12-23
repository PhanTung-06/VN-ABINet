## Get Started
- Enviroment
```sh
conda create --name abinet python=3.7
conda activate abinet
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install ipdb jupyter ipython opencv-python
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
git clone https://github.com/PhanTung-06/VN-ABINet.git
cd VN-ABINet
python -m pip install ninja yacs cython matplotlib tqdm version_utils opencv-python shapely scipy tensorboardX pyclipper Polygon3 weighted-levenshtein editdistance 
python setup.py build develop
```

#!/bin/sh
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -N honkan52tv
#$ -o out_job
#$ -e error_job
#$ -m abe
#$ -M emoto@sl.sc.e.titech.ac.jp
#$ -p -5

#M option no mail-address ha jibun no mono ni kaete kudasai

.  /etc/profile.d/modules.sh
module load python/3.6.5

#cupy ga ninshiki dekinai toki ha, uninstall site kudasai
#pip uninstall chainer -y
#pip uninstall chainercv -y
#pip uninstall chainermn -y
#pip uninstall cupy -y
#pip uninstall cupy-cuda80 -y

module load intel cuda openmpi
module load nccl/2.2.13
module load cudnn/7.1
module list
pip install --user --upgrade pip
pip install --user -q keras
pip install --user matplotlib
pip install --user imutils
pip install --user h5py
pip install --user pillow
pip install --user chainer
pip install --user chainercv
pip install --user cupy-cuda80
pip install --user python modules
pip freeze > piplist.txt
python -c 'import chainer; chainer.print_runtime_info()'
python ./main.py

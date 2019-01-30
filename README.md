# tsubame_honkan

# How to use
 `$ ssh tsubame -YC`  
 でTSUBAMEにログイン後,  
 適当な場所に  
 `$ git clone https://github.com/huehu4575/tsubame_honkan`
 でクローンする    
 その後  
 `$ cd tsubame_honkan`
 した後に  
 `$ qsub job.sh -g tga-systemcontrolproject`  
 とすれば学習が始まる.
 デフォルトでは全ての52ラベル(=13箇所x4階)についてそれぞれ3500枚を学習に使い
 testとvalidデータは写真で撮影したものを用いている.
# BF_files
 bf_job.shとbf_main.pyは地下階の13ラベルのみで学習をする.  
 単に
 `$ qsub bf_job.sh -g tga-systemcontrolproject`
 とすれば始まる.

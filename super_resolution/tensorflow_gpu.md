# 服务器环境搭建

## tensorflow-gpu

[tutorial](http://www.feiguyunai.com/index.php/2017/12/20/pythonai-install-tensorflow-gpu/) 但是我好像没用到, 服务器本身就下好了一些东西, anaconda 也帮我 handle 了 cudatoolkit and cudann

一些隐患: 我升级了 gcc 版本, 看到其他一些教程说什么要手动降低 GCC 版本, Let's agree to disagree

anaconda is great, it handles `cudatoolkit` and `cudann` for you.

jupyter book can be accessed remotely! No need to use teamviewer!

Do I need to set up as server clusters?

**gpu info**

```shell
(yee) vradmin@vr-pc08:/usr/lib/x86_64-linux-gnu$ nvidia-smi
Fri Dec 21 18:04:46 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 396.26                 Driver Version: 396.26                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 1070    Off  | 00000000:02:00.0  On |                  N/A |
| 29%   33C    P8    12W / 151W |    777MiB /  8116MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1141      G   /usr/lib/xorg/Xorg                           393MiB |
|    0      1866      G   compiz                                       158MiB |
|    0     12687      G   ...-token=553E3AD30440BCA0CAF2DCA67294686A   163MiB |
|    0     22235      G   ...-token=F41EA27769397462C054E430C12F1D36    59MiB |
+-----------------------------------------------------------------------------+

```

```shell
vradmin@vr-pc08:/usr/lib/x86_64-linux-gnu$ conda install tensorflow-gpu
Solving environment: done

## Package Plan ##

  environment location: /home/vradmin/anaconda3

  added / updated specs: 
    - tensorflow-gpu


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    cudatoolkit-9.2            |                0       351.0 MB
    tensorflow-base-1.12.0     |gpu_py36had579c0_0       216.9 MB
    _tflow_select-2.1.0        |              gpu           2 KB
    tensorflow-1.12.0          |gpu_py36he74679b_0           3 KB
    cupti-9.2.148              |                0         1.7 MB
    cudnn-7.2.1                |        cuda9.2_0       322.8 MB
    tensorflow-gpu-1.12.0      |       h0d30ee6_0           2 KB
    ------------------------------------------------------------
                                           Total:       892.5 MB

The following NEW packages will be INSTALLED:

    cudatoolkit:     9.2-0                    
    cudnn:           7.2.1-cuda9.2_0          
    cupti:           9.2.148-0                
    tensorflow-gpu:  1.12.0-h0d30ee6_0        

The following packages will be UPDATED:

    tensorflow:      1.12.0-mkl_py36h69b6ba0_0 --> 1.12.0-gpu_py36he74679b_0
    tensorflow-base: 1.12.0-mkl_py36h3c3e929_0 --> 1.12.0-gpu_py36had579c0_0

The following packages will be DOWNGRADED:

    _tflow_select:   2.3.0-mkl                 --> 2.1.0-gpu                


```

## jupyter remote mode

[running a notebook server](https://jupyter-notebook.readthedocs.io/en/latest/public_server.html#notebook-server-security)
流程与官方 tutorial 一样, 但是爆出了的错误

```shell
socket.gaierror: [Errno -5] No address associated with hostname
```

## teamviewer
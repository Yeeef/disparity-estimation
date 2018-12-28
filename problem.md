# 问题汇集

事情的起因是这样的, 我想要在电脑上打开 teamviewer, 但不知道为什么始终打不开, 如果在终端输入 `teamviewer` 依旧没有任何错误提示, gui 始终无法启动, 在网络搜索一些答案后, 决定 `sudo service restart lightdm`, 重启 displaymanager 之后, 就无法重新进入图形界面的系统了, 输完密码之后会重复卡在登陆界面. 真的是十分令人崩溃的问题, 对于这个问题, 网上也有不少的博文.

## 第一种可能的解决方案

第一种可能的解决方案, 需要我们 `ctrl + alt + F1` 后进入命令行界面, 在 ~ 目录下删除 `.Xauthority`, 具体背后的实际问题与机制我还不是很了解, 直觉上, 是和一些用户权限相关事项相关.
很可惜, 我并不是这个问题.

## 第二种可能的解决方案

既然 `lightdm` 挂了, 那么我们可以尝试一下 `gdm`, `gdm` 貌似在 ubuntu 12.04 之后就被 `lightdm` 所替换.
@TODO: 具体安装步骤待补充, 总之, 安装之后, 可以将 `lightdm` 切换为 `gdm`, 然而, 更加悲剧的事情发生了, 我连进入页面都看不到了, 开始无限在登录界面循环启动, 连命令行都进不去.
此时, 学长建议我把独显先拔掉, hdmi 插到集显上, 看看能不能解决这个问题, 很可惜, 还是不行.
到了这, 我的情绪已然渐渐失控, 在尚保留一丝理智的情况下, reboot 后进入 grub 页面, 进入了 recovery mode, 成功将 gdm 替换回了 lightdm.
所以, 这种解决方案依然对我无效.

## 第三种可能的解决方案

最后一种可能性, 那就是 **显卡驱动挂了**, 这台机子上装了一个 gtx-1070 的独立显卡, CPU 自己也有一个集成显卡, 为了重新装驱动, 我们需要删除上一次安装的 NVIDIA 驱动, 我没有在系统中找到 `NVIDIA......run` 文件, 推测上一任学长可能是用 `pip` 的方法直接下载的驱动, 这样我们可以利用 `sudo apt-get purge nvidia*` 命令来删除已经安装的显卡驱动, 删除后可以用 `nvidia-smi` 确认一下, 会提示你找不到命令.

如果是用 `*.run` 文件, `sudo ./NVIDIA-Linux-x86_64-384.59.run --uninstall`, 用这个命令删除原有驱动.

在删除后, hdmi 插在独显后不会有任何输出的, 这时我开始尝试重新安装显卡驱动. 注意, 重点来了, 安装 NVIDIA 显卡驱动的时候, 一定要 **把显卡先插好**, 否则你的驱动安装一定会在一开始提醒你找不到 graphics driver, 且在最后提醒你 cannot insert module (显卡驱动是一个 kernel module), 要知道自己的显卡到底插进去没, 很简单 `lspci | grep VGA`, 然后去看看到底你的独显有没有出现, 在我的情况下, 如果插在集成显卡上, 就会显示两个 `VGA compatible controller`, 这个问题非常重要, 因为我最后才发现, 我的一直以来的一个问题就是...显卡没插好, 恩, 就是这么狗血, 我竟然还是在想去 bios 关了集显时, 发现了我的独显没插进去, 你说说这圈子绕的. 插独显的时候, 对准 pci 槽插!

在这一部分, 我还遇到了 `gcc version` 不合适的问题, 如果我指定 `CC=4.5`, 则在最后会提示我 `cannot find kernel source` (大概是这个意思), 我用学长给我的一个版本较老的驱动, 解决了版本不合适的问题, 至于后一个问题, *我暂时还不能认定* 是因为没插独显造成的, 还是因为换了驱动造成的.

值得一提的是, 在我把 hdmi 线插到集显上后, 突然莫名其妙可以进入图形界面了, 诡异的是竟然不用输入密码... 此时我的心情好了一点, 但是显卡驱动仍然没有装好.

在解决以上两个大坑后, 来正式讲讲如何安装驱动, 首先我们需要禁用 `nouveau`, 这玩意我也没仔细了解, 总之是开源社区搞出来的, ubuntu 预装的驱动, 会影响 nvidia 驱动的安装, 禁用它还需要几步小操作.

1. `sudo vi /etc/modprobe.d/blacklist-nouveau.conf`
2. 加入如下内容
    ```bash
    blacklist nouveau
    options nouveau modeset=0
    ```
3. `sudo update-initramfs -u`
4. `sudo reboot`
5. 重新启动之后, `lsmod | grep nouveau*`, 看一下是否禁用成功

确认装好显卡, 禁用 `nouveau` 之后, 我们还需要暂时关闭 display manager `sudo service lightdm stop`, 在图形界面下, 没法安装显卡驱动

完成上面的步骤后, 我们可以安装驱动咯, 我是通过 `*.run` 文件安装的, 一路 continue 下去之后, 终于成功了, 此时 `nvidia-smi` 有了输出, 但是显示我们的独显并没有干活, 在电脑原来的 `CUDA-NVIDIA-examples` 里有一个 `1_Utilities/deviceQuery` 文件夹, `./deviceQuery`, 这个命令就可以确定一下你的 NVIDIA 显卡到底有木有干活, 这时, 它并没有干活. 而且, 整个登录界面的分辨率完全垮掉, 也回到了无法进入图形系统的情况.

`lspci` `lspci -k` 分别看了一下, 初步判断可能是集显把独显压制了, 把 hdmi 重新插回独显, 重启电脑, 终于解决了问题.... 如果还不行, 可能需要在 blacklist 中再加入你在 `lspci -k` 中看到的集显对应的 kernel module 的名字. 在成功进入图形界面后, 我们再 `lspci` 就只有一个 vga controller 了, 独显开始干活咯, 集显下班!

再次 `./deviceQuery` 之后, 提示我的是版本不符了~ 重新下个 cuda 应该就好~

## CUDA 安装

一开始采用了 `cuda_10.0.130_410.48_linux.run` 注意后面的 `410.48` 就是需要对应的 **驱动版本**, 而我的版本是 384.120, 所以在 deviceQuery 的时候, 爆出了 `insufficient` 的错误, 重新下载 9.2 版本即可





## 知识点

- `lspci -k` 还可以具体显示你的 driver 对应于哪些 kernel module
Please make sure that
 -   PATH includes /usr/local/cuda-10.0/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-10.0/lib64, or, add /usr/local/cuda-10.0/lib64 to /etc/ld.so.conf and run ldconfig as root

path 的初始定义在 `/etc/enviroment` 中
 
dkkg -i 安装包才好使
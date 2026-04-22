# LFPIC
[TMM 2025] Learned Focused Plenoptic Image Compression with Local-Global Correlation Learning, [paper](https://ieeexplore.ieee.org/abstract/document/10856419), by Gaosheng Liu, Huanjing Yue, Bihan Wen, and Jingyu Yang.

The dense light field sampling of focused plenoptic images (FPIs) yields substantial amounts of redundant data, necessitating efficient compression in practical applications. However, the presence of discontinuous structures and long-distance
properties in FPIs poses a challenge. In this paper, we propose a novel end-to-end approach for learned focused plenoptic image compression (LFPIC). Specifically, we introduce a local-global correlation learning strategy to build the nonlinear transforms.
This strategy can effectively handle the discontinuous structures and leverage long-distance correlations in FPI for high compression efficiency. Additionally, we present a spatial-wise context model tailored for LFPIC to help emphasize the most related symbols during coding and further enhance the rate-distortion performance. Experimental results demonstrate the effectiveness of our proposed method, achieving a 22.16% BD-rate reduction (measured in PSNR) on the public dataset compared to the recent state-of-the-art LFPIC method. This improvement holds significant promise for benefiting the applications of focused plenoptic cameras. 


The updated checkpoints for 6 lambda parameters can be downloaded via https://pan.baidu.com/s/1wRKul1dklLDR5X7hDs_3lw, extraction code:flfc


## Citation
If you find this work helpful, please consider citing the following papers:<br> 
```Citation
@article{liu2025learned,
  title={Learned Focused Plenoptic Image Compression With Local-Global Correlation Learning},
  author={Liu, Gaosheng and Yue, Huanjing and Wen, Bihan and Yang, Jingyu},
  journal={IEEE Transactions on Multimedia},
  volume={27},
  pages={1216--1227},
  year={2025},
  publisher={IEEE}
}
@article{10120973,
  title={Learned Focused Plenoptic Image Compression with Microimage Preprocessing and Global Attention}, 
  author={Tong, Kedeng and Jin, Xin and Yang, Yuqing and Wang, Chen and Kang, Jinshi and Jiang, Fan},
  journal={IEEE Transactions on Multimedia},   
  year={2023},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TMM.2023.3272747}}
```
## Acknowledgement
Our work and implementations are based on the following projects: <br> 
[LF-DFnet](https://github.com/YingqianWang/LF-DFnet)<br> 
[LF-InterNet](https://github.com/YingqianWang/LF-InterNet)<br> 
We sincerely thank the authors for sharing their code and amazing research work!

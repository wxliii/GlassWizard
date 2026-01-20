# GlassWizard

## Requirements
Please refer to [Marigold](https://github.com/prs-eth/Marigold).

## Prepare Datasets and Checkpoints

You can refer to the following repositories and their papers for the detailed configurations of the corresponding datasets.
- GSD. Please refer to [GSD](https://drive.google.com/file/d/1W6OZ3OW26sDPuI6CnGAa0xoqR06IGnQj/view?usp=sharing)
- Trans10k. Please refer to [Trans10k](https://xieenze.github.io/projects/TransLAB/TransLAB.html)
- GDD. Please refer to [GDD](https://mhaiyang.github.io/CVPR2020_GDNet/index.html)
- HSO. Please refer to [HSO](https://mhaiyang.github.io/TIP2022-PGSNet/index.html)

Please download the SD v2 checkpoint from [checkpoint](https://huggingface.co/stabilityai/stable-diffusion-2) and download the pretrained textual embedding from [text embedding](https://drive.google.com/file/d/1685MapSch6k7s6BeIHC2uTshEI3AsZnK/view?usp=sharing). (SD v2 checkpoint backup link from [Marigold](https://share.phys.ethz.ch/~pf/bingkedata/marigold/checkpoint/stable-diffusion-2.tar))

Put these checkpoints into ./weights folder

## Results

Our prediction maps can be found on [Google Drive](https://drive.google.com/file/d/1PtYWrFRD9qlUk4BhvoWHmvJ20YuLye58/view?usp=sharing). 

The weights can be downloaded on [HuggingFace](https://huggingface.co/wxli318/GlassWizard). Please move the diffusion_pytorch_model.bin to the unet folder.




## Evaluation

For evaluation, 
```shell
python eval.py
```
We would like to clarify the F-measure metric used in this repository and in the paper.

## Acknowledgement
[Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2)

[Textual Inversion](https://github.com/rinongal/textual_inversion)

[Marigold](https://github.com/prs-eth/Marigold)

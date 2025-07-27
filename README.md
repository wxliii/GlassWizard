# GlassWizard

## Requirements
```shell
pip install -r requirements.txt
```

## Prepare Datasets and Checkpoints

You can refer to the following repositories and their papers for the detailed configurations of the corresponding datasets.
- GSD. Please refer to [GSD](https://drive.google.com/file/d/1W6OZ3OW26sDPuI6CnGAa0xoqR06IGnQj/view?usp=sharing)
- Trans10k. Please refer to [Trans10k](https://xieenze.github.io/projects/TransLAB/TransLAB.html)
- GDD. Please refer to [GDD](https://mhaiyang.github.io/CVPR2020_GDNet/index.html)
- HSO. Please refer to [HSO](https://mhaiyang.github.io/TIP2022-PGSNet/index.html)

Please download the SD v2 checkpoint from [checkpoint](https://huggingface.co/stabilityai/stable-diffusion-2)

## Results

Our prediction maps can be found on [Google Drive](https://drive.google.com/file/d/1PtYWrFRD9qlUk4BhvoWHmvJ20YuLye58/view?usp=sharing). 

The can be downloaded on [Google Drive]().




## Evaluation

For evaluation, 
```shell
python eval.py
```

## Acknowledgement
[Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2)

[Textual Inversion](https://github.com/rinongal/textual_inversion)

[Marigold](https://github.com/prs-eth/Marigold)

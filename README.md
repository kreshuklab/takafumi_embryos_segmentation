# takafumi_embryos_segmentation
This repository contains a configuration files used to reproduce the segmentation experiments from the paper [An ex vivo system to study cellular dynamics underlying mouse peri-implantation development(https://www.sciencedirect.com/science/article/pii/S1534580721010431):
```
@article{ICHIKAWA2022,
title = {An ex vivo system to study cellular dynamics underlying mouse peri-implantation development},
journal = {Developmental Cell},
year = {2022},
issn = {1534-5807},
doi = {https://doi.org/10.1016/j.devcel.2021.12.023},
url = {https://www.sciencedirect.com/science/article/pii/S1534580721010431},
author = {Takafumi Ichikawa and Hui Ting Zhang and Laura Panavaite and Anna Erzberger and Dimitri Fabrèges and Rene Snajder and Adrian Wolny and Ekaterina Korotkevich and Nobuko Tsuchida-Straeten and Lars Hufnagel and Anna Kreshuk and Takashi Hiiragi},
keywords = {mouse embryonic development, embryo implantation, egg cylinder formation, epiblast morphogenesis, lumen formation, tissue-tissue interaction, mechano-chemical interplay, embryo culture,  live-imaging, quantitative image analysis}
}

```

Neural networks used to predict the embryo cell boundaries were trained with [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet) (see configuration files in [unet_configs](unet_configs)).

Given the cell boundary network the final segmentation results were computed using [plant-seg](https://github.com/hci-unihd/plant-seg).
Relevant plant-seg configuration files can be found in [plantseg_configs](plantseg_configs).

For networks trained with sparsely annotated embryo cells used the [SPOCO](https://github.com/kreshuklab/spoco) method
SPOCO configuration files can be found in [spoco_configs](spoco_configs).

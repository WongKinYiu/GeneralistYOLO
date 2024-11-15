# GeneralistYOLO
Generalist YOLO: Towards Real-Time End-to-End Multi-Task Visual Language Models

<div align="center">
    <a href="./">
        <img src="./figures/GeneralistYOLO.png" width="79%"/>
    </a>
</div>


## Performance 

#### Generalist Visual Language Tasks

| Model | Size | \#Param. | FLOPs |  AP<sub>e2e/nms</sub><sup>box</sup> | AP<sub>e2e/nms</sub><sup>mask</sup>  | mIoU<sub>164k/10k</sub><sup>semantic</sup>  | mIoU<sup>stuff</sup> | PQ<sup>panoptic</sup> | BLEU@4<sup>caption</sup> | CIDEr<sup>caption</sup> |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| [**GeneralistYOLO**]() | 640 | 37.3M | \~195.2G | **52.3%/52.7%** | **42.9%/43.1%** | **44.4%/51.7%** | **59.4%** | **44.3%** | **38.5** | **121.0** |
<!--| [****]() |  | M | G | **%/%** | **%/%** | **%/%** | **%** | **%** | **** | **** |-->

## Installation 

#### Generalist Visual Language Tasks

...

## Evaluation

#### Generalist Visual Language Tasks

...

## Training  

#### Generalist Visual Language Tasks

...

## Citation

```
@inproceedings{wang2024yolov9,
  title={{Generalist YOLO}: Towards Real-Time End-to-End Multi-Task Visual Language Models},
  author={Chang, Hung-Shuo and Wang, Chien-Yao and Wang, Richard and Chou, Gene and Liao, Hong-Yuan Mark},
  booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2025}
}
```

## Coming Soon

<details><summary> <b>Coming soon</b> </summary>

* Visual Grounding
* Whole Body Pose Estimation
* Video Understanding Tasks
* Specific Vision Tasks

</details>

## Contributors

<details><summary> <b>Contributors</b> </summary>
    
* @ws6125 : Most of generic vision tasks and visual language tasks.
* @110598074 : Most of specific vision tasks.
* @F84064014 : Video panoptic segmentation.
* @k22708038 : Visual grounding.
* @HuChengXue : Whole body pose estimation.
* ...

</details>


## Acknowledgements

<details><summary> <b>Reference Codes</b> </summary>

* [https://github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)
* [https://github.com/davidnvq/grit](https://github.com/davidnvq/grit)
* [https://github.com/djiajunustc/TransVG](https://github.com/djiajunustc/TransVG)
* [https://github.com/Robertwyq/PanoOcc](https://github.com/Robertwyq/PanoOcc)
* [https://github.com/junjiehe96/FastInst](https://github.com/junjiehe96/FastInst)
* [https://github.com/fan23j/yolov7-pose-whole-body](https://github.com/fan23j/yolov7-pose-whole-body)
* ...

</details>

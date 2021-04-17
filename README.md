# sRender
We provide `Pytorch` implementation for sRender: rendering facial sketches 

Citation:

>Meimei Shang, **Fei Gao** *, Xiang Li, Jingjie Zhu, Lingna Dai. **Bridging Unpaired Facial Photos And Sketches By Line-drawings**. 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 6-11 June 2021, Toronto, Ontario, Canada. (Accepted)

[[paper@arXiv]](https://arxiv.org/abs/2102.00635)   [[project@github]](http://aiart.live/sRender/)

## Pipeline

![Reconstructed sketches](/images/pipeline.png)

## Sketch Reconstruction

**left:** croquis style; **right:** charcoal style
![Reconstructed sketches](/images/sketch2sketch.png)
(a)Sketches  ; (b)Generated Line-drawings ; (c)Synthesized Sketches
## Oil Painting Reconstruction

![Reconstructed oil paintings](/images/oil_painting_reconstruct.png)
(a)Oilpaintings  ; (b)Generated Line-drawings ; (c)Synthesized Oil Paintings
## Sketch Generation (from photos)

**left:** croquis style; **right:** charcoal style
![Generated sketches](/images/photo2sketch.png)
(a)Photos ; (b)Generated Line-drawings ; (c)Synthesized Sketches
## Oil Painting Generation (from photos)

![Generated oilpaintings](/images/oil_painting_generation.png)
(a)Photos ; (b)Generated Line-drawings ; (c)Synthesized Oil Paintings
## Comparison

- Croquis sketches generated by our sRender and unpaired I2I translation methods

![compare with SOTA](/images/sota.jpg)
## Train and test

* `charcoal_style` for systhesising charcoal style images,it contains **sRender w/o Lstr** for model without stroke_loss and **sRender** for model with stroke_loss correspodingly
* `croquis_style` for systhesising croquis style images,it contains **sRenderPix2Pix**, **sRender w/o Lstr** and **sRender**
* download dataset [gray](https://drive.google.com/drive/folders/1ZuRVlPwvtNtfkNIj-DIdi2kr-1kEcPhg?usp=sharing) for **charcol_style**, [binary](https://drive.google.com/drive/folders/1VBUBdGWz324dhCu8LRFU5qB0PqXxNQIJ?usp=sharing) for **croquis_style**
* download pretrain model and put them in **./checkpoints** for test
* for model with stroke_loss, you should download [stroke_model](https://drive.google.com/file/d/16gSERA3TbPVFyCvKGtNKtJrQaOsG8vmO/view?usp=sharing) and specify model root in
  **./models/pix2pixHD_model** for **net_c.load_state_dict**
* you can modify **options/base_option** to specify **--dataroot**, then run train.py or test.py 


## Dataset

- Gray line-drawings for charcol_style from GooleDrive
  - [charcoal](https://drive.google.com/file/d/18MAVm-u0l4Rfh4Ct6bZRG1cNbKwQMgMt/view?usp=sharing)
  - [oil paintings]
  - [CelebAmaskHQ](https://drive.google.com/file/d/16pOkZiaBrot9EeLvxBIqm_UTuI2GSjpA/view?usp=sharing)
  - [FFHQ](https://drive.google.com/file/d/1mC0Vzf6TLD-77vtkzmfmVf3ZZXRlyOVb/view?usp=sharing)  
- Binary line-drawings for croquis_style from GooleDrive
  - [croquis](https://drive.google.com/file/d/1EMzyVvJnYhmyBMnriymgfFd_WCaURfto/view?usp=sharing)
  - [CelebAmaskHQ](https://drive.google.com/file/d/1euiF1197sOEa6_dM6qE4tPMt0V42Vn-6/view?usp=sharing)
  - [FFHQ](https://drive.google.com/file/d/1uJQ5JGttXLfwmpH5LH0yH3bgs2oT7kTa/view?usp=sharing)  

## Pretrained models

- charcoal_style from GooleDrive
  - [sRender w/o Lstr](https://drive.google.com/file/d/1mwGiFpXfMlcUw-ksfsyVhUSKQWQJGC6p/view?usp=sharing)
  - [sRender](https://drive.google.com/file/d/1jwxvZZAJ-0gr_XJX3i3rBNQYqh8FCRCI/view?usp=sharing)  
- croquis_style from GooleDrive
  - [sRenderPix2Pix](https://drive.google.com/file/d/1GIRcc8q-plIXKxSDEug4UMXacB35w0G5/view?usp=sharing)
  - [sRender w/o Lstr](https://drive.google.com/file/d/1JdVhJDVCcFQ1jtNfNy-Q05UL4IVqkqw3/view?usp=sharing)
  - [sRender](https://drive.google.com/file/d/1AKyX1u7RieCwP8b8WEUOzkXufLgRotr0/view?usp=sharing)  
- charcoal_style from GooleDrive
  - [sRender]
## Results

* Our synthesis result for **croquis** and **charcoal** style can be downloaded from 
  [Goole Drive](https://drive.google.com/drive/folders/1rDEe1GhBuoPUKDlj6kflfG1FTR6Xhu4u?usp=sharing)
* Our synthesis result for **oil painting** style can be downloaded from 
  [Goole Drive]
## Acknowledgments

* Our code is inspired by the [NVIDIA/pix2pixHD](https://github.com/NVIDIA/pix2pixHD) repository.

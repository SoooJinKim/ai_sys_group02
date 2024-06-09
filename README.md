# ai_sys_group02
This project is to restore the hidden face from the masked low-quality facial image. Unlike previous studies that used Western-oriented images, this was conducted using Korean images. For details see our [paper]()

## Data Preprocessing
Dataset can be downloaded [here](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71427)

![unmask](https://github.com/SoooJinKim/ai_sys_group02/blob/main/example/image_example.png)
![data preprocessing](https://github.com/SoooJinKim/ai_sys_group02/blob/main/example/data_preprocess.png)

To create a CCTV-like image, we converted the high-resolution image to low-resolution and applied Gaussian noise and Gaussian blur.

The Dlib library was used to detect facial contours, and the OpenCV library was used to mask from the center of the nose to jawline.

## Training

## Citation

    @unpublished{aisys2,
    title={Masked facial image restoration to distinguish identical person on CCTV using high-resolution transform and GAN},
    author={Kim, Serin and Kim, Sujin and Kim, Johyeon and Song, Taewon},
    year={2024},
    howpublished={https://github.com/SoooJinKim/ai_sys_group02/}
    }


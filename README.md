# FC-TFG 
### export PYTHONPATH="./" sets the current directory as project root.

## AudioEncoder : 

        For the audio source, we downsample the audio to 16kHz, then convert the downsampled audio to mel-spectrograms with a window size of 800, 
        a hop length of 200, and 80 Mel filter banks.

        1. KR Prajwal, Rudrabha Mukhopadhyay, Vinay P Namboodiri, and CV Jawahar. 2020. A lip sync expert is all you need for speech to lip generation in the wild. 
        InProc. ACM MM. 484–492. ---------------------------------------------------------- Wav2Lip

## Visual Encoder: StyleGAN Inversion Network

        First of all, we pre-train a StyleGAN2 [24] generator on the VoxCeleb2 dataset and then train HyperStyle [1] inversion network with the pre-trained StyleGAN2 model. Specifically, we replace the e4e [46] encoder in the HyperStyle model with pSp [35]encoder.

        1. Yuval Alaluf, Omer Tov, Ron Mokady, Rinon Gal, and Amit Bermano. 2022.
        Hyperstyle: Stylegan inversion with hypernetworks for real image editing. In
        Proc. CVPR. 18511–18521 ----------------------------------------------------------- HyperStyle

        2. Omer Tov, Yuval Alaluf, Yotam Nitzan, Or Patashnik, and Daniel Cohen-Or. 2021.
        Designing an encoder for stylegan image manipulation. ACM Transactions on
        Graphics 40, 4 (2021), 1–14  ------------------------------------------------------ e4e

        3. Elad Richardson, Yuval Alaluf, Or Patashnik, Yotam Nitzan, Yaniv Azar, Stav
        Shapiro, and Daniel Cohen-Or. 2021. Encoding in style: a stylegan encoder for
        image-to-image translation. In Proc. CVPR. 2287–2296. ----------------------------- pSp


###    CanonicalEncoder : 2 Layer MLP

###    TemporalFusion : Single 1D convolution Layer

###    Decoder : StyleGAN2 Generator

###    Discriminaotr : StyleGAN2 Discriminator

###    MotionEncoder : 3 Layer MLP


### Steps :

        1. First Train StyleGAN2 Generator
        2. Train pSp encoder using HyperStyle code ( replace e4e encoder in HyperStyle with pSp encoder ) with pretrained Generator from step 1
        3. Train whole Network end to end with pSp encoder as Visual Encoder and StyleGAN2 Generator as Decoder.

### Losses :
        1. Orthogonality Loss : To disentangle latent space
        2. Sync Loss : from SyncNet
        3. Identity Loss : From Arcface
        4. Reconstruction Loss : L1
        5. Perceptual Loss : From VGG19
        6. Adversarial Loss : GAN Loss

## Pretrained models required ##

### SyncNet :   
        1. Joon Son Chung and Andrew Zisserman. 2017. Out of time: automated lip sync in the wild. 
        In Proc. ACCV. Springer, 251–263.

### VGG : 
        1. Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The unreasonable effectiveness of deep features as a perceptual metric. InProc. CVPR. 586–595.

### ArcFace : 
        1. Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. 2019. Arcface:
        Additive angular margin loss for deep face recognition. In Proc. CVPR. 4690–4699.
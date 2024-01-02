# FC-TFG 
### export PYTHONPATH="./" sets the current directory as project root.

## AudioEncoder : 

        For the audio source, we downsample the audio to 16kHz, then convert the downsampled audio to mel-spectrograms with a window size of 800, 
        a hop length of 200, and 80 Mel filter banks.

        1. KR Prajwal, Rudrabha Mukhopadhyay, Vinay P Namboodiri, and CV Jawahar. 2020. A lip sync expert is all you need for speech to lip generation in the wild. 
        InProc. ACM MM. 484–492. ---------------------------------------------------------- Wav2Lip
        
        2 . https://towardsdatascience.com/audio-deep-learning-made-simple-part-1-state-of-the-art-techniques-da1d3dff2504
        
#### MFCC: Mel Frequency Cepstral Coefficients

        Mel Spectrograms work well for most audio deep learning applications. However, for problems dealing with human speech, like Automatic Speech Recognition,
        you might find that MFCC (Mel Frequency Cepstral Coefficients) sometimes work better.These essentially take Mel Spectrograms and apply a couple of further 
        processing steps. This selects a compressed representation of the frequency bands from the Mel Spectrogram that correspond to the most common frequencies
         at which humans speak.


#### SpecAugment: 
        
        The normal transforms you would use for an image don’t apply to spectrograms. For instance, a horizontal flip or a rotation would substantially alter the
        spectrogram and the sound that it represents.Instead, we use a method known as SpecAugment where we block out sections of the spectrogram.
        There are two flavors:

        1. Frequency mask — randomly mask out a range of consecutive frequencies by adding horizontal bars on the spectrogram.
        2. Time mask — similar to frequency masks, except that we randomly block out ranges of time from the spectrogram by using vertical bars.


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


##   TemporalFusion : Single 1D convolution Layer 
        Apply a single 1D Conv over temporal features from different modalities independently. Then Fuse these modalities together.

        Temporal aggregation : Combining information over time to create a more compact representation of temporal data.
                               This process is often used to reduce the temporal resolution of a time series or sequence
                               while retaining essential information. eg., Applying pooling on temporal data (along channelwise).

        Modality Fusion : eg., Concatenating diffent modality features and applying a Linear layer.

        1. EPIC-Fusion: Audio-Visual Temporal Binding for Egocentric Action Recognition
        2. TCN : Temporal Convolutional Networks: A Unified Approach to Action Segmentation

###    CanonicalEncoder : 2 Layer MLP

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


### Status :
        1. Canonical Encoder : Done (approx. , need to figure out the layer and sizes)
        2. Motion Encoder : Done (approx. , need to figure out the layer and sizes)
        3. Temporal Fusion : Need to understand the concept behind it. (Number of layer)
        4. Decoder : Done (approx. , need to figure out the layer and sizes)
        5. Visual Encoder : Done (approx. , need to figure out the layer and sizes)
        6. Audio Encoder : Need to understand the concept behind it.

### To Do next: 
        1. Audio Encoder - Reading paper referenced (read Wav2Lip)
        2. Temporal Fusion - Finding paper related to it (process)

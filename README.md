
# VQGAN + Transformer: Text-to-Image From Scratch (TensorFlow/Keras)

A complete from-scratch implementation of the VQGAN + Transformer architecture for the Text-to-Image generation task — built entirely using TensorFlow and Keras, with no reliance on pretrained models.

This project walks you through building the model components step-by-step — from vector quantization to transformer training — and includes training scripts, model weights, and guidance to reproduce the results from the ground up. Perfect for those who want to understand and implement the full pipeline themselves.


## Run Locally

Clone the project

```bash
  git clone https://github.com/Beastojenisto/VQGAN-Transformer.git
```

Go to the project directory

```bash
  cd VQGAN-Transformer
```

Install dependencies

```bash
  pip install -r requirements.txt
```


## Optimizations

### Perceptual Loss
Added a Resnet50 based Perceptual Loss in the VQGAN to make the outputs more semantically similar to the input.

### Latent Basis
The VQGAN is very sensitive to the codebook initialization. The codebook vectors used earlier training gets the gradient while the unused codebook vectors remains the same and does not improve which leads to low codebook usage. To solve this problem I used a learnable latent basis and a dense projection layer. The latent basis is passed through the dense projection layer to get the codebook. Thus when updating one vector of the codebook, all the vectors of the learnable basis are updated(Since each codebook vector is the weighted sum of all the vectors of latent basis). Before using Latent Basis the codebook usage was ~30%. After using Latent Basis the codebook usage became ~90%.

### High Frequency Loss
Added a High Frequency Loss to make the outputs more sharp(Used Chatgpt).

### Adversarial Loss
Added Adversarial Loss using a GAN setup to improve output realism. This helps the model generate sharper and more natural-looking images by encouraging outputs that are harder for the discriminator to distinguish from real ones.

### KL Loss(Uniform Distribution)
To ensure all the codebook vectors are uniformly used, KL loss has been implemented. Thus overusage of a small group of vectors is reduced.




## Lessons Learned

- Different Loss functions have different impact on the output.
- GANs are very unstable and can mode collapse quite often as the VQGAN has collapsed to using 1 codebook index quite some times without any reason
- Setting the seed of the environment is important for reproducability
- Ensuring all the indices get gradient can improve the performance and reconstructions of the VQGAN
- Capacity of the models and the amount of data plays a vital role in the performance of the models
- Patience is the key to success (This project took me 5 months to complete)


## Demo

You can visit a simple streamlit demo of the VQGAN transformer [HERE](https://huggingface.co/spaces/Beasto/Cursed-Text-to-Image).

## Screenshot

![Your internet is Bad](./generation.png)

## Note 

There may be bugs in the code and the outputs of the model may be bad as this model was trained on a relatively small dataset(Flickr30k) on a relatively small model(VQGAN's parameters : 60 m , Transformer's parameters : 60 m) on kaggle's TPU(Has performance roughly equivalent to one A100 GPU). 

And was build by a 14 y/o :)

Any feedback will be highly appreciated!!

Thank you!!

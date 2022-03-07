# Variational X-Ray Report Generation

[[arXiv]](https://arxiv.org/pdf/2107.07314.pdf) [[Springer]](https://link.springer.com/chapter/10.1007/978-3-030-87199-4_59)

### This is the official code repository for "Variational Topic Inference for Chest X-Ray Report Generation" 
### Oral at MICCAI 2021

The paper proposes variational topic inference (VTI), which addresses report generation for chest X-ray images with a probabilistic latent variable
model. VTI uses a set of latent variables z, each defined as a topic governing the sentence generation. It is optimized by maximizing the evidence lower bound objective (ELBO) During training, the topics are inferred from visual and language representations, which are aligned by minimizing the KL divergence between them. By doing so, at test time the model is able to infer topics from the visual representations to generate the sentences. Also, it adopts visual attention to attend to different local image regions when generating words.

## Architecture
![model](https://github.com/ivonajdenkoska/variational-xray-report-gen/blob/main/model.png)

## Usage

- The Indiana U. Chest X-ray dataset or MIMIC-CXR should be downloaded and placed on `data/indiana_chest_xrays` or `data/mimic_cxr` correspondingly. 
- Run the preprocessing scripts: `src/preprocess_indiana.py` or `src/preprocess_mimic.py` to create the appropiriate train/val/test partitions (which will be created on `data/indiana_chest_xrays/data_splits` or `data/mimic_cxr/data_splits`). 
- Then, to train and evaluate the VTI model, simply run `src/main_cvae.py`. The hyperparameters, like batch size, number of epochs, learning rate, dropout rate, and all paths, can be edited on `src/config.py`.


## Reference
If you find this code or the paper useful for your own work, please cite:
```
@inproceedings{najdenkoska2021variational,
  title={Variational Topic Inference for Chest X-Ray Report Generation},
  author={Najdenkoska, Ivona and Zhen, Xiantong and Worring, Marcel and Shao, Ling},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={625--635},
  year={2021},
  organization={Springer}
}
```

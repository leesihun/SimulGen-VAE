# Plans to implemment PCA for dimensionality reduction

Leave the VAE part as it is. The current plan is to only change latent conditioner, not other parts. You are allowed to change the input stage and latent conditioner-related part of SimulGen-VAE.py and laetnt_conditioner.py, etc.

## Enable PCA_MLP mode
- If in condition.txt, input_type == image  & use_pca == 1, enable PCA_MLP mode.
- When PCA_MLP mode is enabled, call read_latent_conditioner_dataset_img_pca.

## Latent conditioner
- Generate a function read_latent_conditioner_dataset_img_pca that imports images and converts them using pca.
- Get the pca coefficients and use them as input to the latent conditioner
- Use latent_conditioner_model_parametric for training.

## Cleanup
- After these steps are completed, delete the pca part in cnn version and cleanup the whole code base
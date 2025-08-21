import fiftyone as fo
import fiftyone.brain as fob
import numpy as np
from load_dataset import Ds 

# Step 1: Load / create dataset
Ds().load()  # This will create/populate the dataset

# Step 2: Re-load the dataset after creation
dataset = fo.load_dataset("AIC_dataset")

# Step 3: Launch FiftyOne app
session = fo.launch_app(dataset, port=3000)


clip_embeddings = np.load("clip-features-32/L21_V001.npy")
dataset.set_values("clip_embedding", [emb.tolist() for emb in clip_embeddings])

fob.compute_similarity(
    dataset,
    model="clip-vit-base32-torch",
    embeddings="clip_embedding", 
    brain_key="img_sim",
)

# Step 5: Query and update view
query = "classroom"
view = dataset.sort_by_similarity(query, k=10, brain_key="img_sim")
session.view = view

session.wait(-1)


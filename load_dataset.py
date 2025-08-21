import fiftyone as fo
import os

ds_name = "AIC_dataset"
if ds_name in fo.list_datasets():
    dataset = fo.load_dataset(ds_name)
else:
    dataset = fo.Dataset(ds_name)

dataset.persistent = True

file_path = "keyframes/L21_V001"
samples = []
for file in os.listdir(file_path):
    path = f"{file_path}/{file}"
    sample = fo.Sample(filepath = path)
    sample["ground_truth"] = fo.Classification(label = "video")
    samples.append(sample)

dataset.add_samples(samples)
dataset.save()

print(fo.list_datasets())


import fiftyone as fo

dataset = fo.load_dataset("AIC_dataset")

session = fo.launch_app(dataset, port=3000)
session.wait(-1)
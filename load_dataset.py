import fiftyone as fo
import os
import json

ds_name = "AIC_dataset"
if ds_name in fo.list_datasets():
    dataset = fo.load_dataset(ds_name)
    fo.delete_dataset(ds_name)

dataset = fo.Dataset(ds_name)

dataset.persistent = True

frame_path = "keyframes/L21_V001"
object_path = "objects/L21_V001"
frame_files = os.listdir(frame_path)
object_files = os.listdir(object_path)

def add_objects(object_file):
    with open(object_file, "r") as file:
        objects = json.load(file)

    detections = []
    for object_index in range(0, len(objects["detection_scores"])):
        detection = fo.Detection(
            label= objects["detection_class_entities"][object_index],
            bounding_box= objects["detection_boxes"][object_index],
            confidence = objects["detection_scores"][object_index]
        )
        detections.append(detection)
    return detections

def add_tags(video_name):
    path = f"media-info/{video_name}.json"
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data["keywords"]

samples = []
for f1, f2 in zip(frame_files, object_files):  # pair by index
    frame_file = os.path.join(frame_path, f1)
    object_file = os.path.join(object_path, f2)

    video_name = frame_path.split("/")[1]

    sample = fo.Sample(filepath=frame_file)

    sample["video"] = fo.Classification(label = video_name)
    sample["entities"] = fo.Detections(detections = add_objects(object_file))
    sample["tags"] = add_tags(video_name)

    samples.append(sample)

dataset.add_samples(samples)
dataset.save()

print(fo.list_datasets())


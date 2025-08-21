import fiftyone as fo
import os
import json

class Ds:
    def load(self):
        ds_name = "AIC_dataset"

        # Delete and recreate dataset for clean start
        if ds_name in fo.list_datasets():
            fo.delete_dataset(ds_name)
        dataset = fo.Dataset(ds_name)
        dataset.persistent = True

        frame_path = "keyframes/L21_V001"
        object_path = "objects/L21_V001"

        frame_files = sorted(os.listdir(frame_path))
        object_files = sorted(os.listdir(object_path))

        def add_objects(object_file):
            with open(object_file, "r", encoding="utf-8") as file:
                objects = json.load(file)

            detections = []
            for idx, score in enumerate(objects["detection_scores"]):
                if float(score) >= 0.3:
                    detection = fo.Detection(
                        label=objects["detection_class_entities"][idx],
                        bounding_box=objects["detection_boxes"][idx],
                        confidence=score
                    )
                    detections.append(detection)
            return detections

        def add_tags(video_name):
            path = f"media-info/{video_name}.json"
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            return data.get("keywords", [])

        samples = []
        for f1, f2 in zip(frame_files, object_files):
            frame_file = os.path.join(frame_path, f1)
            object_file = os.path.join(object_path, f2)
            video_name = os.path.basename(frame_path)  # safer way

            sample = fo.Sample(filepath=frame_file)
            sample["video"] = fo.Classification(label=video_name)
            sample["entities"] = fo.Detections(detections=add_objects(object_file))
            sample["tags"] = add_tags(video_name)

            samples.append(sample)

        dataset.add_samples(samples)
        dataset.save()

        print("Datasets:", fo.list_datasets())

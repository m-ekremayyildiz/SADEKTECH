from roboflow import Roboflow
rf = Roboflow(api_key="ZwrZx0MJoJRdkXHRiOyB")
project = rf.workspace("freelancer-gucsh").project("yolov8-poktl")
dataset = project.version(1).download("yolov8")

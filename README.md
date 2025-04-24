# RetinaFace-in-PyTorch

## 🔍 Face Detection using RetinaFace (ResNet50 Backbone)

This project integrates a powerful deep learning-based **face detection** system using the **RetinaFace** model with a **ResNet50** backbone, derived from the official [Pytorch RetinaFace repository](https://github.com/biubug6/Pytorch_Retinaface).

### 📌 What is RetinaFace?

[RetinaFace](https://github.com/biubug6/Pytorch_Retinaface) is a single-stage dense face detector developed by InsightFace that is capable of:

- Detecting faces with high precision and speed.
- Predicting 5 facial landmarks (eyes, nose tip, mouth corners).
- Detecting faces under extreme poses, occlusions, and lighting conditions.

### 🧠 Backbone Used

In this project, we specifically use the **ResNet50** backbone for the RetinaFace detector.

- **ResNet50** provides deeper and more robust feature extraction compared to lightweight alternatives like MobileNet.
- It is ideal for use cases where detection accuracy is more important than inference speed.

### 🎯 Features

- Face detection on **images** and **video streams**.
- Uses pre-trained RetinaFace weights for ResNet50.
- Supports **CUDA** acceleration if available.
- Returns bounding boxes and landmarks.

### 🏁 How to Run

Make sure you have the `Resnet50_Final.pth` model weights in the correct path.

Pretrain model and trained model are put in google drive(https://drive.google.com/drive/folders/1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1). The model could be put as follows:

  ./weights/
  
      mobilenet0.25_Final.pth
      
      mobilenetV1X0.25_pretrain.tar
      
      Resnet50_Final.pth

##Command to run the .py file

python detect.py --trained_model weights/Resnet50_Final.pth --network resnet50

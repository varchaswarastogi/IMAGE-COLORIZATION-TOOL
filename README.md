Image Colorization Tool

This project is a Flask-based web application that converts black & white images into color using a pre-trained deep learning model (OpenCV DNN + Caffe).

---

## Features

* Upload grayscale OR B/W images
* Automatic colorization using AI model
* Input/output display
* Clean and interactive web interface

---

## ⚠️ Important Note (Read Before Running)

> The trained model used in this project is **large (~120MB)** and cannot be reliably hosted on free cloud platforms due to memory and file size limitations.
> CAFFE MODEL LINK : {https://drive.google.com/file/d/1ugborSR3Dixb8WwtwPuLZ2FSutIGpUNB/view?usp=sharing}
> **For this reason, the model is NOT included in this repository and must be downloaded separately.**

> It is recommended to run this project **locally on your system** for proper functionality.

---

## Project Structure

```
colorizer_app/
│
├── app.py
├── models/
│   ├── colorization_deploy_v2.prototxt
│   ├── colorization_release_v2.caffemodel
│   └── pts_in_hull.npy
│
├── static/
│   ├── uploads/
│   └── results/
│
├── templates/
│   └── index.html
│
└── requirements.txt
```

---

## ⚙️ Setup Instructions (Run Locally)

### 1. Clone the repository

```
git clone https://github.com/varchaswarastogi/IMAGE-COLORIZATION-TOOL.git
cd IMAGE-COLORIZATION-TOOL
```

---

### 2. Create virtual environment

```
python -m venv venv
```

Activate it:

* Windows:

```
venv\Scripts\activate
```

* Mac/Linux:

```
source venv/bin/activate
```

---

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

### 4. Download Model Files

Download the following files and place them inside the `models/` folder:

* `colorization_release_v2.caffemodel`
* `colorization_deploy_v2.prototxt`
* `pts_in_hull.npy`

👉 (FULL FILE LINK INCLUDING EVERYTHING AT ONCE : https://drive.google.com/drive/folders/1YWVBypHC8FnQiBPILwIHKm2SsP_iotoy?usp=drive_link )

⚠️ Make sure all files are correctly placed, otherwise the app will fail to start.

---

### 5. Run the application

```
python app.py
```

---

### 6. Open in browser

```
http://127.0.0.1:5000/
```

---

## Technologies Used

* Python
* Flask
* OpenCV (DNN module)
* NumPy
* AI AND ML ALGOS AND MODELS

---

##  Notes

* This project is intended for **local execution** due to model size and memory requirements.
* Deployment on free hosting platforms (like Render/Vercel) may fail due to:

  * RAM limitations
  * File size restrictions
  * Model loading constraints

---

##  Author: VARCHASWA RASTOGI

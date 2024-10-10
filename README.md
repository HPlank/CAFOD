# CAFOD: A Camera-Adaptive Foreign Object Detection Model for Coal Conveyor Belts

#### Furong Peng*, Kangjiang Hao, Guodong Li, Huiyuan Huang
The foreign objects on conveyor belt, such as ironbars, bigcoal, ironmesh and wood,  poses a threat of the damage to the conveyor belt that is a critical component of coal transportation in mines. However, foreign objects on the conveyor belt are similar to the coal in appearance, making foreign object detection model susceptible to noise. The view changing across surveillance cameras brings a new challenge for model's generalization and requiring model fin-tuning for each camera. To address these challenges, we propose a camera-adaptive foreign object detection model for coal conveyor belts. First, we apply multi-view data augmentation to simulate the view-changing and enable the model to learn more view-robust features. Next, we enhance the model's ability to distinguish foreign objects by introducing an enclosing feature perception module, which is able to capture the surrounding features of potential anomalies and enhance the context information. Additionally, we design a conveyor belt area loss function to enforce the model's focus on the belt area, mitigating interference from the complex background outside the belt.Finally, VariFocal Loss is introduced, enhancing the model's attention to difficult samples. In real coal mine environments, we collected a dataset of foreign objects on coal conveyor belts,  including lronbars,  bigcoal,  lronmesh,  and wood, from different cameras. Compared to the recent developed object detection methods,  the proposed model achieved state-of-the-art (SOTA) performance in foreign object detection, with a multi-camera adaptability score of 65.9%. Code are available at https://github.com/HPlank/CAFOD.

# Introduction
{T}{he} coal conveyor belt is a core equipment in the coal mining production,  and its stable operation is crucial for mining safety. However, due to the complexity of the mining environment, rocks, coal gangue,  and other debris often fall onto the conveyor belt, leading to coal pile blockages, breakages, and damage. Therefore, identifying foreign objects during coal production and transfer is particularly important 

Recently, deep learning-based object detection has achieved significant results in various fields. 
However, applying this technology to the field of foreign object detection on coal conveyor belts still faces numerous challenges. 
First, collecting foreign object data on coal conveyor belts in real production environments is exceptionally challenging.
Second, the camera views are vary across devices, making it difficult for the model to identify and locate the foreign object target across different cameras. 
Third, the complex background behind the conveyor belt may be falsely regarded as the foreign object as it is quite different from coal.
Finally, the images captured by the camera are susceptible to noise such as light, dust, and water vapor, which makes it difficult to distinguish between the foreground (foreign object) and background (coal).
To address these issues,  we propose a  Camera-Adaptive Foreign Object Detection CAFOD model for conveyor belt in coal mine, in view of the following considerations.
(1) Multi-View Data Augmentation (MVDA): Improving camera-adaptability for foreign object detection model through view augmentation in the limited data.
(2) Enclosing Feature Perception(EFP): Fusing the surrounding features of potential foreign objects to include more context information of the object, can improve the model's ability to reject false foreign objects in the complex background.
(3) Conveyor Belt Area Loss (CBAL): The conveyor belt area loss function is designed to enforce the model to focus on the conveyor belt area and reduce the interference of the complex background outside the conveyor belt.
(4) VariFocal Loss: By dynamically adjusting the weight of the focus loss, the model's attention to difficult samples is enhanced.
To evaluate the the model's camera-adaptability, we design a multi-camera validation experiment in which all model are trained in one camera's data and tested on other cameras' data. Experiments demonstrate that compared with the baseline model, the camera adaptability of our model has achieved a performance improvement of an average of 65.88%.

# Installation

### Requirements

We have trained and tested our models on `Ubuntu 16.0`, `CUDA 10.2`, `GCC 5.4`, `Python 3.7`

```bash
conda create -n owdetr python=3.7 pip
conda activate owdetr
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

# CAFOD: A Camera-Adaptive Foreign Object Detection Model for Coal Conveyor Belts

#### Furong Peng, Kangjiang Hao, Xuan Lu*
Foreign object detection on coal mine conveyor belts is crucial for ensuring operational safety and efficiency. How-
ever, applying deep learning to this task is challenging due to variations in camera perspectives, which alter the
appearance of foreign objects and their surrounding environment, thereby hindering model generalization. Despite
these viewpoint changes, certain core characteristics of foreign objects remain consistent. Specifically, (1) foreign
objects must be located on the conveyor belt, and (2) their surroundings are predominantly coal, rather than other ob-
jects. To leverage these stable features, we propose the Camera-Adaptive Foreign Object Detection (CAFOD) model,
designed to improve cross-camera generalization. CAFOD builds upon Deformable DETR, incorporating three main
strategies: (1) Multi-View Data Augmentation (MVDA) simulates viewpoint variations during training, enabling the
model to learn robust, view-invariant features; (2) Context Feature Perception (CFP) integrates local coal background
information to reduce false detections outside the conveyor belt; and (3) Conveyor Belt Area Loss (CBAL) enforces
explicit attention to the conveyor belt region, minimizing background interference. We evaluate CAFOD on a dataset
collected from real coal mines using three distinct cameras. Experimental results demonstrate that CAFOD outper-
forms state-of-the-art object detection methods, achieving superior accuracy and robustness across varying camera
perspectives. The code is available at https://github.com/HPlank/CAFOD.


# Introduction
Coal mining operations heavily rely on conveyor belt systems to transport coal from extraction sites to storage or processing facilities, ensuring efficiency and productivity in the industry. However, foreign objects such as rocks, coal gangue, tools, and broken equipment parts can accumulate on the conveyor belt due to various factors, including operational mishandling or mechanical failures. These foreign objects pose significant risks, potentially causing equipment damage, operational disruptions, and severe safety hazards. Therefore, accurate and efficient foreign object detection is critical to ensuring safe and smooth coal transportation.

Recent advances in deep learning-based object detection have shown remarkable success across diverse domains, including autonomous driving for pedestrian and vehicle recognition, medical image analysis for tumor detection, and security surveillance for anomaly detection. However, applying these techniques to foreign object detection on coal conveyor belts remains challenging due to the unique environmental conditions and variations in camera perspectives.

One primary challenge arises from the substantial variations in camera perspectives across different working environments. Changes in viewpoint can alter the appearance of both foreign objects and their backgrounds, making it difficult for a model trained on existing camera data to adapt effectively to new, unseen camera perspectives. Traditional deep learning models often experience significant performance degradation when applied to images from different angles. Fine-tuning the model on new camera data is a common approach, but collecting and annotating data across multiple coal mining sites is both labor-intensive and costly. Moreover, existing models are prone to false detection, particularly when background elements outside the conveyor belt are mistakenly classified as foreign objects. These challenges highlight the need for a more robust detection approach that effectively addresses viewpoint variations and improves adaptability across different cameras.

In response to these challenges, we propose the **Camera-Adaptive Foreign Object Detection (CAFOD)** model, specifically designed to enhance generalization across diverse camera perspectives. CAFOD is built on two key observations: (1) foreign objects must lie on the conveyor belt, and (2) their immediate surroundings are predominantly coal. Based on these insights, CAFOD integrates multiple targeted strategies to achieve robust and efficient performance. First, we introduce **Multi-View Data Augmentation (MVDA)** to simulate a wide range of camera viewpoints during training, enabling the model to learn view-invariant features. We then incorporate **Context Feature Perception (CFP)** to leverage local coal-background information, reducing the likelihood of false detections in non-coal regions. Additionally, we propose a **Conveyor Belt Area Loss (CBAL)**, which enforces explicit attention to the conveyor belt region, minimizing background interference. Finally, we integrate **Varifocal Loss (VFL)**, allowing the model to adaptively reweight challenging samples and improve the detection of small or visually similar foreign objects. These coordinated methods enable CAFOD to successfully handle viewpoint variability and achieve high detection accuracy in real coal conveyor belt environments.

Through these enhancements, CAFOD can improve model generalization across different camera perspectives, effectively addressing the limitations of conventional deep learning-based object detection methods. The key contributions of this work are summarized as follows:
- We introduce a novel approach to camera-adaptive foreign object detection that enhances generalization across varying camera perspectives without requiring extensive retraining.
- We incorporate Context Feature Perception (CFP) and Conveyor Belt Area Loss (CBAL) to enhance scene understanding. CFP helps the model focus on the coal background surrounding foreign objects, reducing false detections in non-coal areas, while CBAL explicitly guides attention to the conveyor belt region, minimizing background interference.
- Extensive experiments on real-world coal mine datasets demonstrate that CAFOD outperforms state-of-the-art detection models in terms of accuracy and cross-camera adaptability, highlighting its robustness across diverse viewpoints and challenging real-world conditions.
# Installation

### Requirements

We have trained and tested our models on `Ubuntu 16.0`, `CUDA 10.2`, `GCC 5.4`, `Python 3.7`

```bash
conda create -n CAFOD python=3.7 pip
conda activate CAFOD
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

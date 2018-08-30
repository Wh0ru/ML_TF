# Kaggles-facial-keypoints-detection（来自Kaggle论坛中Yumi的blog）
### 采用FCN的方法，因为在面部关键点检测的时候对空间信息的要求较高，FCN将如VGG模型的Fullconnect层都变为Conv2D层，在用上采样恢复为input的shape，
### 这样能保留较多的空间信息。采用guassian_kernel为关键点创建heatmap，作为label。
### train3.py会过拟合（改进中）

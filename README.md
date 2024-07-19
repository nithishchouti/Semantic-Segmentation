# Semantic-Segmentation

The project focuses on semantic segmentation, a vital task in computer vision that involves classifying each pixel in an image into a specific category. This capability is essential for applications like autonomous driving, medical imaging, and environmental monitoring, where understanding visual scenes at a granular level is crucial.

### Key Models and Techniques

1. **DeepLabv3+**: This state-of-the-art model utilizes atrous convolution to capture multi-scale contextual information and employs a sophisticated decoder for refining segmentation results. It is designed to work with various backbone architectures, including MobileNet, which enhances its efficiency and performance.

2. **MobileNet Backbones**: The project evaluates three variants of MobileNet:
   - **MobileNetV2**: Known for its inverted residuals and linear bottlenecks, it significantly reduces computational costs while maintaining performance.
   - **MobileNetV3 Large**: Incorporates advanced features like squeeze-and-excitation modules to improve efficiency.
   - **MobileNetV3 Small**: Optimized for mobile applications, offering a balance between performance and resource usage.

3. **SegFormer**: An alternative model that employs a hierarchical vision transformer backbone, providing efficient processing and excellent performance on segmentation tasks.

### Datasets Used

The project utilizes two prominent datasets for training and evaluation:
- **PASCAL VOC 2012**: A benchmark dataset widely used for semantic segmentation tasks.
- **ADE20K**: Another comprehensive dataset that provides a diverse range of scenes for evaluating segmentation performance.

### Objectives

The main objectives of the project include:
- Exploring the architectures of DeepLabv3+ and SegFormer for semantic segmentation.
- Evaluating the performance of different MobileNet backbones integrated with DeepLabv3+.
- Conducting a comparative analysis of model efficiency, focusing on metrics like accuracy, mean Intersection over Union (mIoU), inference time, GPU memory usage, and computational complexity.

### Conclusion

Through this project, the aim is to provide insights into the trade-offs between model complexity and performance, helping to inform decisions about model selection for deployment in resource-constrained environments. The findings will contribute to the broader goal of developing vision capabilities for bi-pedaled robots, enhancing their ability to interpret and navigate complex environments.

## New architectures for object detection based on DEtection TRansformers, Visual Transformers and class embeddings

### General information
The purpose of this project is to research the benefits and limitations of transformer-based object detection models, in comparison with classical CNN based approaches. Specifically, the recently proposed DEtection TRansformer (DETR) network is used to conduct experiments using various datasets. The goal is to design and implement architecture variations in an attempt to improve the base model and extract useful insights.

### Proposed architectures
Three novel DETR model variants are designed and implemented: MobileNet DETR, DETR with class embeddings, ViT DETR. MobileNet DETR involves modifying the backbone of the architecture with a significantly smaller model than the original ResNet50. The backbone is used as a feature map extractor. DETR with class embeddings consists of adding static class representations to the object queries of the transformer decoder. It is inspired from the UP-DETR paper, in which a self-supervised pretraining technique is developed. The final proposed solution, ViT DETR represents the union of two transformer-based computer vision models. ViT is a classification model that uses image patches to encode attention and derive class labels. The result is a transformer-only object detection architecture, from which the CNN backbone is completely eliminated.

![](https://github.com/carlarusu/ThesisProject/blob/master/images/sys_design.png?raw=true)

### Results
MobileNet DETR shows that constructing a capable compact transformer detector is possible, at the cost of only a few percentage points. This model shortens the training schedule by 45%. 

Class embeddings were determined to be incompatible with the model, as they slightly harmed the mAP.

ViT DETR attained promising results, still short of the base model in mAP, but with a much faster convergence rate. The model needs half the amount of iterations of the base model to converge. This points to the conjecture that a transformer-only encoding is a better fit for the decoder. The lower mAP can be explained by the fixed small size input requirements of ViT and can be mitigated by a change in design to allow high-resolution images while maintaining ratio.

![](https://github.com/carlarusu/ThesisProject/blob/master/images/plot.png?raw=true)
![](https://github.com/carlarusu/ThesisProject/blob/master/images/table.png?raw=true)

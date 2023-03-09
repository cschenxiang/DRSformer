# Learning A Sparse Transformer Network for Effective Image Deraining (CVPR 2023)

[Xiang Chen](https://cschenxiang.github.io/), [Hao Li](https://house-leo.github.io/), Mingqiang Li, and [Jinshan Pan](https://jspan.github.io/)

## Updates
- [2023-03-02] The Arxiv version of this paper will be available at March 20.
- [2023-02-28] This paper will appear in CVPR 2023.  

<hr />

> **Abstract:** *Transformers have achieved significant decent performance in image deraining as they can better model the non-local information which is vital for high-quality image reconstruction. In this paper, we find that most existing Transformers usually use all similarities of the tokens from the query-key pairs for the feature aggregation. However, if the tokens from the query are different from those of the key, the self-attention values estimated from these tokens also involve in feature aggregation, which accordingly interferes with the clear image restoration. To overcome this problem, we propose an effective deraining network, sparse Transformer (DRSformer) that can adaptively keep the most useful self-attention values for feature aggregation so that the aggregated features better facilitate high-quality image reconstruction. Specifically, we develop a simple yet learnable top-k selection operator to adaptively retain the most crucial attention scores from the keys for each query for better feature aggregation. Simultaneously, as the naive feed-forward network in Transformers does not model the multi-scale information that is important for latent clear image restoration, we develop an effective mixed-scale feed-forward network to generate better features for image deraining. To learn an enriched set of hybrid features that combines local context from CNN operators, we also equip our model with mixture of experts feature compensator to present a cooperation refinement deraining scheme. Extensive experimental results on the commonly used benchmarks demonstrate that the proposed method achieves favorable performance against state-of-the-art approaches. The source codes are available at https://github.com/cschenxiang/DRSformer.*
<hr />

## Network Architecture

<img src = "figs/network.png">

## Datasets
<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Rain200L</th>
    <th>Rain200H</th>
    <th>DID-Data</th>
    <th>DDN-Data</th>
    <th>SPA-Data</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Baidu Cloud</td>
    <td> <a href="https://pan.baidu.com/s/1rTb4qU3fCEA4MRpQss__DA?pwd=s2yx">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1KK8R2bPKgcOX8gMXSuKtCQ?pwd=z9br">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1aPFJExxxTBOzJjngMAOQDA?pwd=5luo">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1g_m7RfSUJUtknlWugO1nrw?pwd=ldzo">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1YfxC5OvgYcQCffEttFz8Kg?pwd=yjow">Download</a> </td>
  </tr>
</tbody>
</table>
Here, these datasets we provided are fully paired images, especially SPA-Data.

## Performance Evaluation
See folder "evaluations" 

1) *for Rain200L/H and SPA-Data datasets*: 
PSNR and SSIM results are computed by using this [Matlab Code](https://github.com/swz30/Restormer/blob/main/Deraining/evaluate_PSNR_SSIM.m).

2) *for DID-Data and DDN-Data datasets*: 
PSNR and SSIM results are computed by using this [Matlab Code](https://github.com/hongwang01/RCDNet/tree/master/Performance_evaluation).

## Visual Results
<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Rain200L</th>
    <th>Rain200H</th>
    <th>DID-Data</th>
    <th>DDN-Data</th>
    <th>SPA-Data</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>DualGCN</td>
    <td> Not available </td>
    <td> Not available </td>
    <td> Not available </td>
    <td> Not available </td>
    <td> Not available </td>
  </tr>
  <tr>
    <td>SPDNet</td>
    <td> Not available </td>
    <td> Not available </td>
    <td> Not available </td>
    <td> Not available </td>
    <td> Not available </td>
  </tr>
  <tr>
    <td>Uformer-B</td>
    <td> Not available </td>
    <td> Not available </td>
    <td> Not available </td>
    <td> Not available </td>
    <td> Not available </td>
  </tr>
</tbody>
</table>


## Citation
If you are interested in this work, please consider citing:

    @inproceedings{DRSformer,
        title={Learning A Sparse Transformer Network for Effective Image Deraining}, 
        author={Chen, Xiang and and Li, Hao and Li, Mingqiang and Pan, Jinshan},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2023}
    }

## Acknowledgment
This code is based on the [Restormer](https://github.com/swz30/Restormer). Thanks for their awesome work.

## Contact
Should you have any question or suggestion, please contact chenxiang@njust.edu.cn.

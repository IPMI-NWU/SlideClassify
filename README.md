
# Hard example mining in Multi-Instance Learning for Whole-Slide Image Classification
## 切图及提特征

切图有两种方式，使用[CLAM](https://github.com/mahmoodlab/CLAM)的切图及特征提取代码，或者使用SlideFilter的切图代码。在20x下切256大小的patch，然后使用$extract_feature_form_patchesCut.py$， 设置路径为patch结果目录。

## 训练分类模型

项目代码设置了统一的文件路径，只需要设置对应的数据类型的名字，会匹配相应的数据目录。

1. 训练初始轮权重

```
python M2_update_MIL_classifier.py --round 0 --datasetsName [数据的名字]
```

2. 提取伪标签

```
python PseudoLabeling_hardmining.py --round 1 --datasetsName [数据的名字]
```

3. 训练特征提取网络

```
python M1_update_feat_encoder.py --round 1  --datasetsName [数据的名字]
```

4. 从第一步开始重复前三步，round需要对应的增加

```
python M2_update_MIL_classifier.py --round 1 --datasetsName [数据的名字]
python PseudoLabeling_hardmining.py --round 2 --datasetsName [数据的名字]
python M1_update_feat_encoder.py --round 2  --datasetsName [数据的名字]
python M2_update_MIL_classifier.py --round 2 --datasetsName [数据的名字]
......
```



## 测试结果

设置参数为使用的

```
python M2_update_MIL_classifier.py --is_test --datasetsName [数据的名字]
```


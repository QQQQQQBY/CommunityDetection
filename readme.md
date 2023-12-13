### 社区检测方法集成

#### 集成方法包含：非重叠社区检测方法与重叠社区检测方法
非重叠社区算法：louvain, deepwalk, girvan_newman, lfm, clusternet, infomap, walktrap, multilevel, fast, spectral_cluster

重叠社区检测算法：slpa, scan, lpa

#### 关于复现
```
pip install -r requirements.txt
python community_detection_all_methods.py
```
部分方法会运行时间超时，在代码中做了注释


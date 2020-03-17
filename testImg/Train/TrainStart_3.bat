opencv_traincascade -data output_hook -vec ./pos_hook/pos.vec -bg ./neg/neg.txt -numPos 6 -numNeg 6 -numStages 9 -w 50 -h 50 -minHitRate 0.998 -maxFalseAlarmRate 0.1 -featureType HAAR -precalcValBufSize 4096 -precalcIdxBufSize 4096
pause
::# -data：指定保存训练结果的文件夹；
::# -vec:指定正样本集；
::# -bg:指定负样本的描述文件夹；
::# -numPos：指定每一级参与训练的正样本的数目（要小于正样本总数）；
::# -numNeg:指定每一级参与训练的负样本的数目（可以大于负样本图片的总数）；
::# -numStage:训练的级数；
::# -w:正样本的宽；
::# -h:正样本的高；
::# -minHitRate:每一级需要达到的命中率（一般取值0.95-0.995）；
::# -maxFalseAlarmRate:每一级所允许的最大误检率；
::# -mode:使用Haar-like特征时使用，可选BASIC、CORE或者ALL；
::# 另外，还可指定以下字段：
::# -featureType:可选HAAR或LBP，默认为HAAR;

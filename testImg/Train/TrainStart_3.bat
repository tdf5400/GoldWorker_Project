opencv_traincascade -data output_hook -vec ./pos_hook/pos.vec -bg ./neg/neg.txt -numPos 6 -numNeg 6 -numStages 9 -w 50 -h 50 -minHitRate 0.998 -maxFalseAlarmRate 0.1 -featureType HAAR -precalcValBufSize 4096 -precalcIdxBufSize 4096
pause
::# -data��ָ������ѵ��������ļ��У�
::# -vec:ָ������������
::# -bg:ָ���������������ļ��У�
::# -numPos��ָ��ÿһ������ѵ��������������Ŀ��ҪС����������������
::# -numNeg:ָ��ÿһ������ѵ���ĸ���������Ŀ�����Դ��ڸ�����ͼƬ����������
::# -numStage:ѵ���ļ�����
::# -w:�������Ŀ�
::# -h:�������ĸߣ�
::# -minHitRate:ÿһ����Ҫ�ﵽ�������ʣ�һ��ȡֵ0.95-0.995����
::# -maxFalseAlarmRate:ÿһ����������������ʣ�
::# -mode:ʹ��Haar-like����ʱʹ�ã���ѡBASIC��CORE����ALL��
::# ���⣬����ָ�������ֶΣ�
::# -featureType:��ѡHAAR��LBP��Ĭ��ΪHAAR;

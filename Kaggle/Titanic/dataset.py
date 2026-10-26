import os
import kagglehub

path = kagglehub.competition_download('titanic')
# 将 kagglehub 的缓存根目录指向你的目标文件夹
os.environ['KAGGLEHUB_CACHE'] = './Kaggle/source/Titanic'

path = kagglehub.competition_download('titanic')
print("数据已下载至:", path)
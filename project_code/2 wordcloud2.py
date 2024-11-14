# 1. word cloud

import matplotlib.pyplot as plt
import jieba
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
import PIL.Image as image
import numpy as np

# 1.2 Reports

with open('report2.txt', encoding='utf-8') as fp:
    text = fp.read()

wl_space_split = list(jieba.cut(text))

sw = set(STOPWORDS)
sw.add('的')
sw.add('和')
sw.add('是')
sw.add('与')
sw.add('之间')
sw.add('以及')
sw.add('在')
sw.add('了')
sw.add('可以')
sw.add('称为')
sw.add('问题')
sw.add('将')
sw.add('进行')
sw.add('一个')
sw.add('上')
sw.add('举例')
sw.add('中')

mask = np.array(image.open("cur1.png"))

image_colors = ImageColorGenerator(mask)
cloud_ = WordCloud(background_color=None, mode="RGBA", width=1000, height=1000,
                   color_func=image_colors, mask=mask, font_path='font1.ttf',
                   stopwords=sw, relative_scaling=1.0, collocations=False,
                   min_font_size=10, max_font_size=1000)
cloud_.generate(" ".join(wl_space_split))

plt.figure(figsize=(12, 12))
plt.xticks([])
plt.yticks([])
plt.imshow(cloud_)
plt.show()
cloud_.to_file('word_cloud_report.png')

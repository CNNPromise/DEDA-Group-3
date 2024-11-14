# 1. word cloud

import matplotlib.pyplot as plt
import jieba
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
import PIL.Image as image
import numpy as np

# 1.1 Poems

with open('jiu2.txt', encoding='utf-8') as fp:
    text = fp.read()

wl_space_split = list(jieba.cut(text))

sw = set(STOPWORDS)
sw.add('首')
sw.add('二首')
sw.add('三首')
sw.add('四首')
sw.add('之一')
sw.add('之二')
sw.add('的')
sw.add('上')
sw.add('下')
sw.add('去')
sw.add('来')
sw.add('是')
sw.add('与')
sw.add('之')
sw.add('有')
sw.add('无')
sw.add('在')
sw.add('不')
sw.add('将')
sw.add('将')
sw.add('为')

mask = np.array(image.open("cur9.jpg"))

image_colors = ImageColorGenerator(mask)
cloud_ = WordCloud(background_color=None, mode="RGBA", width=1000, height=1000,
                   color_func=image_colors, mask=mask, font_path='font1.ttf',
                   stopwords=sw, relative_scaling=1.0, collocations=False,
                   min_font_size=10, max_font_size=1000)
cloud_.generate(" ".join(wl_space_split))

plt.figure(figsize=(10, 10))
plt.xticks([])
plt.yticks([])
plt.imshow(cloud_)
plt.show()

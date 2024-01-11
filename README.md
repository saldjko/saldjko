import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import matplotlib.pyplot as plt
#输入数据
miRNA_sequences = np.random.rand(100, 50)
gene_sequences = np.random.rand(100, 100)
labels = np.random.randint(2, size=(100, 1))
#构建深度学习模型
embedding_dim = 16
model = Sequential([
    Embedding(input_dim=21, output_dim=embedding_dim, input_length=50, name='miRNA_embedding'),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#调整输入数据键
input_data = {'miRNA_embedding_input': miRNA_sequences, 'gene_input': gene_sequences}
#训练模型并积累准确值
history = model.fit(input_data, labels, epochs=10, batch_size=32, verbose=1)
#可视化训练精度
#绘制图像
plt.plot(range(1, 11), history.history['accuracy'], label='Accuracy')
plt.title('Training Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

import tensorflow as tf
import tensorflow_datasets as tfds
ds = tfds.load('DeepWeeds', split='train', shuffle_files=True)
ds_size = 17509
train_split = 0.8
test_split = 0.2
shuffle_size = 10000
assert (train_split + test_split) == 1

ds = ds.shuffle(shuffle_size, seed=12)

train_size = int(train_split * ds_size)

train_ds = ds.take(train_size)
test_ds = ds.skip(train_size)

cnn = tf.keras.models.Sequential()

#resize images


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[256,256,3]))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

cnn.fit(x = train_ds, validation_data = test_ds, epochs = 25)

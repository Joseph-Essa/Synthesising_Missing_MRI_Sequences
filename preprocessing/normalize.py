import tensorflow as tf


def normalize(image):
    image = (image/127.5)-1
    return image

def preprocess_dataset(A_train, A_test, B_train, B_test):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    A_train = A_train.map(lambda x: (normalize(x))).cache().repeat().prefetch(buffer_size=AUTOTUNE)
    A_test = A_test.map(lambda x: (normalize(x))).cache().repeat().prefetch(buffer_size=AUTOTUNE)

    B_train = B_train.map(lambda x: (normalize(x))).cache().repeat().prefetch(buffer_size=AUTOTUNE)
    B_test = B_test.map(lambda x: (normalize(x))).cache().repeat().prefetch(buffer_size=AUTOTUNE)

    return A_train, A_test, B_train, B_test

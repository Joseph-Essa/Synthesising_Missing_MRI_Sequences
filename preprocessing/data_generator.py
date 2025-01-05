import tensorflow as tf
import random


def create_dataset(data_dir_A, data_dir_B, BATCH_SIZE, img_height, img_width):
    
    seed = random.randint(0, 10000)
    print("Random seed:", seed)
    A_train = tf.keras.preprocessing.image_dataset_from_directory(
                              data_dir_A,
                              seed=seed,
                              validation_split = 0.06,
                              subset = 'training',
                              labels=None,
                              image_size = (img_height, img_width),
                              batch_size=BATCH_SIZE)
    
    A_test = tf.keras.preprocessing.image_dataset_from_directory(
                                data_dir_A,
                                seed=seed,
                                validation_split = 0.06,
                                subset = 'validation',
                                labels=None,
                                image_size=(img_height, img_width),
                                batch_size=1)
    
    B_train = tf.keras.preprocessing.image_dataset_from_directory(
                                data_dir_B,
                                seed=seed,
                                validation_split = 0.06,
                                subset = 'training',
                                labels=None,
                                image_size=(img_height, img_width),
                                batch_size=BATCH_SIZE)
    
    B_test = tf.keras.preprocessing.image_dataset_from_directory(
                                data_dir_B,
                                seed=seed,
                                validation_split = 0.06,
                                subset = 'validation',
                                labels=None,
                                image_size=(img_height, img_width),
                                batch_size=1)
    
    return A_train, A_test, B_train, B_test
from generator import squeeze_attention_unet as generator_g
from discriminator import discriminator as discriminator_x
from normalize import preprocess_dataset 
from data_generator import create_dataset
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tkinter import Tcl
import time
import ast
import re

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

print("Beginning of the program")

A = 't2'
B = 'flair'
BEGINNING = 1 # The starting epoch number

path = "/lfs01/workdirs/hlwn041u6/use-hpc/jupyter/Model_Output"
os.makedirs(path, exist_ok=True)
Path(os.path.join(path, f"{A} TO {B}")).mkdir(parents=True, exist_ok=True)
model_path = os.path.join(path, f"{A} TO {B}")
Path(os.path.join(model_path, f"ckp")).mkdir(parents=True, exist_ok=True)
checkpoint_path = os.path.join(model_path, "ckp")
Path(os.path.join(model_path, f"loss")).mkdir(parents=True, exist_ok=True)
loss_path = os.path.join(model_path, "loss")
Path(os.path.join(model_path, f"results")).mkdir(parents=True, exist_ok=True)
results_path = os.path.join(model_path, "results")

# Losses
LAMBDA = 100.0 

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss
    
def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    l2_loss = 1 - tf.reduce_mean(tf.image.ssim(target, gen_output, max_val = 2.0))
    l3_loss = (l1_loss + l2_loss) / 2
    total_gen_loss = gan_loss + (LAMBDA * l3_loss)
    return total_gen_loss, gan_loss, l3_loss

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5 ) # , beta_1=0.5
discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5 ) # , beta_1=0.5

# Show Sample Every Epoch
def generate_images(model1, test1,test2,  gen_g_loss, disc_x_loss, epoch):
    prediction1 = model1(test1)
#     prediction2 = model2(test2)
    
    test1 = np.rot90(test1[0, :, :, 0], 3)
    test2 = np.rot90(test2[0, :, :, 0], 3)
    prediction1 = np.rot90(prediction1[0, :, :, 0], 3)
#     prediction2 = np.rot90(prediction2[0, :, :, 0], 3)
    
    plt.figure(figsize=(10, 10))
    display_list = [test1, prediction1, test2]
    
    title = [f'{A} True', f'{B} predicted', f'{B} True']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5, cmap='gray')
        plt.axis('off')
    plt.text(-600, 300 ,"gen_g_loss = {:.3f}, disc_x_loss = {:.3f}".format(gen_g_loss, disc_x_loss))
    save_path = os.path.join(model_path, f"results/image_at_epoch_{epoch:04d}.png".format(epoch))
    plt.savefig(save_path)
    # plt.show()

# Training Step
# Ref: https://www.tensorflow.org/guide/function
@tf.function
def train_step(real_x, real_y):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        fake_y = generator_g(real_x, training=True)
        
        disc_real_x = discriminator_x(real_y, training=True)
#         disc_real_y = discriminator_y(real_y, training=True)
        
        disc_fake_x = discriminator_x(fake_y, training=True)
#         disc_fake_y = discriminator_y(fake_y, training=True)
        #print(f"Discriminator Real X: {disc_fake_x}")

        # calculate the loss
        total_gen_g_loss = generator_loss(disc_fake_x,fake_y,real_y)
#         gen_f_loss = generator_loss(disc_fake_x)
        # total_gen_g_loss = generator_loss(disc_fake_x)
        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        
    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
    
    discriminator_x_gradients = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
    
    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables))
    
    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, discriminator_x.trainable_variables))
    
    return total_gen_g_loss, disc_x_loss

# Test Step
@tf.function
def test_step(real_x, real_y):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        fake_y = generator_g(real_x, training=True)
        
        disc_real_x = discriminator_x(real_y, training=True)
        
        disc_fake_x = discriminator_x(fake_y, training=True)

        # calculate the loss
        total_gen_g_loss = generator_loss(disc_fake_x,fake_y,real_y)
        
        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    
    return total_gen_g_loss, disc_x_loss


# history of generator loss (training)
gen_loss = []
# history of discrimnitor loss (training)
disc_loss = []
# history of generator loss (validation)
gen_test_loss = []
# history of discrimnitor loss (validation)
disc_test_loss = []

if os.path.exists(os.path.join(loss_path, 'training_history.txt')) and os.path.getsize(os.path.join(loss_path, 'training_history.txt')) > 0:
    with open(os.path.join(loss_path, 'training_history.txt'), 'rb') as file:
        data_str = file.read()
        decoded_string = data_str.decode("UTF-8")
        
        cleaned_string = re.sub(r'<tf\.Tensor: shape=\(\), dtype=float32, numpy=[\d\.]+>', '0', decoded_string)
        data = ast.literal_eval(decoded_string)
else:
    data = {'gen_loss': [], 'disc_loss': [], 'gen_test_loss': [], 'disc_test_loss': []}

def training_loop(A_train, A_test, B_train, B_test, BATCH_SIZE, EPOCHS, ckpt_manager, sample_A, sample_B):
    end = BEGINNING + EPOCHS
    for epoch in range(BEGINNING, end):
        iteration_count = 0
        start_time = time.time()
        gen_g_loss = 0
        disc_x_loss = 0

        #print(gen_g_loss_temp[0])
        iteration_count = 0
        for image_x,image_y in tf.data.Dataset.zip((A_train,B_train)):
            gen_g_loss_temp, disc_x_temp = train_step(image_x,image_y)
            
            if iteration_count % 100 == 0:
                print(f'Epoch: {epoch}, Iteration: {iteration_count}, Gen Loss: {gen_g_loss_temp[0]}, Disc Loss: {disc_x_temp}, Time taken: {(time.time() - start_time)//60} Min\n')
                
            iteration_count += 1
    
            gen_g_loss = gen_g_loss + gen_g_loss_temp[0]  / 14025 * BATCH_SIZE
            disc_x_loss = disc_x_loss + disc_x_temp / 14025 * BATCH_SIZE

        print(f'End of epoch {epoch}, Gen Loss: {gen_g_loss}, Disc Loss: {disc_x_loss}')
        print(f'Time taken for epoch {epoch} is {(time.time() - start_time)//60} Min\n')

        tot_loss_gen = gen_g_loss 
        tot_loss_dis = disc_x_loss 

        gen_loss.append(tot_loss_gen)
        disc_loss.append(tot_loss_dis)

        generate_images(generator_g, sample_A, sample_B, tot_loss_gen, disc_x_loss, epoch)

        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch', epoch, 'at', ckpt_save_path)
        gen_g_loss = 0
        gen_f_loss = 0
        disc_x_loss = 0
        disc_y = 0
        for image_x ,image_y in tf.data.Dataset.zip((A_test , B_test)):
            gen_g_loss_temp, disc_x_temp = test_step(image_x,image_y)
            print(gen_g_loss_temp[0])
            gen_g_loss = gen_g_loss + gen_g_loss_temp[0]  / 895 * BATCH_SIZE
            disc_x_loss = disc_x_loss + disc_x_temp / 895  * BATCH_SIZE
        tot_test_loss_gen = gen_g_loss 
        tot_test_loss_dis = disc_x_loss 

        gen_test_loss.append(tot_loss_gen)
        disc_test_loss.append(tot_loss_dis)

        data['gen_loss'].append(float(tot_loss_gen))
        data['disc_loss'].append(float(tot_loss_dis))
        data['gen_test_loss'].append(tot_test_loss_gen)
        data['disc_test_loss'].append(tot_test_loss_dis)
        # gen_loss.append(data['gen_loss'])

        # Save the dictionary to a text file
        with open(os.path.join(loss_path, 'training_history.txt'), 'w') as file:
            file.write(str(data) + '\n')

        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time//60, "Min")

def main():
    # Initialize constants    
    BUFFER_SIZE = 14920
    BATCH_SIZE = 6
    EPOCHS = 200
    img_height = 256
    img_width = 256

    # Path to the dataset
    data_dir_A = Path(f"/lfs01/workdirs/hlwn041u6/dataset_png/{A}_png")
    data_dir_B = Path(f"/lfs01/workdirs/hlwn041u6/dataset_png/{B}_png")

    print(tf.config.list_physical_devices('GPU'))

    print(f"{A} MRI images: ",len(list(data_dir_A.glob('*B*.png'))))
    print(f"{B} MRI images: ",len(list(data_dir_B.glob('*B*.png'))))

    A_train, A_test, B_train, B_test = create_dataset(data_dir_A, data_dir_B, BATCH_SIZE, img_height, img_width)    

    A_train, A_test, B_train, B_test = preprocess_dataset(A_train, A_test, B_train, B_test)

    # Load the model
    ckpt = tf.train.Checkpoint(generator_g=generator_g,
                            discriminator_x=discriminator_x,
                            generator_g_optimizer=generator_g_optimizer,
                            discriminator_x_optimizer=discriminator_x_optimizer,
                            )

    # Ref: https://www.tensorflow.org/api_docs/python/tf/train/CheckpointManager
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=300)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)

        print(f'Last Check Point: {ckpt_manager.latest_checkpoint}')
        print('Latest checkpoint restored!!')

    sample_A = next(iter(A_train))
    sample_B = next(iter(B_train))

    print("Starting the training loop...")
    training_loop(A_train, A_test, B_train, B_test, BATCH_SIZE, EPOCHS, ckpt_manager, sample_A, sample_B)


if __name__ == "__main__":
    try:
        print("Starting the main function...")
        main()
    except Exception as e:
        print(f"Error occurred: {e}")

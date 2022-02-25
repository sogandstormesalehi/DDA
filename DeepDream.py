#!/usr/bin/env python
# coding: utf-8

# In[32]:


#import packages
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import IPython.display as display
import PIL.Image
import time


# In[33]:


#download image and fit it
url = 'file:///C:/Users/Asus/Dropbox/My%20PC%20(DESKTOP-20TNHRH)/Desktop/b89b53197cc2fda841a56e668e1a7d8d.jpg'
def download(url, max_dim = None):
    name = url.split('/')[-1]
    image_path = tf.keras.utils.get_file(name, origin = url)
    img = PIL.Image.open(image_path)
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img)    


# In[34]:


#normalize the image
def deprocess(img):
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)


# In[35]:


#display
def show(img):
    display.display(PIL.Image.fromarray(np.array(img)))


# In[36]:


#resize
original_img = download(url, max_dim = 500)
show(original_img)


# In[37]:


base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')


# In[38]:


chosen_mixed = ['mixed3', 'mixed5']
layers = [base_model.get_layer(name).output for mixed in chosen_mixed]
dream_model = tf.keras.Model(inputs = base_model.input, outputs = layers)


# In[39]:


#maximize loss to make everything weirder
def calculate_loss(img, model):
    img_batch = tf.expand_dims(img, axis = 0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]
    losses = []
    for activation in layer_activations:
        loss = tf.math.reduce_mean(activation)
        losses.append(loss)
    return tf.reduce_sum(losses)    


# In[40]:


#add gradient
class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model
    @tf.function(
        input_signature = (
        tf.TensorSpec(shape = [None, None, 3], dtype = tf.float32),
        tf.TensorSpec(shape = [], dtype = tf.int32),
        tf.TensorSpec(shape=[], dtype = tf.float32),
        )
    )
    
    def __call__(self, img, steps, step_size):
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(img)
                loss = calculate_loss(img, self.model)
            gradients = tape.gradient(loss, img)
            gradients /= tf.math.reduce_std(gradients) + 1e-8
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)
            
        return loss, img    


# In[41]:


deep_dream = DeepDream(dream_model)


# In[42]:


def run_deep_dream_simple(img, steps = 100, step_size = 0.01):
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)
    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining > 100:
            run_steps = tf.constant(100)
        else:
            run_steps = tf.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps
        loss, img = deep_dream(img, run_steps, tf.constant(step_size))
        display.clear_output(wait = True)
        show(deprocess(img))
        
    result = deprocess(img)
    display.clear_output(wait = True)
    show(result)
    
    return result


# In[43]:


dream_img = run_deep_dream_simple(img = original_img, steps = 100, step_size = 0.01)


# In[44]:


#go batshit crazy
start = time.time()
octave_scale = 1.30
img = tf.constant(np.array(original_img))
base_shape = tf.shape(img)[:-1]
float_base_shape = tf.cast(base_shape, tf.float32)
for n in range(-2, 3):
    new_shape = tf.cast(float_base_shape * (octave_scale ** n), tf.int32)
    img = tf.image.resize(img, new_shape).numpy()
    img = run_deep_dream_simple(img = img, steps = 50, step_size = 0.01)
display.clear_output(wait = True)
show(img)
end = time.time()
end-start


# In[45]:



def random_roll(img, maxroll):
    shift = tf.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32)
    img_rolled = tf.roll(img, shift=shift, axis=[0,1])
    return shift, img_rolled


# In[46]:


shift, img_rolled = random_roll(np.array(original_img), 512)
show(img_rolled)


# In[47]:


class TiledGradients(tf.Module):
    def __init__(self, model):
        self.model = model
    
    @tf.function(
    input_signature = (
    tf.TensorSpec(shape = [None, None, 3], dtype = tf.float32),
    tf.TensorSpec(shape=[2], dtype = tf.int32),
    tf.TensorSpec(shape=[], dtype = tf.int32),    
    )
    )
    
    def __call__(self, img, img_size, tile_size = 512):
        shift, img_rolled = random_roll(img, tile_size)
        gradients = tf.zeros_like(img_rolled)
        xs = tf.range(0, img_size[1], tile_size)[:-1]
        if not tf.cast(len(xs), bool):
            xs = tf.constant([0])
        ys = tf.range(0, img_size[0], tile_size)[:-1]
        if not tf.cast(len(ys), bool):
            ys = tf.constant([0])
        
        for x in xs:
            for y in ys:
                with tf.GradientTape() as gradient_tape:
                    gradient_tape.watch(img_rolled)
                    img_tile = img_rolled[y: y + tile_size, x: x + tile_size]
                    loss = calculate_loss(img_tile, self.model)
                    
                gradients = gradients + gradient_tape.gradient(loss, img_rolled)
                
        gradients = tf.roll(gradients, shift = -shift, axis = [0, 1])
        gradients /= tf.math.reduce_std(gradients) + 1e-8
        
        return gradients


# In[48]:


get_tiled_gradients = TiledGradients(dream_model)


# In[49]:


def run_deep_dream_with_octaves(img, steps_per_octave = 100, step_size = 0.01, octaves = range(-2, 3), octave_scale = 1.3):
    base_shape = tf.shape(img)
    img = tf.keras.utils.img_to_array(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    
    initial_shape = img.shape[:-1]
    img = tf.image.resize(img, initial_shape)
    for octave in octaves:
        new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32) * (octave_scale ** octave)
        new_size = tf.cast(new_size, tf.int32)
        img = tf.image.resize(img, new_size)
        
        for step in range(steps_per_octave):
            gradients = get_tiled_gradients(img, new_size)
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)
            
            if step % 10 == 0:
                display.clear_output(wait = True)
                show(deprocess(img))
                print("Octave {}, Step {}".format(octave, step))
                
        result = deprocess(img)
        return result


# In[52]:


img = run_deep_dream_with_octaves(img = original_img, step_size = 0.01)
display.clear_output(wait = True)
show(img)


# In[ ]:





# In[ ]:





<span align="left">
  <img src="https://cdn-icons-png.flaticon.com/512/6461/6461928.png" width=40 height=40 align="left">
  <h1 align="left">PokédexAI Neural Network</h1>
</span>
<img src="https://github.com/user-attachments/assets/5d7e2909-d63b-4a07-b791-913024aeb901" width=700/ >

This Python file contains code extracted from the Jupyter Notebook where I trained the ResNet-9 convolutional neural network for the PokédexAI project. It showcases the various techniques and optimizations I implemented to enhance the model's performance. For a comprehensive understanding of the training process and the methodologies used, please explore the complete repository.

If you're interested in the PokédexAI website and the user interface I developed around this neural network, I invite you to read my detailed blog post. It provides an in-depth look at the design and features that make PokédexAI an engaging and powerful tool for Pokémon enthusiasts.

## Model Loss and Accuracy Over Time

<img src="https://github.com/user-attachments/assets/f6437b6d-4bb1-4af3-8c68-091737e30ee1" width=350 height=300 />
<img src="https://github.com/user-attachments/assets/71975d5b-4a79-4b50-8093-b08efc76be8d" width=350 height=300 />


## Training Techniques Used

<div>
  <p>
    <h4>Fit-One-Cycle Learning Rate</h4> 
    The Fit-One-Cycle learning rate policy adjusts the learning rate dynamically during training. It statrs slow, gradually increases to the maximum (I used 0.01), and slowly decreases again.
    This allows the model to converge faster and can lead to better perfomance by avoiding local minima and saddle points. My model used Fit-One-Cycle for 50 epochs.
  </p>
  <p>
    <h4>Dropout Layers</h4> 
     Dropout is a regularization technique that helps prevent overfitting by randomly setting to zero (i.e. "dropping out") a fraction of the neurons during training. This forces the network to learn more robust
     features that are not reliant on specific neurons. Dropout layers improve generaliation and reduce the risk of overfitting to training data so that it performs better on unseen images.
  </p>
  <p>
    <h4>Data Augmentation</h4>
    Data Augmentation involves creating additional training examples by applying various transformations such as rotations, translations and flips to existing image. This increases the diversity of the training data
    without needing to collect more samples. By expanding the training dataset, it makes the model more robbust to variations in the input images, enhancing its ability to generalize, and increasing its classificaiton accuracy
  </p>
  <p>
    <h4>Image Normalization</h4>
    Normalization standardizes the input images by scaling pixel values to a consistent range of [0,1] or [-1,1]. This involves calculating the mean and standard deviation of pixel values to ensure the data has zero mean
    and unit variance. 
  </p>
</div>

## Dataset Class Distribution


<img src="https://github.com/user-attachments/assets/33e88cbc-8d3b-44aa-8db7-308b2fd0904d" width=900/>


## Sample Augmented & Normalized Image Data

<img src="https://github.com/user-attachments/assets/7654674e-05b5-440f-99dd-c4a4ac987b57" width=600 />

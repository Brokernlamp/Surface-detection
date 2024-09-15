# Surface-detection
This is the code for model training which can be used to predict the given video sample/image contains 'Even, uneven, slippery' surface. A classifier is obtained which can further be used to deploy the trained model.
Step-by-Step Guide for Direct Dataset Upload in Google Colab
Upload the Dataset Folder Directly:

In the Colab interface, you can use the "Files" tab on the left side.
Click on the Upload button (the file icon).
Select your zipped dataset folder (which contains the Even, Uneven, Slippery folders).
Once uploaded, use the following code to unzip and extract it:
bash
Copy code
!unzip /content/your_dataset.zip -d /content/dataset/
This will unzip your dataset to the /content/dataset/ folder.

Accessing the Dataset in Colab: Once uploaded and extracted, you can directly access the dataset for training as follows:

python
Copy code
dataset_path = '/content/dataset/'  # Make sure the folder has subdirectories: Even, Uneven, Slippery
Training the Model Using the Uploaded Dataset: Now that your dataset is uploaded and extracted, you can proceed with the training code I shared earlier. Just make sure the dataset_path points to the correct directory.

# **HW4 Handout**

In this assignment, we are introducing a new format for assignment delivery, designed to enhance your development workflow. The key motivations for this change are:

- **Test Suite Integration**: Your code will be tested in a manner similar to `HWP1's`.
- **Local Development**: You will be able to perform most of your initial development locally, reducing the need for compute resources.
- **Hands-on Experience**: This assignment provides an opportunity to build an end-to-end deep learning pipeline from scratch. We will be substantially reducing the amount of abstractions compared to previous assignments.

For our provided notebook's to work, your notebook's current working directory must be the same as the handout.
This is important because the relative imports in the notebook's depend on the current working directory.
This can be achieved by:

1. Physically moving the notebook's into the handout directory.
2. Changing the notebook's current working directory to the handout directory using the `os.chdir()` function.

Your current working directory should have the following files for this assignment:

```
.
├── README.md
├── hw4lib/
├── mytorch/
├── tests/
├── hw4_data_subset/
└── requirements.txt
```

## 📊 Dataset Structure

We have provided a subset of the dataset for you to use. This subset has been provided with the intention of allowing you to implement and test your code locally. The subset follows the same structure as the original dataset and is organized as follows:

```
hw4_data_subset/
├── hw4p1_data/ # For causal language modeling
│ ├── train/
│ ├── valid/
│ └── test/
└── hw4p2_data/ # For end-to-end speech recognition
├── dev-clean/
│ ├── fbank/
│ └── text/
├── test-clean/
│ └── fbank/
└── train-clean-100/
├── fbank/
└── text/

```

## 🔧 Implementation Files

### Main Library (`hw4lib/`)

For `HW4P1` and `HW4P2`, you will incrementally implement components of `hw4lib` to build and train two models:

- **HW4P1**: A _Decoder-only Transformer_ for causal language modeling.
- **HW4P2**: An _Encoder-Decoder Transformer_ for end-to-end speech recognition.

Many of the components you implement will be reusable across both parts, reinforcing modular design and efficient implementation. You should see the following files in the `hw4lib/` directory (`__init__.py`'s are not shown):

```
hw4lib/
├── data/
│ ├── tokenizer_jsons/
│ ├── asr_dataset.py
│ ├── lm_dataset.py
│ └── tokenizer.py
├── decoding/
│ └── sequence_generator.py
├── model/
│ ├── masks.py
│ ├── positional_encoding.py
│ ├── speech_embedding.py
│ ├── sublayers.py
│ ├── decoder_layers.py
│ ├── encoder_layers.py
│ └── transformers.py
├── trainers/
│  ├── base_trainer.py
│  ├── asr_trainer.py
│  └── lm_trainer.py
└── utils/
   ├── create_lr_scheduler.py
   └── create_optimizer.py
```

### MyTorch Library Components (`mytorch/`)

In `HW4P1` and `HW4P2`, you will build and train Transformer models using PyTorch’s `nn.MultiHeadAttention`. To deepen your understanding of its internals, you will also implement a custom `MultiHeadAttention` module from scratch as part of your `mytorch` library, designed to closely match the PyTorch interface. You should see the following files in the `mytorch/` directory:

```

mytorch/nn/
├── activation.py
├── linear.py
├── scaled_dot_product_attention.py
└── multi_head_attention.py

```

### Test Suite (`tests/`)

In `HW4P1` and `HW4P2`, you will be provided with a test suite that will be used to test your implementation. You should see the following files in the `tests/` directory:

```

tests/
├── testing_framework.py
├── test_mytorch*.py
├── test_dataset*.py
├── test_mask*.py
├── test_positional_encoding.py
├── test_sublayers*.py
├── test_encoderlayers*.py
├── test_decoderlayers*.py
├── test_transformers\*.py
├── test_hw4p1.py
└── test_decoding.py

```

## Setup

Follow the setup instructions based on your preferred environment!

### Local

One of our key goals in designing this assignment is to allow you to complete most of the preliminary implementation work locally.  
We highly recommend that you **pass all tests locally** using the provided `hw4_data_subset` before moving to a GPU runtime.  
To do this, simply:

#### Step 1: Create a new conda environment

```bash
# Be sure to deactivate any active environments first
conda create -n hw4 python=3.12.4
```

#### Step 2: Activate the conda environment

```bash
conda activate hw4
```

#### Step 3: Install the dependencies using the provided `requirements.txt`

```bash
pip install --no-cache-dir --ignore-installed -r requirements.txt
```

#### Step 4: Ensure that your notebook is in the same working directory as the `Handout`

This can be achieved by:

1. Physically moving the notebook into the handout directory.
2. Changing the notebook’s current working directory to the handout directory using the os.chdir() function.

#### Step 5: Open the notebook and select the newly created environment from the kernel selector.

If everything was done correctly, You should see atleast the following files in your current working directory after running `!ls`:

```
.
├── README.md
├── requirements.txt
├── hw4lib/
├── mytorch/
├── tests/
└── hw4_data_subset/
```

### Colab

#### Step 1: Get your handout

- See writeup for recommended approaches.

##### Example: My preferred approach

```python
import os

# Settings -> Developer Settings -> Personal Access Tokens -> Token (classic)
os.environ['GITHUB_TOKEN'] = "your-token"

GITHUB_USERNAME = "your-username"
REPO_NAME = "IDL-HW4"
TOKEN = os.environ.get("GITHUB_TOKEN")
repo_url = f"https://{TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
!git clone {repo_url}

## -------------

# To pull latest changes (Must be in the repo dir, use pwd/ls to verify)
!cd {REPO_NAME} && git pull
```

#### Step 2: Install Dependencies

- `NOTE`: Your runtime will be restarted to ensure all dependencies are updated.
- `NOTE`: You will see a runtime crashed message, this was intentionally done. Simply move on to the next cell.

```python
%pip install --no-deps -r IDL-HW4/requirements.txt
import os
# NOTE: This will restart the your colab Python runtime (required)!
os.kill(os.getpid(), 9)
```

#### Step 3: Obtain Data

- `NOTE`: This process will automatically download and unzip data for both `HW4P1` and `HW4P2`.

```bash
!curl -L -o /content/s25-hw4-data.zip https://www.kaggle.com/api/v1/datasets/download/cmu11785/s25-hw4-data
!unzip -q -o /content/s25-hw4-data.zip -d /content/hw4_data
!rm -rf /content/s25-hw4-data.zip
!du -h --max-depth=2 /content/hw4_data
```

#### Step 4: Move to Handout Directory

```python
import os
os.chdir('IDL-HW4')
!ls
```

You must be within the handout directory for the library imports to work!

- `NOTE`: You may have to repeat running this command anytime you restart your runtime.
- `NOTE`: You can do a `pwd` to check if you are in the right directory.
- `NOTE`: The way it is setup currently, Your data directory should be one level up from your project directory. Keep this in mind when you are setting your `root` in the config file.

If everything was done correctly, You should see atleast the following files in your current working directory after running `!ls`:

```
.
├── README.md
├── requirements.txt
├── hw4lib/
├── mytorch/
├── tests/
└── hw4_data_subset/
```

### Kaggle

While it is possible to run the notebook on Kaggle, we would recommend against it. This assignment is more resource intensive and may run slower on Kaggle.

#### Step 1: Get your handout

- See writeup for recommended approaches.

##### Example: My preferred approach

```python
import os

# Settings -> Developer Settings -> Personal Access Tokens -> Token (classic)
os.environ['GITHUB_TOKEN'] = "your-token"

GITHUB_USERNAME = "your-username"
REPO_NAME = "IDL-HW4"
TOKEN = os.environ.get("GITHUB_TOKEN")
repo_url = f"https://{TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
!git clone {repo_url}

## -------------

# To pull latest changes (Must be in the repo dir, use pwd/ls to verify)
!cd {REPO_NAME} && git pull
```

#### Step 2: Install Dependencies

Simply set the `Environment` setting in the notebook to `Always use latest environment`. No need to install anything.

#### Step 3: Obtain Data

##### ⚠️ Important: Kaggle Users

If you are using Kaggle, **do not manually download the data!** The dataset is large and may exceed your available disk space. Instead, follow these steps to add the dataset directly to your notebook:

1. Open your **Kaggle Notebook**.
2. Navigate to **Notebook → Input**.
3. Click **Add Input**.
4. In the search bar, paste the following URL:

👉 [https://www.kaggle.com/datasets/cmu11785/s25-hw4-data](https://www.kaggle.com/datasets/cmu11785/s25-hw4-data)

5. Click the **➕ (plus sign)** to add the dataset to your notebook.

##### 📌 Note:

This process will automatically download and unzip data for both `HW4P1` and `HW4P2`.

#### Step 4: Move to Handout Directory

```python
import os
os.chdir('IDL-HW4')
!ls
```

You must be within the handout directory for the library imports to work!

- `NOTE`: You may have to repeat running this command anytime you restart your runtime.
- `NOTE`: You can do a `pwd` to check if you are in the right directory.
- `NOTE`: The way it is setup currently, Your data directory should be one level up from your project directory. Keep this in mind when you are setting your `root` in the config file.

If everything was done correctly, You should see atleast the following files in your current working directory after running `!ls`:

```
.
├── README.md
├── requirements.txt
├── hw4lib/
├── mytorch/
├── tests/
└── hw4_data_subset/
```

### PSC

#### Step 1: Get your handout

- See writeup for recommended approaches.
- If you use Remote - SSH to connect to Bridges2, you can upload the handout to your project directory and work from there.

##### Example: My preferred approach

```python
import os

# Settings -> Developer Settings -> Personal Access Tokens -> Token (classic)
os.environ['GITHUB_TOKEN'] = "your-token"

GITHUB_USERNAME = "your-username"
REPO_NAME = "IDL-HW4"
TOKEN = os.environ.get("GITHUB_TOKEN")
repo_url = f"https://{TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
!git clone {repo_url}

## -------------

# To pull latest changes (Must be in the repo dir, use pwd/ls to verify)
!cd {REPO_NAME} && git pull
```

#### Step 2: Setting Up Your Environment on Bridges2

For this homework, we are providing a shared Conda environment for the entire class. Follow these steps to set up the environment and start a Jupyter notebook on Bridges2:

##### 1. SSH into Bridges2

```bash
ssh username@bridges2.psc.edu
```

##### 2. Navigate to your Project Directory

```bash
cd $PROJECT
```

##### 3. Load the Anaconda Module

```bash
module load anaconda3
```

##### 4. Activate the provided HW4 Environment

```bash
# First, deactivate any existing Conda environment
conda deactivate
conda activate /jet/home/psamal/hw_envs/idl_hw4
```

##### 5. Request a Compute Node

```bash
interact -p GPU-shared --gres=gpu:v100-32:1 -t 8:00:00
```

##### 6. Re-activate Environment

If your Conda environment was deactivated due to node allocation:

```bash
# First, deactivate any existing Conda environment
conda deactivate
conda activate /jet/home/psamal/hw_envs/idl_hw4
```

##### 7. Start Jupyter Notebook

```bash
jupyter notebook --no-browser --ip=0.0.0.0
```

##### 8. Connect to Jupyter Server

You can now use your prefered way of connecting to the Jupyter Server. Your options should be covered in the docs linked in post 558 @ piazza.

The following is my preferred way of connecting to the Jupyter Server:

###### 8.1 Connect in VSCode

I prefer uploading the notebook to PSC Bridges2 storage ($PROJECT directory) and then connecting to the Jupyter Server from there.

1. Use Remote - SSH to connect to Bridges2 and navigate to your project directory.

2. Upload the notebook to the project directory.

3. Open the notebook in VSCode.

4. Go to **Kernel** → **Select Another Kernel** → **Existing Jupyter Server**

5. Enter the URL of the Jupyter Server:`http://{hostname}:{port}/tree?token={token}`

- eg: `http://v011.ib.bridges2.psc.edu:8888/tree?token=e4b302434e68990f28bc2b4ae8d216eb87eecb7090526249`

> **Note**: Replace `{hostname}`, `{port}` and `{token}` with your actual values from the Jupyter output.

#### Step 3: Get Data

- `NOTE`: This will download and unzip data for both `HW4P1` and `HW4P2`
- `NOTE`: We are using `$LOCAL`: the scratch storage on local disk on the node running a job to store out data.
  - Disk accesses are much faster than what you would get from `$PROJECT` storage
  - `IT IS NOT PERSISTENT`
- `NOTE`: Make sure you have completed the previous steps before running this cell.
- Read more about it PSC File Spaces [here](https://www.psc.edu/resources/bridges-2/user-guide#file-spaces).

```bash
!curl -L -o $LOCAL/s25-hw4-data.zip https://www.kaggle.com/api/v1/datasets/download/cmu11785/s25-hw4-data
!unzip -q -o $LOCAL/s25-hw4-data.zip -d $LOCAL/hw4_data
!rm -rf $LOCAL/s25-hw4-data.zip
!du -h --max-depth=2 $LOCAL/hw4_data
```

#### Step 4: Move to Handout Directory

Depending on the way you are running your notebook, you may or may not need to run this cell. As long as you are within the handout directory for the library imports to work!

```python
# Move to the handout directory if you are not there already
import os
os.chdir('IDL-HW4')
!ls
```

- `NOTE`: You may have to repeat running this command anytime you restart your runtime.
- `NOTE`: You can do a `pwd` to check if you are in the right directory.
- `NOTE`: The way it is setup currently, Your data directory should be one level up from your project directory. Keep this in mind when you are setting your `root` in the config file.

If everything was done correctly, You should see atleast the following files in your current working directory after running `!ls`:

```
.
├── README.md
├── requirements.txt
├── hw4lib/
├── mytorch/
├── tests/
└── hw4_data_subset/
```

---

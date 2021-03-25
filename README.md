## Speech-to-text models 

Pytorch implementation and comparison `speech-to-text` (STT) models.

References:
- [Deep Speech: Scaling up end-to-end speech recognition](https://arxiv.org/abs/1412.5567)
- [Wav2Letter](https://arxiv.org/abs/1609.03193) (WIP)
- [Jasper: An End-to-End Convolutional Neural Acoustic Model](https://arxiv.org/abs/1904.03288) (WIP)

## Run on CPU/GPU
    python main.py \
    --num-workers 1 \
    --batch-size 8 \
    --num-epochs 10

## Run on TPU (not recommended since TPU needs to recompile the RNN graph for each training example)

### Create a Google Cloud [project](https://cloud.google.com/resource-manager/docs/creating-managing-projects)

#### Install `gcloud` sdk on Mac OS
    wget https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-332.0.0-darwin-x86_64.tar.gz
    tar -xf google-cloud-sdk-332.0.0-darwin-x86_64.tar.gz
    ./google-cloud-sdk/install.sh

#### Run `gcloud init` to initialize the SDK:

#### Create a new project:
    gcloud projects create <PROJECT_ID>

#### Listing projects
    gcloud projects list

#### Shutting down projects
    gcloud projects delete <PROJECT_ID>

### Turn on the [Cloud TPU API](https://console.cloud.google.com/apis/library/tpu.googleapis.com) for that project.

#### Set up compute zone
    gcloud config set compute/zone <your-zone-here> --project <PROJECT_ID>

### Set up a Compute Engine [instance](https://cloud.google.com/tpu/docs/tutorials/pytorch-dlrm#set_up_a_instance)

#### Open a Cloud Shell window
    https://console.cloud.google.com/?cloudshell=true

#### Create a variable for your project's ID
    export PROJECT_ID=<project-id>

#### Configure gcloud command-line tool to use the project where you want to create Cloud TPU.
    gcloud config set project ${PROJECT_ID}

#### From the Cloud Shell, launch the Compute Engine resource
    gcloud compute instances create deepspeech-xla \
    --zone=europe-west4-a \
    --machine-type=n1-highmem-16 \
    --image-family=torch-xla \
    --image-project=ml-images  \
    --boot-disk-size=200GB \
    --scopes=https://www.googleapis.com/auth/cloud-platform

#### Connect to the new Compute Engine instance
    gcloud compute ssh deepspeech-xla --zone=europe-west4-a

#### Launch a Cloud TPU resource
    gcloud compute tpus create deepspeech-xla \
    --zone=europe-west4-a \
    --network=default \
    --version=pytorch-1.8  \
    --accelerator-type=v3-8

#### Identify the IP address for the Cloud TPU resource
    gcloud compute tpus list --zone=europe-west4-a

### Create and configure the PyTorch environment

#### Start a conda environment.
    conda activate torch-xla-1.8

#### Configure environmental variables for the Cloud TPU resource
    export TPU_IP_ADDRESS=<ip-address>
    export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

#### Get TPU compatible model by running:
    git clone https://github.com/discort/stt_models.git

#### Install dependencies
    pip install -r requirements.txt

#### Run the model
    python main.py \
    --use-tpu 1 \
    --world-size 1 \
    --num-workers 1 \
    --batch-size 128 \
    --num-epochs 10

#### List TPU Node+VM
    gcloud compute tpus list --zone=europe-west4-a
    gcloud compute instances list

#### To delete both TPU and the VM run
    gcloud compute tpus delete deepspeech-xla --zone=europe-west4-a
    gcloud compute instances delete deepspeech-xla --zone=europe-west4-a
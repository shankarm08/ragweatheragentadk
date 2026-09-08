# ragweatheragentadk
login to cloud

pip install google-cloud-aiplatform

Authentication

gcloud auth application-default login

gcloud auth login

gcloud config set project YOUR_PROJECT_ID

to set the location

gcloud config set ai/region us-central1

command to create bucket

gsutil mb -l us-central1 gs://your-bucket-name/

list the bucket
gsutil ls
gsutil ls gs://bucketname/

#Enable billing for the current project

gcloud billing projects link my-rag-project-12345 \
    --billing-account=BILLING_ACCOUNT_ID


Enable vertex ai
gcloud services enable aiplatform.googleapis.com

#alternative command to create
gcloud storage buckets create gs://my-rag-bucket-98231 --location=us-central1

#upload the file in the bucket

gsutil cp *.txt gs://my-rag-bucket-98231/

to check the file in the bucket

AgentChroma



pip install chromadb google-cloud-aiplatform google-cloud-storage google-cloud-speech google-cloud-texttospeech sounddevice scipy


authenticate

gcloud auth application-default login

set the project

gcloud config set project rag-project-495007

enable apis

gcloud services enable aiplatform.googleapis.com

gcloud services enable storage.googleapis.com

gcloud services enable speech.googleapis.com


gcloud services enable texttospeech.googleapis.com


gcloud storage ls  gs://my-rag-bucket-98231/

**AGENTCV** 

pip install opencv-python chromadb google-cloud-aiplatform google-cloud-storage google-cloud-speech google-cloud-texttospeech pytesseract sounddevice scipy

gcloud auth application-default login












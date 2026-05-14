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
Enable vertex ai
gcloud services enable aiplatform.googleapis.com

#alternative command to create
gcloud storage buckets create gs://my-rag-bucket-98231 --location=us-central1

#upload the file in the bucket

gsutil cp *.txt gs://my-rag-bucket-98231/

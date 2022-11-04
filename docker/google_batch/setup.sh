source config.sh

gcloud config set project $PROJECT_ID

gcloud services enable \
  artifactregistry.googleapis.com \
  batch.googleapis.com \
  cloudbuild.googleapis.com \
  workflowexecutions.googleapis.com \
  workflows.googleapis.com

echo "Create a repository for containers"
gcloud artifacts repositories create containers --repository-format=docker --location=$REGION

echo "Build the container"
gcloud builds submit -t $REGION-docker.pkg.dev/$PROJECT_ID/containers/moleculekit-service:v1 recipe/

echo "Create a service account: $SERVICE_ACCOUNT for Workflows"
gcloud iam service-accounts create $SERVICE_ACCOUNT

echo "Add necessary roles to the service account"
# Needed for Workflows to create Jobs
# See https://cloud.google.com/batch/docs/release-notes#October_03_2022
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member serviceAccount:$SERVICE_ACCOUNT@$PROJECT_ID.iam.gserviceaccount.com \
    --role roles/batch.jobsEditor

# Needed for Workflows to submit Jobs
# See https://cloud.google.com/batch/docs/release-notes#October_03_2022
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member serviceAccount:$SERVICE_ACCOUNT@$PROJECT_ID.iam.gserviceaccount.com \
    --role roles/iam.serviceAccountUser

# Need for Workflows to log
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member serviceAccount:$SERVICE_ACCOUNT@$PROJECT_ID.iam.gserviceaccount.com \
    --role roles/logging.logWriter

# Needed for Workflows to create buckets
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member serviceAccount:$SERVICE_ACCOUNT@$PROJECT_ID.iam.gserviceaccount.com \
    --role roles/storage.admin
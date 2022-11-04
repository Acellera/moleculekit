source config.sh

echo "Transfering test data to a bucket"
gsutil mb -p $PROJECT_ID -b on -l $REGION gs://$BUCKET_NAME
gsutil cp -R test gs://$BUCKET_NAME

echo "Deploy the workflow: $WORKFLOW_NAME"
gcloud workflows deploy $WORKFLOW_NAME \
  --source workflow.yaml \
  --service-account=$SERVICE_ACCOUNT@$PROJECT_ID.iam.gserviceaccount.com \
  --location=$REGION

echo "Execute the workflow: $WORKFLOW_NAME"
gcloud workflows execute $WORKFLOW_NAME --location=$REGION

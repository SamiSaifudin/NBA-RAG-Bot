import boto3
import os
from dotenv import load_dotenv

load_dotenv()


s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

def upload_csv():
    bucket = os.getenv('S3_BUCKET_NAME')
    local_path = 'backend/data_pipeline/box_scores_2025_26.csv'
    s3_key = 'box_scores_2025_26.csv'
    
    s3.upload_file(local_path, bucket, s3_key)
    print(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")

if __name__ == "__main__":
    upload_csv()
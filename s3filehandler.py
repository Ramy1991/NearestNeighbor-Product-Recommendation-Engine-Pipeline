"""
AWS S3 File Handler Module

This module provides functionality to interact with AWS S3 buckets for data processing operations.
It handles file downloads, uploads, and backup operations for ML pipeline data.
"""

import boto3
import pandas as pd
from io import StringIO
import logging

# Initialize S3 client
s3 = boto3.client('s3')


class S3FileHandler:
    """
    A class to handle S3 file operations including downloading, uploading, and backing up files.
    
    Attributes:
        ingest_bucket (str): Primary S3 bucket for input data
        folder_name (str): Input folder path in the ingest bucket
        local_file_path (str): Local directory path for temporary file storage
        backup_bucket (str): S3 bucket for backing up processed files
        backup_folder (str): Folder path in backup bucket
        upload_bucket (str): S3 bucket for output data
        upload_folder (str): Output folder path
        file_df (pd.DataFrame): DataFrame to store combined input data
    """

    def __init__(self) -> None:
        # Configure S3 bucket and folder paths
        self.ingest_bucket = 'ml-v1'
        self.folder_name = 'input/ingest/'
        
        # Local storage path for downloaded files
        self.local_file_path = f'/home/ec2-user/ml/datasets/input/'
        
        # Backup storage configuration
        self.backup_bucket = 'ml-backup'
        self.backup_folder = 'backup_input/'
        
        # Output storage configuration
        self.upload_bucket = 'ml-v1'
        self.upload_folder = 'output/'
        
        # Initialize empty DataFrame for storing file contents
        self.file_df = pd.DataFrame()

    def get_files(self) -> pd.DataFrame:
        """
        Retrieves CSV files from S3, processes them, and creates a backup.
        
        Process:
        1. Lists all files in the input folder
        2. Downloads each CSV file locally
        3. Combines all CSV data into a single DataFrame
        4. Creates a backup of processed files
        
        Returns:
            pd.DataFrame: Combined data from all processed CSV files
            
        Raises:
            Exception: If no files found or any processing error occurs
        """
        try:
            # List objects in the input folder
            response = s3.list_objects_v2(Bucket=self.ingest_bucket, Prefix=self.folder_name)

            if 'Contents' not in response:
                raise Exception({
                    'event_message': 'FAILURE', 
                    'reason': f"No files found in the folder: {self.folder_name}"
                })

            # Process each file in the folder
            for obj in response['Contents']:
                file_key = obj['Key']
                sheet_name = file_key.split("/")[-1]
                
                # Skip non-CSV files
                if not sheet_name.endswith('.csv'):
                    continue

                # Download and process the file
                sheet_path = self.local_file_path + sheet_name
                s3.download_file(self.ingest_bucket, file_key, sheet_path)

                # Read and combine data
                sheet_df = pd.read_csv(sheet_path)
                self.file_df = pd.concat([self.file_df, sheet_df], ignore_index=True)

                # Backup the processed file
                source_key = f'{self.folder_name}{sheet_name}'
                destination_key = f'{self.backup_folder}{sheet_name}'

                s3.copy_object(
                    Bucket=self.backup_bucket,
                    Key=destination_key,
                    CopySource={
                        'Bucket': self.ingest_bucket,
                        'Key': source_key
                    }
                )
                
                # Commented out deletion of source files for safety
                # s3.delete_object(
                #     Bucket = self.ingest_bucket, 
                #     Key = source_key
                # )
            return self.file_df

        except Exception as e:
            raise Exception({
                'event_message': 'FAILURE', 
                'reason': str(e)
            })

    def upload_file(self, local_file_path: str):
        """
        Uploads a local file to the configured S3 output bucket.
        
        Args:
            local_file_path (str): Path to the local file to be uploaded
            
        Returns:
            dict: Status message indicating success or failure
            
        Raises:
            Exception: If upload fails
        """
        try:
            upload_path = self.upload_folder + local_file_path.split('/')[-1]
            s3.upload_file(local_file_path, self.upload_bucket, upload_path)
            return {
                'event_message': 'SUCCESS',
                'reason': f"File uploaded successfully to {self.upload_bucket}/{upload_path}"
            }
        except Exception as e:
            raise Exception({
                'event_message': 'FAILURE', 
                'reason': str(e)
            })

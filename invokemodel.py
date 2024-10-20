"""
SageMaker Model Invocation Module

This module handles the invocation of SageMaker endpoints for ML model inference,
including authentication, batch processing, and result formatting.
"""

import time
import json
import numpy as np
import boto3
from botocore.exceptions import NoCredentialsError
import pandas as pd


class InvokeModel:
    """
    A class to handle SageMaker model invocations with proper authentication and batch processing.
    
    Attributes:
        role_arn (str): AWS IAM role ARN for authentication
        external_account_id (str): AWS account ID
        mps_mapping (dict): Marketplace ID to region code mapping
        region (str): AWS region for SageMaker endpoint
        batch_size (int): Maximum batch size for model inference
        df_chunk (pd.DataFrame): Input data chunk for processing
        img_id_list (list): List of image IDs from input data
        pt (str): Product type
        mp (str): Marketplace code
        prompt (dict): Formatted input for the model
        output_df (pd.DataFrame): Results storage
    """

    def __init__(self, df_chunk: str) -> None:
        # AWS authentication configuration
        self.role_arn = 'arn:aws:iam::123456789012:role/MyEC2Role'
        self.external_account_id = 'AWS_ACCOUNT_ID'

        # Marketplace ID mapping
        self.mps_mapping = {
            '000000': 'US',
            '111111': 'UK',
            '222222': 'ES'
        }

        # AWS and batch configuration
        self.region = 'us-east-2'
        self.batch_size = 32

        # Input data processing
        self.df_chunk = df_chunk
        self.img_id_list = df_chunk['img_id'].tolist()
        self.pt = df_chunk['product_type'].tolist()[0]
        self.mp = self.mps_mapping[df_chunk['marketplace_id'].tolist()[0].__str__()]

        # Prepare model input
        self.prompt = {
            'physical_id': self.img_id_list,
            'pt': self.pt
        }
        self.output_df = pd.DataFrame()

    def invoke_model(self) -> pd.DataFrame:
        """
        Invokes SageMaker endpoint for model inference with proper authentication.
        
        Process:
        1. Assumes IAM role for authentication
        2. Creates SageMaker client
        3. Invokes first endpoint for embeddings
        4. Processes embeddings
        5. Invokes second endpoint for neighbor finding
        6. Formats results into DataFrame
        
        Returns:
            pd.DataFrame: Processed results including neighbor information
            
        Raises:
            Exception: If batch size exceeded, authentication fails, or processing error occurs
        """
        
        # Initialize STS client for authentication
        sts_client = boto3.client('sts', region_name=self.region)

        # Validate batch size
        if self.img_id_list.__len__() > self.batch_size:
            raise Exception({
                'event_message': 'FAILURE', 
                'reason': f'the batch_size should be {str(self.batch_size)}'
            })

        try:
            # Assume IAM role
            assumed_role = sts_client.assume_role(
                RoleArn=self.role_arn,
                RoleSessionName='RoleSessionName'
            )

            # Create SageMaker client with assumed role
            sagemaker_client = boto3.client(
                'sagemaker-runtime',
                region_name=self.region,
                aws_access_key_id=assumed_role['Credentials']['AccessKeyId'],
                aws_secret_access_key=assumed_role['Credentials']['SecretAccessKey'],
                aws_session_token=assumed_role['Credentials']['SessionToken']
            )
       
            # First endpoint invocation - Get embeddings
            strt = time.time()
            response = sagemaker_client.invoke_endpoint(
                EndpointName='EndpointName', 
                Body=str(self.prompt).encode(encoding="UTF-8"), 
                ContentType="text/csv",
                Accept="application/json"
            )

            print(f'embedding scored at {time.time()-strt}')

            # Process embeddings response
            outp = response["Body"].read()
            outp = outp.decode('utf-8')
            embeddings = np.array(eval(outp)['embeddings'])

            # Prepare input for second endpoint
            input_artifact = {
                'embedding': embeddings.tolist(),
                'pt': self.pt,
                'marketplace': self.mp
            }
            print(f'post processing done at at {time.time()-strt}')

            # Second endpoint invocation - Find neighbors
            response = sagemaker_client.invoke_endpoint(
                EndpointName='EndpointName',
                Body=json.dumps(input_artifact), 
                ContentType="application/json",
                Accept="application/json"
            )

            print(f'neighbors found at {time.time()-strt}')

            # Process neighbors response
            outp = response["Body"].read()
            outp = outp.decode('utf-8')
            outp = eval(eval(outp))

            neighbor_item_ids = outp['neighbor_item_ids']
            neighbor_item_ids_distances = outp['neighbor_item_ids_distances']

        except NoCredentialsError:
            raise Exception({
                'event_message': 'FAILURE', 
                'reason': 'Credentials could not be found'
            })
        except Exception as e:
            raise Exception({
                'event_message': 'FAILURE', 
                'reason': f'An error occurred: {str(e)}'
            })
            
        # Format results into DataFrame
        data = []
        index = 0
        for n_item_ids_list, distance_list in zip(neighbor_item_ids, neighbor_item_ids_distances):
            for idx, row in self.df_chunk.iterrows():
                for n_item_id, distance in zip(n_item_ids_list, distance_list):
                    data.append({
                        'item_id': row['item_id'],
                        'marketplace_id': row['marketplace_id'],
                        'img_id': row['img_id'],
                        'product_type': row['product_type'],
                        'neighbor_item_ids': n_item_id,
                        'neighbors_dist': distance
                    })
                    index += 1

        self.output_df = pd.DataFrame(data)
        return self.output_df

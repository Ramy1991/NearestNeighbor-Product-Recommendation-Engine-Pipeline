# Import required libraries
import pandas as pd
from s3filehandler import S3FileHandler  # Custom module for S3 operations
from invokemodel import InvokeModel        # Custom module for ML model invocation
from datetime import datetime


class Main:
    """
    Main orchestrator class that coordinates the ML pipeline workflow.
    This class handles:
    - Data ingestion from S3
    - Batch processing of data
    - Model predictions
    - Result aggregation and storage
    - Upload of final results back to S3
    """
    
    def __init__(self) -> None:
        # List of product types to be processed
        self.test_pts = ['FLAT_SHEET']
        
        # DataFrame to store combined results from all batches
        self.output_combined = pd.DataFrame()
        
        # Local path where output files will be saved
        self.output_path = f'/home/ec2-user/ml/datasets/output/'
        
        # Counters and configuration
        self.total_rows = 0          # Track total processed rows
        self.batch_size = 32         # Number of items to process in each batch
        self.file_path = ''          # Path to store the final output file
        
        # Dictionary mapping marketplace IDs to country codes
        self.mps_mapping = {
            '000000': 'US',
            '111111': 'UK',
            '222222': 'ES'
        }

    def get_predictions(self) -> str:
        """
        Main processing method that:
        1. Retrieves data from S3
        2. Splits data by product type and marketplace
        3. Processes data in batches through the ML model
        4. Aggregates results and saves to CSV
        
        Returns:
            str: Path to the output file containing all predictions
            
        Raises:
            Exception: If any step in the prediction pipeline fails
        """
        try:
            # Fetch input data from S3 bucket
            df = S3FileHandler().get_files()
            
            # Handle case where S3 operation returns error message
            if type(df) == dict:
                return df

            # Process each unique product type in the dataset
            for pt in df['product_type'].unique():
                # Filter data for current product type
                df_pt = df[df['product_type'] == pt]
                print(pt, df_pt.shape)

                # Process each marketplace separately within the product type
                for mp_id in df_pt['marketplace_id'].unique():
                    # Filter data for current marketplace
                    df_pt_mp = df_pt[df_pt['marketplace_id'] == mp_id]
                    num_rows = df_pt_mp.shape[0]

                    # Calculate number of full batches and remaining rows
                    num_splits = num_rows // self.batch_size
                    remaining_rows = num_rows % self.batch_size

                    # Split data into batches of specified size
                    dfs = [df_pt_mp.iloc[i*self.batch_size:(i+1)*self.batch_size] 
                          for i in range(num_splits)]

                    # Handle remaining rows that don't fill a complete batch
                    if remaining_rows > 0:
                        dfs.append(df_pt_mp.iloc[-remaining_rows:])

                    # Process each batch through the ML model
                    for df_chunk in dfs:
                        # Create model instance and get predictions
                        model_obj = InvokeModel(df_chunk)
                        output = model_obj.invoke_model()
                        
                        # Handle both successful predictions and errors
                        if type(output) == dict:
                            # If error occurred, add error status to chunk
                            df_chunk['ErroStatus'] = output['reason']
                            self.output_combined = pd.concat([self.output_combined, df_chunk], 
                                                           ignore_index=True)
                        else:
                            # Combine successful predictions with previous results
                            self.output_combined = pd.concat([self.output_combined, output], 
                                                           ignore_index=True)
                        print(self.output_combined.shape)

            print(self.output_combined.shape)

            # Generate output filename with current date
            formatted_date = datetime.now().strftime("%d-%m-%Y")
            self.file_path = f'{self.output_path}output-{str(formatted_date)}.csv'
            
            # Save combined results to CSV file
            self.output_combined.to_csv(self.file_path, index=False)
            return self.file_path
        
        except Exception as e:
            # Propagate any errors with failure status
            raise Exception({
                'event_message': 'FAILURE', 'reason': str(e)
            })
    
    def upload_s3(self) -> dict:
        """
        Uploads the final output file to S3 storage.
        
        Returns:
            dict: Status message indicating success or failure of upload
        """
        status = S3FileHandler().upload_file(self.file_path)
        return status
      
    def execute(self) -> dict:
        """
        Main execution method that orchestrates the entire pipeline:
        1. Generate predictions
        2. Upload results to S3
        
        Returns:
            dict: Final execution status
        """
        self.get_predictions()
        return self.upload_s3()


# Script entry point
if __name__ == "__main__":
    # Initialize pipeline
    St = Main()
    # Execute pipeline and get results
    results = St.execute()
    # Print final status
    print(results)

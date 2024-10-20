# ML Pipeline for Product Recommendations

## Overview
This project implements an automated ML pipeline for generating product recommendations using AWS SageMaker. The pipeline downloads input data from S3, processes it in batches, generates embeddings and neighbor recommendations using ML models, and uploads the results back to S3.

## Architecture
The pipeline consists of three main components:
1. **S3 File Handler** (`s3filehandler.py`): Manages all S3 operations including downloading input files and uploading results
2. **Model Invoker** (`invokemodel.py`): Handles the ML model invocations using AWS SageMaker endpoints
3. **Main Pipeline** (`main.py`): Orchestrates the entire process and manages batch processing

## Prerequisites
- AWS credentials with appropriate permissions
- Python 3.x
- Required Python packages:
  - boto3
  - pandas
  - numpy
  - json

## Configuration
The following configurations need to be set:
- AWS role ARN and account ID in `invokemodel.py`
- S3 bucket names and folder paths in `s3filehandler.py`
- Marketplace mappings and batch size in `main.py`

## Installation
```bash
# Clone the repository
git clone https://github.com/Ramy1991/NearestNeighbor-Product-Recommendation-Engine-Pipeline.git

# Install required packages
pip install boto3 pandas numpy
```

## Usage
1. Ensure your input CSV files are uploaded to the configured S3 input bucket
2. Run the pipeline:
```bash
python main.py
```

## Input Data Format
The input CSV files should contain the following columns:
- `img_id` (Physical ID)
- `product_type`
- `marketplace_id`
- `item_id`

## Output
The pipeline generates a CSV file with the following columns:
- `item_id`
- `marketplace_id`
- `img_id`
- `product_type`
- `neighbor_item_id`
- `neighbors_dist`

## Process Flow
1. Downloads CSV files from S3 input bucket
2. Creates backup of input files
3. Processes data in batches (default 32 rows)
4. Generates embeddings using first SageMaker endpoint
5. Finds neighbor recommendations using second SageMaker endpoint
6. Combines results and uploads to S3 output bucket

## Error Handling
- The pipeline includes comprehensive error handling for:
  - S3 operations
  - Model invocations
  - Credential validation
  - Input data validation

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Author
Ramy Gharb

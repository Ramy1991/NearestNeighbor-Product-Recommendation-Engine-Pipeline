# ML Pipeline for Product Recommendations

## Overview
An automated machine learning pipeline built with AWS SageMaker for generating product recommendations at scale. The pipeline efficiently processes product data, generates embeddings, and finds nearest neighbor recommendations using deployed ML models.

## Key Features
- Automated batch processing of product data
- Cross-account ML model invocations
- S3-based data handling and storage
- Configurable batch sizes and marketplace mappings
- Comprehensive error handling

## Architecture
The pipeline consists of three main components:

1. **S3 File Handler** (`s3filehandler.py`)
   - Manages all S3 operations
   - Handles file downloads and uploads
   - Creates data backups

2. **Model Invoker** (`invokemodel.py`)
   - Manages SageMaker endpoint invocations
   - Handles cross-account authentication
   - Processes model responses

3. **Main Pipeline** (`main.py`)
   - Orchestrates the entire workflow
   - Manages batch processing
   - Combines and validates results

## Prerequisites
- AWS credentials with appropriate S3 and SageMaker permissions
- Python 3.x
- Required Python packages:
  ```
  boto3>=1.26.0
  pandas>=1.5.0
  numpy>=1.23.0
  ```

## Important Note
The ML models are deployed in a separate AWS account. You will need to configure the appropriate cross-account credentials and permissions in `invokemodel.py` before running the pipeline.

## Installation
```bash
# Clone the repository
git clone https://github.com/Ramy1991/NearestNeighbor-Product-Recommendation-Engine-Pipeline.git

# Navigate to project directory
cd NearestNeighbor-Product-Recommendation-Engine-Pipeline

# Install required packages
pip install -r requirements.txt
```

## Configuration
Update the following configuration files:

1. `invokemodel.py`:
   - AWS role ARN
   - Account ID
   - Cross-account credentials

2. `s3filehandler.py`:
   - Input bucket name
   - Output bucket name
   - Folder paths

3. `main.py`:
   - Marketplace mappings
   - Batch size settings

## Input Data Format
Required CSV columns:
- `img_id` (Physical ID)
- `product_type`
- `marketplace_id`
- `item_id`

## Output Format
The pipeline generates CSV files containing:
- `item_id`
- `marketplace_id`
- `img_id`
- `product_type`
- `neighbor_item_id`
- `neighbors_dist`

## Usage
1. Upload input CSV files to the configured S3 input bucket
2. Run the pipeline:
   ```bash
   python main.py
   ```

## Process Flow
1. Download CSV files from S3 input bucket
2. Create backup of input files
3. Process data in configurable batches (default: 32 rows)
4. Generate embeddings using first SageMaker endpoint
5. Find nearest neighbor recommendations using second endpoint
6. Combine results and upload to S3 output bucket

## Error Handling
The pipeline includes comprehensive error handling for:
- S3 operations
- Model invocations
- Cross-account authentication
- Credential validation
- Input data validation
- Batch processing errors

## Contributing
1. Fork the repository
2. Create your feature branch:
   ```bash
   git checkout -b feature/YourAmazingFeature
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add some amazing feature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/YourAmazingFeature
   ```
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author
Ramy Gharb

## Support
For support, please open an issue in the GitHub repository.

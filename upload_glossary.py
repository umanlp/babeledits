# %%
import argparse
from google.cloud import storage
from google.cloud import translate_v3 as translate
import argparse

# Uploading the CSV file to a Google Cloud Storage bucket
def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # Initialize a storage client
    storage_client = storage.Client()
    
    # Get the bucket
    bucket = storage_client.bucket(bucket_name)
    
    # Create a blob object from the file path
    blob = bucket.blob(destination_blob_name)
    
    # Upload the file to GCS
    blob.upload_from_filename(source_file_name)
    
    print(f"File {source_file_name} uploaded to {destination_blob_name} in bucket {bucket_name}.")

def create_glossary(
    project_id: str = "YOUR_PROJECT_ID",
    input_uri: str = "YOUR_INPUT_URI",
    glossary_id: str = "YOUR_GLOSSARY_ID",
    timeout: int = 180,
) -> translate.Glossary:
    """
    Create a equivalent term sets glossary. Glossary can be words or
    short phrases (usually fewer than five words).
    https://cloud.google.com/translate/docs/advanced/glossary#format-glossary
    """
    client = translate.TranslationServiceClient()

    # Supported language codes: https://cloud.google.com/translate/docs/languages
    source_lang_code = "en"
    target_lang_code = "it"
    location = "us-central1"  # The location of the glossary

    name = client.glossary_path(project_id, location, glossary_id)
    language_codes_set = translate.types.Glossary.LanguageCodesSet(
        language_codes=[source_lang_code, target_lang_code]
    )

    gcs_source = translate.types.GcsSource(input_uri=input_uri)

    input_config = translate.types.GlossaryInputConfig(gcs_source=gcs_source)

    glossary = translate.types.Glossary(
        name=name, language_codes_set=language_codes_set, input_config=input_config
    )

    parent = f"projects/{project_id}/locations/{location}"
    # glossary is a custom dictionary Translation API uses
    # to translate the domain-specific terminology.
    operation = client.create_glossary(parent=parent, glossary=glossary, timeout=6000)

    result = operation.result(timeout)
    print(f"Created: {result.name}")
    print(f"Input Uri: {result.input_config.gcs_source.input_uri}")

    return result


# Create the argument parser
parser = argparse.ArgumentParser(description='Upload a file to Google Cloud Storage')

# Add the command line arguments with default values
parser.add_argument('--bucket_name', default='glossary-babeledits', help='Name of the bucket')
parser.add_argument('--source_file_name', default='./translation/glossaries/glossary_no_id.csv', help='Path to the source file')
parser.add_argument('--destination_blob_name', default='glossary_no_id.csv', help='Name of the destination blob')
parser.add_argument('--project_id', default='babeledits-trial', help='Project ID')
parser.add_argument('--glossary_id', default='it_fr_v3', help='Glossary ID')

args = parser.parse_args()

# Assign the values to variables
bucket_name = args.bucket_name
source_file_name = args.source_file_name
destination_blob_name = args.destination_blob_name
project_id = args.project_id
glossary_id = args.glossary_id

# Upload the file
upload_to_gcs(bucket_name, source_file_name, destination_blob_name)

# Create the glossary
input_uri = f"gs://{bucket_name}/{source_file_name}"
create_glossary(project_id, input_uri, glossary_id)

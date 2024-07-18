# %%
import argparse
from google.cloud import storage
from google.cloud import translate_v3 as translate
import pandas as pd
from utils import upload_to_gcs

# Uploading the CSV file to a Google Cloud Storage bucket


def create_glossary(
    project_id: str = "YOUR_PROJECT_ID",
    input_uri: str = "YOUR_INPUT_URI",
    glossary_id: str = "YOUR_GLOSSARY_ID",
    timeout: int = 180,
    languages: list = ["en", "it"],
) -> translate.Glossary:
    """
    Create a equivalent term sets glossary. Glossary can be words or
    short phrases (usually fewer than five words).
    https://cloud.google.com/translate/docs/advanced/glossary#format-glossary
    """
    print(
        f"Creating glossary {glossary_id} for project {project_id}. Input URI: {input_uri}"
    )
    client = translate.TranslationServiceClient()

    # Supported language codes: https://cloud.google.com/translate/docs/languages
    location = "us-central1"  # The location of the glossary

    print(f"Using location {location}")
    name = client.glossary_path(project_id, location, glossary_id)
    language_codes_set = translate.types.Glossary.LanguageCodesSet(
        language_codes=languages
    )
    print(f"Creating glossary with language codes set:\n{language_codes_set}")

    gcs_source = translate.types.GcsSource(input_uri=input_uri)

    input_config = translate.types.GlossaryInputConfig(gcs_source=gcs_source)

    glossary = translate.types.Glossary(
        name=name, language_codes_set=language_codes_set, input_config=input_config
    )

    parent = f"projects/{project_id}/locations/{location}"
    # glossary is a custom dictionary Translation API uses
    # to translate the domain-specific terminology.
    operation = client.create_glossary(parent=parent, glossary=glossary)

    result = operation.result(timeout)
    print(f"Created: {result.name}")
    print(f"Input Uri: {result.input_config.gcs_source.input_uri}")

    return result


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="Upload a file to Google Cloud Storage"
    )

    # Add the command line arguments with default values
    parser.add_argument(
        "--bucket_name", default="glossary-babeledits", help="Name of the bucket"
    )
    parser.add_argument(
        "--source_file_name",
        default="./translation/glossaries/v4/glossary_no_id.csv",
        help="Path to the source file",
    )
    parser.add_argument(
        "--destination_blob_name",
        default="glossaries/v4/glossary_no_id_v4.csv",
        help="Name of the destination blob",
    )
    parser.add_argument("--project_id", default="babeledits-trial", help="Project ID")
    parser.add_argument("--glossary_id", default="multi_v4", help="Glossary ID")

    args, _ = parser.parse_known_args()

    # Assign the values to variables
    bucket_name = args.bucket_name
    source_file_name = args.source_file_name
    destination_blob_name = args.destination_blob_name
    project_id = args.project_id
    glossary_id = args.glossary_id

    print(args)
    print(f"Loading the glossary from {source_file_name}")
    glossary_df = pd.read_csv(source_file_name)
    languages = glossary_df.columns
    print(f"Languages: {languages}")
    print(f"Glossary contains {len(glossary_df)} synsets")
    print(glossary_df.head())

    # Upload the file
    upload_to_gcs(bucket_name, source_file_name, destination_blob_name)

    # Create the glossary
    input_uri = f"gs://{bucket_name}/{destination_blob_name}"
    create_glossary(
        project_id,
        input_uri,
        glossary_id,
        timeout=100000,
        languages=languages,
    )

# %%

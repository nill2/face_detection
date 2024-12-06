"""This it a config file that contains all the secrets for the app."""

# config.py
import os
import subprocess

# FTP server configuration

FTP_ROOT = os.getenv("FTP_ROOT", os.path.expanduser("~"))
print("FTP_ROOT: " + FTP_ROOT)

HISTORY = os.getenv("HISTORY", "24")
USE_S3 = os.getenv("USE_S3", "false")

FTP_PORT_STR = os.getenv("FTP_PORT", "")
ERROR_LVL = "debug"
MONGO_DB = "nill-test"
FTP_USER = "user"
FTP_PASSWORD = "password"
IS_TEST = ""
HOURS_KEEP = int(HISTORY)  # time in hours we should keep photos
FACES_HISTORY_DAYS = int(os.getenv("FACES_HISTORY", "30"))

# Check if the string is empty or not
if FTP_PORT_STR:
    # Convert the string to an integer if it's not empty
    FTP_PORT = int(FTP_PORT_STR)
else:
    # Use a default value (e.g., 2121) if the string is empty or not set
    FTP_PORT = 2121
FTP_HOST = "0.0.0.0"


# MongoDB configuration
MONGO_HOST = "localhost"
if "MONGO_HOST" in os.environ:
    MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
    print("Getting from env vars" + MONGO_HOST)
else:
    try:
        # Run the vlt command and capture its output
        VLT_COMMAND = "vlt secrets get --plaintext MONGO_HOST"
        MONGO_HOST = subprocess.check_output(VLT_COMMAND, shell=True, text=True)
        print("Value from hashicorp for MONGO_HOST: " + str(MONGO_HOST))
    except subprocess.CalledProcessError as hashi_e:
        # Handle errors, e.g., if the secret does not exist
        print(f"Error: {hashi_e}")

# Check if the code is running in a test environment
if "IS_TEST" in os.environ:
    # Use localhost address for tests
    if os.environ.get("IS_TEST") == "GHA":
        MONGO_HOST = "localhost"
        print("You are running in a GHA test environment: " + MONGO_HOST)
    else:
        if os.environ.get("IS_TEST") == "prod":
            MONGO_DB = "nill-home"
            ERROR_LVL = "debug"
            FTP_USER = os.getenv("FTP_USER", "user")
            FTP_PASSWORD = os.getenv("FTP_PASSWORD", "password")
            print("You are running in a prod environment: " + MONGO_HOST)
        else:
            print("You are running in a local test environment: " + MONGO_HOST)
            ERROR_LVL = "debug"
            MONGO_DB = "nill-test"
            FTP_USER = "user"
            FTP_PASSWORD = "password"
else:
    MONGO_DB = "nill-home"
    ERROR_LVL = "production"
    FTP_USER = os.getenv("FTP_USER", "user")
    FTP_PASSWORD = os.getenv("FTP_PASSWORD", "password")
    print("You are running in a prod environment: " + MONGO_HOST)
MONGO_PORT = 27017
MONGO_COLLECTION = "nill-home-photos"
FACE_COLLECTION = "nill-home-faces"
FTP_PASSIVE_PORT_FROM = 52000
FTP_PASSIVE_PORT_TO = 52003
PYROSCOPE_SERVER_ADDRESS = "http://10.0.0.225:4040"
if "PYROSCOPE_SERVER_ADDRESS" in os.environ:
    PYROSCOPE_SERVER_ADDRESS = os.getenv("PYROSCOPE_SERVER_ADDRESS", "None")
    print("Getting PYROSCOPE_SERVER_ADDRESS from env vars: " + PYROSCOPE_SERVER_ADDRESS)
else:
    try:
        # Run the vlt command and capture its output
        VLT_COMMAND = "vlt secrets get --plaintext PYROSCOPE_SERVER_ADDRESS"
        PYROSCOPE_SERVER_ADDRESS = subprocess.check_output(
            VLT_COMMAND, shell=True, text=True
        )
        print("Value from hashicorp for MONGO_HOST: " + str(PYROSCOPE_SERVER_ADDRESS))
    except subprocess.CalledProcessError as hashi_e:
        # Handle errors, e.g., if the secret does not exist
        print(f"Error: {hashi_e}")

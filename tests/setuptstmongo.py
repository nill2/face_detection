'''
Script to create a test user with read and write access, test db and collection in Mongodb
'''
import pymongo

# MongoDB connection details
MONGO_HOST = 'localhost'
MONGO_PORT = 27017
MONGO_DB = 'nill-home'
MONGO_COLLECTION = 'nill-home-photos'
MONGO_USER = 'user'
MONGO_PASSWORD = 'password'
FTP_PASSIVE_PORT_FROM = 52000
FTP_PASSIVE_PORT_TO = 52050

# Create a MongoDB client
client = pymongo.MongoClient(host=MONGO_HOST, port=MONGO_PORT)

# Create the 'nill-home' database
db = client[MONGO_DB]

# Create a test user with read and write access
try:
    db.command("createUser", MONGO_USER, pwd=MONGO_PASSWORD, roles=["readWrite", "dbAdmin"])
except pymongo.errors.OperationFailure as e:
    if "already exists" in str(e):
        print(f"User '{MONGO_COLLECTION}' already exists.")
    else:
        raise e

# Create the 'nill-home-photos' collection
collection = db[MONGO_COLLECTION]

# Close the MongoDB client when done
client.close()

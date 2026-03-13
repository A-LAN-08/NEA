
# Standard library imports
import json
import os

# External library imports
import bcrypt
from cryptography.fernet import Fernet

# Custom imports
from scripts.config import DATA_DIR

############################################################################

class Profile:
    def __init__(self, manager: 'DataManager', username: str, key: bytes, data: dict):
        self.__manager = manager; self.__username = username
        self.__key = key; self.__data = data

    # Update data file for itself
    def update_data(self, new_data: dict) -> None:
        # Strip it of illegal data values, and update self data and file
        new_data = {key: value for key,value in new_data.items() if key != "password"}
        self.__data.update(new_data)
        self.__manager.save_profile_data(self, self.__data.get("password"), self.__key)

    # Get profile data
    def get_data(self) -> dict:
        return {key: value for key,value in self.__data.items() if key != "password"}

    # Get profile username
    def get_username(self) -> str:
        return self.__username

    # Get username AND data
    def get_full_data(self) -> dict:
        return {"username": self.__username, "data": self.get_data()}

    # Validate the password for the profile
    def validate_password(self, password: str) -> bool:
        # Find what hash was used for that password and use it to hash the new password
        stored_hash = self.__data.get("password", "").encode('utf-8')
        password_bytes = password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, stored_hash)

############################################################################

class DataManager:
    def __init__(self):
        self.__master_key = b"pCt-V4cMUKFvqrWFTD6YjoaXmioK20H3zHzXn1dfd_o=" # noqa
        self.__key_file = os.path.join(DATA_DIR, "keys.dat").replace("\\", "/")
        self.__data_file = os.path.join(DATA_DIR, "data.dat").replace("\\", "/")

        # Ensure key and data files exist, if not create them
        if not os.path.exists(self.__key_file): self.save_encrypt_file(self.__key_file, {}, self.__master_key)
        if not os.path.exists(self.__data_file): json.dump({}, open(self.__data_file, "w"))

        # {username: key, username: key ...}
        self.__keys = self.load_decrypt_file(self.__key_file, self.__master_key)
        # {username: <encrypted data> ...}
        with open(self.__data_file, "r") as f: self.__profile_datas = json.load(f)

    # Helper function to load a file and decrypt it with the key
    @staticmethod
    def load_decrypt_file(filename: str, key: bytes) -> dict:
        fernet = Fernet(key)
        with open(filename, "rb") as f: encrypted_data = f.read()

        decrypted_data = fernet.decrypt(encrypted_data)
        return json.loads(decrypted_data.decode())

    # Helper function to save a file encrypted with the key
    @staticmethod
    def save_encrypt_file(filename: str, data: dict, key: bytes) -> None:
        fernet = Fernet(key)
        json_data = json.dumps(data).encode()

        encrypted_data = fernet.encrypt(json_data)
        with open(filename, "wb") as f: f.write(encrypted_data)

    # Helper function to hash a password
    @staticmethod
    def hash_password(password: str) -> str:
        password_bytes = password.encode('utf-8') # Password string to bytes

        # Generate a "salt" to hash the password
        salt = bcrypt.gensalt(rounds=12)
        hashed_bytes = bcrypt.hashpw(password_bytes, salt)

        return hashed_bytes.decode('utf-8') # return string of hashed password

    # Return a profile if given username and password correct
    def get_profile(self, username: str, password: str) -> Profile | str:
        # Get key for correct profile
        target_key = self.__keys.get(username) # key as string
        if target_key is None: return "Non-existent profile"
        target_key = target_key.encode("utf-8")  # key string to bytes

        # Get data for correct profile
        target_data = self.__profile_datas.get(username)  # target data as string
        target_data = target_data.encode("utf-8")  # data string to bytes

        # Decrypt the data and return it
        fernet = Fernet(target_key)
        target_data = fernet.decrypt(target_data)  # decrypt target data
        target_data = json.loads(target_data.decode())  # convert back into dict

        # Ensure the entered password was correct
        stored_hash = target_data.get("password", "").encode('utf-8')
        password_bytes = password.encode('utf-8')
        if not bcrypt.checkpw(password_bytes, stored_hash):
            return "Incorrect password"

        return Profile(self, username, target_key, target_data)

    # Create a new profile for the given username and password
    def create_profile(self, username: str, password: str) -> str:
        # Ensure no duplicate being created
        if username in self.__profile_datas.keys(): return "Profile already exists"

        # Generate a new encryption key for that profile and encrypt empty data for it
        new_key = Fernet.generate_key(); fernet = Fernet(new_key)
        data = fernet.encrypt(
            json.dumps(
                {"password": self.hash_password(password), "Saved stocks": [], "Risk tolerance": 5}
            ).encode()
        ).decode("utf-8")
        self.__profile_datas[username] = data
        with open(self.__data_file, "w") as f: json.dump(self.__profile_datas, f)

        # Add key to keys dict and save file
        self.__keys[username] = new_key.decode("utf-8")
        self.save_encrypt_file(self.__key_file, self.__keys, self.__master_key)
        return "Profile created"

    # Save/update data for a profile
    def save_profile_data(self, profile: Profile, password: str, key: bytes) -> None:
        # Validate values
        if not isinstance(profile, Profile): return
        if not profile.validate_password(password): return
        # Get profile data and add back in password since it was stripped
        data_to_save = profile.get_data(); data_to_save["password"] = password

        # Save data encrypted
        fernet = Fernet(key)
        json_data = json.dumps(data_to_save).encode()
        encrypted_data = fernet.encrypt(json_data).decode("utf-8")
        self.__profile_datas[profile.get_username()] = encrypted_data
        with open(self.__data_file, "w") as f: json.dump(self.__profile_datas, f)

    # Delete a profile
    def delete_profile(self, profile: Profile, password: str) -> bool | str:
        # Validate values
        if not isinstance(profile, Profile): return False
        if not profile.validate_password(password): return "Incorrect password"

        # Remove entries for that username in keys and data
        self.__keys.pop(profile.get_username(), None)
        self.__profile_datas.pop(profile.get_username(), None)

        # Save files without removed data
        self.save_encrypt_file(self.__key_file, self.__keys, self.__master_key)
        with open(self.__data_file, "w") as f: json.dump(self.__profile_datas, f)
        return True

############################################################################



from hashlib import sha256
import os
cwd=os.getcwd()
shahash=sha256()
with open(cwd+"\dist\krispdanger12_nogate.exe","rb") as f :
    for chunk in iter(lambda:f.read(4096),b""):
        shahash.update(chunk)
hashed=shahash.hexdigest()
print(hashed) 

# jkla
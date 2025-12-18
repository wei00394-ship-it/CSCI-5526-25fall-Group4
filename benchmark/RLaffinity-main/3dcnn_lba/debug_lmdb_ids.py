import lmdb
import gzip
import io
import msgpack

def deserialize(x):
    return msgpack.unpackb(x, raw=False)

path = r'my_data\fold1_lmdb\data'
env = lmdb.open(path, readonly=True, lock=False)
txn = env.begin()
try:
    print(f"Num examples: {int(txn.get(b'num_examples').decode())}")
except Exception as e:
    print(f"Error reading num_examples: {e}")

cursor = txn.cursor()
count = 0
for key, value in cursor:
    if key in [b'num_examples', b'serialization_format', b'id_to_idx']:
        continue
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(value)) as f:
            item = deserialize(f.read())
            print(f"ID: {item['id']}")
    except Exception as e:
        print(f"Error reading item {key}: {e}")
    
    count += 1
    if count > 10:
        break
